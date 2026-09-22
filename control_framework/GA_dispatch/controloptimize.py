"""Genetic algorithm and connected-tank model for hourly pump dispatch.

Classes
-------
GeneticAlgorithm
    Discrete GA with crossover / mutation / elitist selection.
PumpControl (tank model)
    Maps DNA gene indices to pump flow rates, integrates section volumes,
    applies connected-tank redistribution near WWTP approaches, and scores
    volume / upstream-storage / switching penalties.
"""

from __future__ import annotations

import math
import random
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
import pandas as pd

from naming_local import (
    KEY_HJT,
    KEY_TJQ,
    KEY_WJT,
    KEY_XHJT,
    KEY_XL,
    SEC_HJT_XL,
    SEC_TJQ_WWTP2,
    SEC_WJT_TJQ,
    SEC_XHJT_WWTP2,
    SEC_XL_WWTP1,
)


class GeneticAlgorithm:
    def __init__(self, npop: int, dna_size: int, value_max: list[int], config: dict):
        self.config = config
        self.npop = npop
        self.dna_size = dna_size
        self.value_max = value_max
        self.crossover_rate = config["GA_setting"]["crossover_rate"]
        self.mutation_rate = config["GA_setting"]["mutation_rate"]
        self.pop = pd.DataFrame(columns=list(range(dna_size)) + ["fit"], index=range(npop))
        self.popnew = pd.DataFrame()
        self.fitness_list: list[float] = []

    def init_pop(self) -> None:
        for igene in range(self.dna_size):
            self.pop[igene] = pd.Series(
                np.random.randint(low=0, high=self.value_max[igene] + 1, size=self.npop)
            )

    def crossover(self) -> None:
        for i in range(0, self.npop, 2):
            if np.random.rand() < self.crossover_rate:
                cross_points = sorted(random.sample(range(self.dna_size), 2))
                idx1, idx2 = int(cross_points[0]), int(cross_points[1])
                child1 = self.pop.loc[i].copy()
                child1[idx1:idx2] = self.pop.loc[i + 1][idx1:idx2]
                child2 = self.pop.loc[i + 1].copy()
                child2[idx1:idx2] = self.pop.loc[i][idx1:idx2]
                self.popnew = pd.concat(
                    [self.popnew, pd.DataFrame(child1).T, pd.DataFrame(child2).T],
                    ignore_index=True,
                )

    def mutate(self) -> None:
        n_mutate = math.ceil(self.dna_size / 20)
        for i in range(self.npop):
            if np.random.rand() < self.mutation_rate:
                child1 = self.pop.loc[i].copy()
                for idx in random.sample(range(self.dna_size), n_mutate):
                    rnd = 1 if np.random.rand() >= 0.5 else -1
                    new_value = min(max(child1[idx] + rnd, 0), self.value_max[idx])
                    child1[idx] = new_value
                self.popnew = pd.concat(
                    [self.popnew, pd.DataFrame(child1).T], ignore_index=True
                )

    def add_fitness(self, fit: pd.Series) -> None:
        self.pop["fit"] = fit.copy()

    def add_fitness_new(self, fit: pd.Series) -> None:
        self.popnew["fit"] = fit.copy()

    def selection(self) -> None:
        self.pop = pd.concat([self.pop, self.popnew], ignore_index=True)
        self.pop.drop_duplicates(subset=range(self.dna_size), inplace=True)
        self.pop.sort_values(by="fit", inplace=True, ignore_index=True)
        self.pop = self.pop.loc[0 : self.npop]
        self.popnew = pd.DataFrame()
        self.fitness_list.append(float(self.pop.loc[0, "fit"]))


class PumpControl:
    """Connected-tank hydraulic surrogate used inside the GA fitness."""

    def __init__(self, config: dict):
        self.config = config
        self.section_names = [
            SEC_WJT_TJQ,
            SEC_TJQ_WWTP2,
            SEC_XHJT_WWTP2,
            SEC_HJT_XL,
            SEC_XL_WWTP1,
        ]
        # Internal short keys -> anonymized station IDs (display only)
        self.pump_keys = [KEY_WJT, KEY_XHJT, KEY_XL, KEY_TJQ, KEY_HJT]
        parallel = config.get("parallel_setting", {})
        self.USE_PARALELL = bool(parallel.get("if_parallel", False))
        self.cpu_count = int(parallel.get("cpu_count") or max(cpu_count() - 1, 1))

        ms = config["model_setting"]
        self.t_step = float(ms["t_step"])  # minutes
        self.t_total = int(ms["t_total"])  # hours
        self.total_steps = int(round(self.t_total * 60 / self.t_step))
        start_hour = int(ms["start_hour"])
        self.fix_steps = []
        for s in ms["fix_hours"]:
            delta = s - start_hour
            self.fix_steps.append(delta if delta > 0 else delta + 24)
        self.len_decvar = self.total_steps - len(self.fix_steps)

        flow = config["flow"]
        self.wwtp1_desired = float(flow["wwtp1"])
        self.wwtp2_max = float(flow["wwtp2_max"])
        self.wwtp2_min = float(flow["wwtp2_min"])

        pipe = config["pipe_setting"]
        vol_limit = pipe["volume_limit"]
        self.min_volume = pd.Series({k: v[0] for k, v in vol_limit.items()})
        self.max_volume = pd.Series({k: v[1] for k, v in vol_limit.items()})
        self.total_volume = self.max_volume.copy()
        self.max_storage = pd.Series(pipe["max_storage"])
        self.min_storage = float(pipe["min_storage"])

        self.flowratecand = pd.DataFrame(columns=self.pump_keys)
        self.topology = pd.DataFrame()
        self.flow_quantity = pd.Series(dtype=float)
        self.flow_pattern: dict[str, pd.DataFrame] = {}
        self.flow_uncontrol = pd.DataFrame()
        self.init_volume = pd.Series(dtype=float)
        self.init_storage = pd.Series(dtype=float)
        self.wwtp2_flow = 0.0
        self.wwtp1_flow = 0.0
        self.volume_record = None
        self.storage_record = None

        self.gen_pump_topology()

    def load_flow_candidates(self, cand: dict[str, list[float]]) -> None:
        """cand: {pump_key: [discrete flow options m3/h]}"""
        max_len = max(len(v) for v in cand.values())
        self.flowratecand = pd.DataFrame(index=range(max_len), columns=self.pump_keys)
        for k in self.pump_keys:
            series = pd.Series(cand[k])
            self.flowratecand[k] = series.reindex(range(max_len))

    def gen_pump_topology(self) -> None:
        """Build in/out section edges for controlled pumps and uncontrolled feeders."""
        pump_list = []
        for sec in self.section_names:
            for pump in sec.split("-"):
                # Section tokens are UPPERCASE initials; map to short keys where needed
                token = pump.lower()
                if token not in pump_list and token in (
                    KEY_WJT,
                    KEY_TJQ,
                    KEY_XHJT,
                    KEY_HJT,
                    KEY_XL,
                    "wwtp1",
                    "wwtp2",
                ):
                    pump_list.append(token)

        # Explicit controlled-pump topology using short keys
        self.topology = pd.DataFrame(columns=["in", "out"], index=self.pump_keys)
        mapping = {
            KEY_WJT: (SEC_WJT_TJQ, None),
            KEY_TJQ: (SEC_TJQ_WWTP2, SEC_WJT_TJQ),
            KEY_XHJT: (SEC_XHJT_WWTP2, None),
            KEY_HJT: (SEC_HJT_XL, None),
            KEY_XL: (SEC_XL_WWTP1, SEC_HJT_XL),
        }
        for k, (vin, vout) in mapping.items():
            self.topology.loc[k, "in"] = vin
            self.topology.loc[k, "out"] = vout

        # Uncontrolled feeder inflows into tanks
        extra = pd.DataFrame(
            {
                "in": [
                    SEC_WJT_TJQ,
                    SEC_TJQ_WWTP2,
                    SEC_HJT_XL,
                    SEC_HJT_XL,
                    SEC_XL_WWTP1,
                    SEC_XL_WWTP1,
                    SEC_XL_WWTP1,
                ],
                "out": [None] * 7,
            },
            index=["U7", "U8", "U5", "U6", "U9", "U10", "gravity"],
        )
        self.topology = pd.concat([self.topology, extra])

    def load_quantity(self, q_wwtp_daily: float, q_pump: pd.Series) -> None:
        self.flow_quantity = q_pump.copy()
        self.wwtp2_flow = min(
            self.wwtp2_max, max(self.wwtp2_min, q_wwtp_daily - self.wwtp1_desired)
        )
        self.wwtp1_flow = q_wwtp_daily - self.wwtp2_flow

    def load_pattern(self, patterns: dict[str, pd.DataFrame]) -> None:
        self.flow_pattern = patterns

    def load_extflow(self, df: pd.DataFrame) -> None:
        self.flow_uncontrol = df.copy()

    def load_init_state(self, volume: pd.Series, storage: pd.Series) -> None:
        self.init_volume = volume.reindex(self.section_names).fillna(0.0)
        self.init_storage = storage.reindex([KEY_WJT, KEY_XHJT, KEY_HJT]).fillna(0.0)

    def water_balance_wwtp(self, curr_volume: pd.Series) -> None:
        """Redistribute connected tanks feeding WWTP-1 / WWTP-2 approaches."""
        crit2 = self.total_volume[SEC_XHJT_WWTP2] + self.min_volume[SEC_TJQ_WWTP2]
        crit1 = self.min_volume[SEC_XL_WWTP1]
        total1 = curr_volume[SEC_XL_WWTP1]
        total2 = curr_volume[SEC_TJQ_WWTP2] + curr_volume[SEC_XHJT_WWTP2]

        if total2 <= crit2 and total1 <= crit1:
            if total2 < self.total_volume[SEC_XHJT_WWTP2]:
                curr_volume[SEC_XHJT_WWTP2] = total2
                curr_volume[SEC_TJQ_WWTP2] = 0.0
            else:
                curr_volume[SEC_XHJT_WWTP2] = self.total_volume[SEC_XHJT_WWTP2]
                curr_volume[SEC_TJQ_WWTP2] = total2 - self.total_volume[SEC_XHJT_WWTP2]
        elif total2 + total1 >= crit1 + crit2:
            above = (total2 + total1) - (crit1 + crit2)
            curr_volume[SEC_XHJT_WWTP2] = self.total_volume[SEC_XHJT_WWTP2]
            curr_volume[SEC_TJQ_WWTP2] = (
                crit2 - self.total_volume[SEC_XHJT_WWTP2]
            ) + above / 2.0
            curr_volume[SEC_XL_WWTP1] = crit1 + above / 2.0
        elif total2 > crit2:
            transfer = total2 - crit2
            curr_volume[SEC_XHJT_WWTP2] = self.total_volume[SEC_XHJT_WWTP2]
            curr_volume[SEC_TJQ_WWTP2] = crit2 - self.total_volume[SEC_XHJT_WWTP2]
            curr_volume[SEC_XL_WWTP1] += transfer
        elif crit1 > total1:
            transfer = crit1 - total1
            curr_volume[SEC_XL_WWTP1] = crit1
            curr_volume[SEC_TJQ_WWTP2] += transfer
            total2 = curr_volume[SEC_TJQ_WWTP2] + curr_volume[SEC_XHJT_WWTP2]
            if total2 < self.total_volume[SEC_XHJT_WWTP2]:
                curr_volume[SEC_XHJT_WWTP2] = total2
                curr_volume[SEC_TJQ_WWTP2] = 0.0
            else:
                curr_volume[SEC_XHJT_WWTP2] = self.total_volume[SEC_XHJT_WWTP2]
                curr_volume[SEC_TJQ_WWTP2] = total2 - self.total_volume[SEC_XHJT_WWTP2]

    def dna_to_flow(self, dna: pd.Series) -> pd.DataFrame:
        df_flow = pd.DataFrame(columns=self.flowratecand.columns, index=range(self.total_steps))
        for icol, pump in enumerate(df_flow.columns):
            dna_this = dna[icol * self.len_decvar : (icol + 1) * self.len_decvar]
            tem_flow = dna_this.apply(lambda x: self.flowratecand[pump][x]).values
            list_flow = [np.nan] * self.total_steps
            j = 0
            for i in range(self.total_steps):
                if i in self.fix_steps:
                    list_flow[i] = list_flow[i - 1]
                else:
                    list_flow[i] = tem_flow[j]
                    j += 1
            df_flow[pump] = list_flow
        return df_flow

    def calc_punish_1(self, df_flow: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Section volume constraint over the horizon."""
        volume_df = pd.DataFrame(index=self.init_volume.index, columns=range(self.total_steps))
        punish1 = 0.0
        curr_volume = self.init_volume.copy()
        for t in range(self.total_steps):
            for pump in self.flowratecand.columns:
                quantity = self.t_step / 60.0 * df_flow[pump][t]
                vin = self.topology["in"].get(pump)
                vout = self.topology["out"].get(pump)
                if pd.notna(vin):
                    curr_volume[vin] += quantity
                if pd.notna(vout):
                    curr_volume[vout] -= quantity

            for pump in self.flow_uncontrol.columns:
                quantity = self.t_step / 60.0 * self.flow_uncontrol[pump][t]
                curr_volume[self.topology["in"][pump]] += quantity

            curr_volume[SEC_TJQ_WWTP2] -= self.t_step / 60.0 * self.wwtp2_flow / 24.0
            curr_volume[SEC_XL_WWTP1] -= self.t_step / 60.0 * self.wwtp1_flow / 24.0
            self.water_balance_wwtp(curr_volume)

            excess = curr_volume - self.max_volume
            if (excess > 0).any():
                punish1 += float(excess[excess > 0].sum())
            deficit = self.min_volume - curr_volume
            if (deficit > 0).any():
                punish1 += float(deficit[deficit > 0].sum())
            volume_df[t] = curr_volume
        return punish1, volume_df

    def calc_punish_2(self, df_flow: pd.DataFrame, qup: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Upstream catchment storage constraint (hourly)."""
        curr_storage = self.init_storage.copy()
        punish2 = 0.0
        storage_df = pd.DataFrame(index=self.init_storage.index, columns=range(self.t_total))
        qup_cum = qup.cumsum()
        df_flow_cum = df_flow.cumsum() * self.t_step / 60.0
        for thour in range(self.t_total):
            for pump in qup.columns:
                inflow = qup_cum[pump][thour]
                nsteps = round(thour / (self.t_step / 60.0))
                nsteps = min(nsteps, self.total_steps - 1)
                outflow = df_flow_cum[pump][nsteps]
                curr_storage[pump] = self.init_storage[pump] + inflow - outflow
            surplus = curr_storage[KEY_WJT] - self.max_storage[KEY_WJT]
            if surplus > 0:
                curr_storage[KEY_WJT] = 0.0
                curr_storage[KEY_XHJT] += surplus
            excess = curr_storage - self.max_storage
            if (excess > 0).any():
                punish2 += float(excess[excess > 0].sum())
            deficit = self.min_storage - curr_storage
            if (deficit > 0).any():
                punish2 += float(deficit[deficit > 0].sum())
            storage_df[thour] = curr_storage
        return punish2, storage_df

    def calc_punish_3(self, dna: pd.Series) -> float:
        """Penalize large consecutive gene jumps (pump switching)."""
        coef3 = 100.0
        punish3 = 0.0
        for icol, pump in enumerate(self.flowratecand.columns):
            dna_this = dna[icol * self.len_decvar : (icol + 1) * self.len_decvar]
            tem = abs(dna_this.diff())
            thr = 2 if pump == KEY_XHJT else 1
            punish3 += float(tem.loc[tem > thr].sum())
        return punish3 * coef3

    def flow_to_fitness(self, df_flow: pd.DataFrame, coef: float, saveflag: bool = False) -> float:
        punish1, volume_df = self.calc_punish_1(df_flow)
        punish2 = 0.0
        storage_df = None
        for pattern in self.flow_pattern.values():
            qup = pattern.copy()
            for idx in qup.index:
                qup.loc[idx] *= self.flow_quantity
            p2, storage_df = self.calc_punish_2(df_flow, qup)
            punish2 += p2
        if self.flow_pattern:
            punish2 /= len(self.flow_pattern)
        fitness = coef * (punish1 + punish2)
        if saveflag:
            self.volume_record = volume_df
            self.storage_record = storage_df
        return float(fitness)

    def calc_individual_fitness(self, dna: pd.Series, coef: float, saveflag: bool = False) -> float:
        df_flow = self.dna_to_flow(dna)
        return self.flow_to_fitness(df_flow, coef, saveflag) + self.calc_punish_3(dna)

    def calc_fitness(self, popin: pd.DataFrame, coef: float = 10.0) -> pd.Series:
        fit = pd.Series(dtype=float)
        if self.USE_PARALELL:
            data_for_pool = [(dna, coef) for _, dna in popin.iterrows()]
            with Pool(processes=self.cpu_count) as pool:
                results = pool.starmap(self.calc_individual_fitness, data_for_pool)
            for i, ifit in enumerate(results):
                fit.loc[i] = ifit
        else:
            for i, dna in popin.iterrows():
                fit.loc[i] = self.calc_individual_fitness(dna, coef)
        return fit
