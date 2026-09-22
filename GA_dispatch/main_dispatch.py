"""Hourly GA dispatch entry point (offline demo).

Builds synthetic inflow patterns, runs GA + tank-model fitness, and exports
the best 24 h discrete flow plan. No remote I/O.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from ruamel import yaml

from controloptimize import GeneticAlgorithm, PumpControl
from naming_local import KEY_HJT, KEY_TJQ, KEY_WJT, KEY_XHJT, KEY_XL
from naming_local import (
    SEC_HJT_XL,
    SEC_TJQ_WWTP2,
    SEC_WJT_TJQ,
    SEC_XHJT_WWTP2,
    SEC_XL_WWTP1,
)

HERE = Path(__file__).resolve().parent


def demo_patterns(t_total: int) -> dict[str, pd.DataFrame]:
    """Two illustrative diurnal patterns for upstream catchments."""
    hours = np.arange(t_total)
    base = 0.8 + 0.2 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    # Upstream storage is tracked only for catchment pumps (not TJQ/XL).
    pat1 = pd.DataFrame(
        {
            KEY_WJT: base,
            KEY_XHJT: base * 1.1,
            KEY_HJT: base * 0.9,
        }
    )
    pat2 = pat1 * 1.05
    return {"dry": pat1, "wet": pat2}


def demo_uncontrolled(total_steps: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    cols = ["U7", "U8", "U5", "U6", "U9", "U10", "gravity"]
    data = {c: np.clip(200 + rng.normal(0, 20, total_steps), 0, None) for c in cols}
    return pd.DataFrame(data)


def main() -> None:
    with open(HERE / "config.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.RoundTripLoader)

    ctrl = PumpControl(config)
    ctrl.load_flow_candidates({k: list(v) for k, v in config["flow_candidates"].items()})

    q_pump = pd.Series(
        {
            KEY_WJT: 3.0e4,
            KEY_XHJT: 8.0e4,
            KEY_HJT: 2.5e4,
            KEY_TJQ: 4.0e4,
            KEY_XL: 5.0e4,
        }
    )
    q_wwtp = float(config["flow"]["wwtp1"]) + float(config["flow"]["wwtp2_min"])
    ctrl.load_quantity(q_wwtp, q_pump)
    ctrl.load_pattern(demo_patterns(ctrl.t_total))
    ctrl.load_extflow(demo_uncontrolled(ctrl.total_steps))
    ctrl.load_init_state(
        volume=pd.Series(
            {
                SEC_WJT_TJQ: 4000.0,
                SEC_TJQ_WWTP2: 4500.0,
                SEC_XHJT_WWTP2: 9000.0,
                SEC_HJT_XL: 3000.0,
                SEC_XL_WWTP1: 4500.0,
            }
        ),
        storage=pd.Series({KEY_WJT: 2000.0, KEY_XHJT: 10000.0, KEY_HJT: 2000.0}),
    )

    # DNA length = n_pumps * free decision steps
    dna_size = len(ctrl.flowratecand.columns) * ctrl.len_decvar
    value_max = []
    for pump in ctrl.flowratecand.columns:
        n_opt = int(ctrl.flowratecand[pump].dropna().shape[0])
        value_max.extend([n_opt - 1] * ctrl.len_decvar)

    ga = GeneticAlgorithm(
        npop=int(config["GA_setting"]["npop"]),
        dna_size=dna_size,
        value_max=value_max,
        config=config,
    )
    ga.init_pop()
    coef = float(config["GA_setting"]["punish_coef_0"])
    fit = ctrl.calc_fitness(ga.pop.drop(columns=["fit"]), coef=coef)
    ga.add_fitness(fit)

    ngen = int(config["GA_setting"]["ngen"])
    for gen in range(ngen):
        ga.crossover()
        ga.mutate()
        if not ga.popnew.empty:
            genes = ga.popnew.drop(columns=["fit"], errors="ignore")
            fit_new = ctrl.calc_fitness(genes, coef=coef)
            ga.popnew = genes.copy()
            ga.add_fitness_new(fit_new)
            ga.selection()
        best_fit = ga.fitness_list[-1] if ga.fitness_list else float(ga.pop["fit"].iloc[0])
        print(f"gen={gen + 1}/{ngen} best_fit={best_fit:.3f}")

    best = ga.pop.iloc[0].drop(labels=["fit"])
    plan = ctrl.dna_to_flow(best)
    _ = ctrl.calc_individual_fitness(best, coef=coef, saveflag=True)

    plan.columns = [c.upper() for c in plan.columns]
    print(plan.head())
    if config.get("write_output"):
        out = HERE / str(config.get("output_csv") or "dispatch_plan.csv")
        plan.to_csv(out, index_label="step")
        print(f"Wrote {out} at {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
