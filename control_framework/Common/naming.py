"""Canonical anonymized asset IDs used across the control framework.

Controlled pumps use the Pump-<INITIALS> pattern.
Wastewater treatment plants use WWTP-1 / WWTP-2.
Uncontrolled / upstream feeders use Pump-U#.
Pipe sections use <FROM>-<TO> tokens built from the same initials.
"""

# Controlled pump stations
PUMP_XL = "Pump-XL"
PUMP_HJT = "Pump-HJT"
PUMP_XHJT = "Pump-XHJT"
PUMP_TJQ = "Pump-TJQ"
PUMP_WJT = "Pump-WJT"

CONTROLLED_PUMPS = [PUMP_XHJT, PUMP_XL, PUMP_HJT, PUMP_TJQ, PUMP_WJT]

# Wastewater treatment plants
WWTP_1 = "WWTP-1"
WWTP_2 = "WWTP-2"
WWTPS = [WWTP_1, WWTP_2]

# Upstream / uncontrolled feeders (topology inputs only)
PUMP_U1 = "Pump-U1"  # feeder to Pump-HJT
PUMP_U2 = "Pump-U2"  # feeder to Pump-XHJT
PUMP_U3 = "Pump-U3"  # feeder to Pump-WJT
PUMP_U4 = "Pump-U4"  # feeder to Pump-WJT
PUMP_U5 = "Pump-U5"  # feeder toward Pump-XL corridor
PUMP_U6 = "Pump-U6"  # feeder toward Pump-XL corridor
PUMP_U7 = "Pump-U7"  # feeder toward Pump-TJQ corridor
PUMP_U8 = "Pump-U8"  # feeder toward WWTP-2 corridor
PUMP_U9 = "Pump-U9"  # feeder toward WWTP-1 corridor
PUMP_U10 = "Pump-U10"  # feeder toward WWTP-1 corridor

# Short keys used inside the tank / GA model (must stay ASCII-safe)
KEY_XL = "xl"
KEY_HJT = "hjt"
KEY_XHJT = "xhjt"
KEY_TJQ = "tjq"
KEY_WJT = "wjt"

PUMP_KEY_TO_ID = {
    KEY_XHJT: PUMP_XHJT,
    KEY_XL: PUMP_XL,
    KEY_HJT: PUMP_HJT,
    KEY_TJQ: PUMP_TJQ,
    KEY_WJT: PUMP_WJT,
}
PUMP_ID_TO_KEY = {v: k for k, v in PUMP_KEY_TO_ID.items()}

# Pipe / tank sections
SEC_WJT_TJQ = "WJT-TJQ"
SEC_TJQ_WWTP2 = "TJQ-WWTP2"
SEC_XHJT_WWTP2 = "XHJT-WWTP2"
SEC_HJT_XL = "HJT-XL"
SEC_XL_WWTP1 = "XL-WWTP1"
SEC_WJT_XHJT = "WJT-XHJT"
SEC_A = "SEC-A"
SEC_B = "SEC-B"
SEC_C = "SEC-C"
SEC_D = "SEC-D"
SEC_E = "SEC-E"

TANK_SECTIONS = [
    SEC_WJT_TJQ,
    SEC_TJQ_WWTP2,
    SEC_XHJT_WWTP2,
    SEC_HJT_XL,
    SEC_XL_WWTP1,
]

SCAN_SECTIONS = [
    SEC_A,
    SEC_B,
    SEC_HJT_XL,
    SEC_C,
    SEC_D,
    SEC_TJQ_WWTP2,
    SEC_WJT_TJQ,
    SEC_WJT_XHJT,
    SEC_XL_WWTP1,
    SEC_E,
]
