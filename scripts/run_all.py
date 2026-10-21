"""Valve and damper faults in a single-duct air-handling unit (LBNL FDD Data Sets, SDAHU: one simulated year at 1-minute
resolution per case, Chicago TMY weather; Granderson et al. 2022, doi 10.25984/1881324).

1. Data audit: duplicate files, constant or swapped columns, and whether each fault file shows its fault.
2. Detection with the points a building automation system logs ("basic" points in the data documentation):
   physics residuals of the mixing box and the cooling coil, learned on the fault-free year.
3. Detection when the valve also reports its position (as smart valves do): command-position residual.
4. Energy impact of each fault: fan energy and the cooling coil load (airflow x enthalpy drop, sensible part).
Writes results/summary.json and figures/.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "LBNL_FDD_Dataset_SDAHU"
FIG, RES = ROOT / "figures", ROOT / "results"
VALID = ["coi_leakage_010", "coi_stuck_010", "coi_stuck_025", "coi_stuck_050", "coi_stuck_075",
         "damper_stuck_010", "damper_stuck_025", "damper_stuck_075", "damper_stuck_100"]
TRAIN_END = "2018-07-01"           # fault-free model fitted on Jan-Jun, false alarms counted on Jul-Dec


def load(name):
    f = DATA / (name + "_annual.csv") if name != "damper_stuck_100" else DATA / "damper_stuck_100_annual_short.csv"
    if name == "AHU":
        f = DATA / "AHU_annual.csv"
    d = pd.read_csv(f, parse_dates=["Datetime"]).set_index("Datetime")
    return d


def audit():
    rows = []
    hashes = {}
    for f in sorted(DATA.glob("*.csv")):
        h = hashlib.md5(f.read_bytes()).hexdigest()
        hashes.setdefault(h, []).append(f.stem)
    dup = [v for v in hashes.values() if len(v) > 1]
    ff = load("AHU")
    for col in ff.columns:
        rows.append(dict(column=col, fault_free_std=float(ff[col].std()), fault_free_median=float(ff[col].median())))
    const = [r["column"] for r in rows if r["fault_free_std"] < 1e-6]
    # does the outdoor-air sensor bias appear in the oa_bias files?
    ob = load("oa_bias_2")
    st = load("coi_leakage_010")
    return dict(duplicate_groups=dup, constant_columns=const,
                oa_bias_2_minus_other_fault_file_OA_TEMP_degF=float((ob.OA_TEMP - st.OA_TEMP).abs().max()),
                sa_spspt_fault_free_median=float(ff.SA_SPSPT.median()), sa_spspt_fault_files_median=float(st.SA_SPSPT.median()),
                sa_sp_fault_free_median=float(ff.SA_SP.median()))


def features(d):
    occ = (d.SYS_CTL > 0.5) & (d.SF_SPD_DM > 0.5)
    x = pd.DataFrame(index=d.index)
    dT = d.OA_TEMP - d.RA_TEMP
    x["oa_frac"] = np.where(dT.abs() > 5, (d.MA_TEMP - d.RA_TEMP) / dT.replace(0, np.nan), np.nan)   # mixing balance
    x["oa_cmd"] = d.OA_DMPR_DM
    x["coil_dT"] = d.MA_TEMP - d.SA_TEMP                    # air-side temperature drop over the coil (+ fan heat)
    x["vlv_cmd"] = d.CHWC_VLV_DM
    x["sat_err"] = d.SA_TEMP - d.SA_TEMPSPT
    x["vlv_pos_err"] = d.CHWC_VLV - d.CHWC_VLV_DM           # needs position feedback
    x["occ"] = occ
    return x


def residual_model(ff):
    """Fault-free expectations learned on Jan-Jun: OA fraction vs damper command (binned), coil temperature drop when the
    valve is closed, and supply-temperature error when the valve is saturated open."""
    x = features(ff)
    tr = x[(x.index < TRAIN_END) & x.occ]
    bins = np.linspace(0, 1, 21)
    cmd_bin = np.digitize(tr.oa_cmd, bins)
    frac_med = tr.groupby(cmd_bin).oa_frac.median()
    closed = tr[tr.vlv_cmd < 0.01].coil_dT
    return dict(bins=bins, frac_med=frac_med, dT_closed_q99=float(closed.quantile(0.99)),
                dT_closed_median=float(closed.median()))


def daily_flags(d, m):
    """Daily flags from hourly-averaged residuals (occupied hours only)."""
    x = features(d)
    x = x[x.occ]
    h = x.resample("1h").mean().dropna(subset=["oa_cmd"])
    exp = m["frac_med"].reindex(np.digitize(h.oa_cmd, m["bins"])).values
    damper = (h.oa_frac - exp).abs() > 0.15                                  # mixing inconsistent with the command
    leak = (h.vlv_cmd < 0.01) & (h.coil_dT > m["dT_closed_q99"] + 1.0)        # cooling while the valve is told to close
    starve = (h.vlv_cmd > 0.98) & (h.sat_err > 2.0)                            # valve fully open, supply still too warm
    pos = h.vlv_pos_err.abs() > 0.05                                           # smart valve: position != command
    flags = pd.DataFrame(dict(damper=damper, leak=leak, starve=starve, position=pos), index=h.index)
    daily = flags.resample("1D").sum()
    hours = h.resample("1D").size()
    return daily[(hours > 4)] >= 2                                             # at least 2 flagged hours in a day


def energy(d):
    occ = (d.SYS_CTL > 0.5)
    fan_kwh = float((d.SF_WAT + d.RF_WAT).clip(lower=0).sum() / 60 / 1000)
    load = (d.SA_CFM.clip(lower=0) * (d.MA_TEMP - d.SA_TEMP).clip(lower=0))[occ].sum()   # relative units
    return fan_kwh, float(load)


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    out = dict(audit=audit())
    ff = load("AHU")
    m = residual_model(ff)
    out["model"] = dict(dT_closed_q99_F=m["dT_closed_q99"], dT_closed_median_F=m["dT_closed_median"])
    f0 = daily_flags(ff, m)
    f0 = f0[f0.index >= TRAIN_END]
    out["false_alarm_days_fault_free_JulDec"] = {c: int(f0[c].sum()) for c in f0.columns}
    out["fault_free_days_JulDec"] = int(len(f0))
    fan0, coil0 = energy(ff)
    out["fault_free_energy"] = dict(fan_kwh=fan0)
    res = {}
    for name in VALID:
        d = load(name)
        fl = daily_flags(d, m)
        fl = fl[fl.index >= TRAIN_END] if name != "damper_stuck_100" else fl
        fan, coil = energy(d)
        basic = fl[["damper", "leak", "starve"]].any(axis=1)
        res[name] = dict(days=int(len(fl)), basic_any=float(basic.mean()),
                         by_rule={c: float(fl[c].mean()) for c in fl.columns},
                         smart_valve=float((basic | fl["position"]).mean()),
                         fan_energy_change_pct=100 * (fan / fan0 - 1) if name != "damper_stuck_100" else None,
                         coil_load_change_pct=100 * (coil / coil0 - 1) if name != "damper_stuck_100" else None)
        print(name, json.dumps(res[name]), flush=True)
    out["faults"] = res
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    names = [n for n in VALID if n != "damper_stuck_100"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(range(len(names)), [res[n]["basic_any"] for n in names], color="0.6", label="building-system points")
    ax[0].bar(range(len(names)), [res[n]["smart_valve"] - res[n]["basic_any"] for n in names],
              bottom=[res[n]["basic_any"] for n in names], color="tab:red", label="+ valve position feedback")
    ax[0].set_xticks(range(len(names)), [n.replace("_", " ") for n in names], rotation=60, ha="right", fontsize=8)
    ax[0].set_ylabel("share of days flagged (Jul-Dec)"); ax[0].legend(fontsize=8, frameon=True)
    ax[1].bar(range(len(names)), [res[n]["fan_energy_change_pct"] for n in names], color="0.4", label="fan energy")
    ax[1].bar(np.arange(len(names)) + 0.4, [res[n]["coil_load_change_pct"] for n in names], width=0.4, color="tab:blue",
              label="cooling coil load")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(names)), [n.replace("_", " ") for n in names], rotation=60, ha="right", fontsize=8)
    ax[1].set_ylabel("change vs fault-free year (%)"); ax[1].legend(fontsize=8, frameon=True)
    fig.tight_layout(); fig.savefig(FIG / "detection_energy.png", dpi=150); plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "faults"}, indent=1, default=float))


if __name__ == "__main__":
    main()
