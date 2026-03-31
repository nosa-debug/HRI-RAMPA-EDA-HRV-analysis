# ============================================
# HRV ↔ EDA correlations from z-robust metrics
# - Per-mode Spearman (cross-subject) with Holm correction
# - Repeated-measures correlation (rmcorr) across modes
# ============================================


# If running for the first time in this runtime, install pingouin:
!pip -q install pingouin


import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import pingouin as pg


# -----------------------------
# 1) Load data
# -----------------------------
PATH = "/content/preprocessed_normalized_zrob.csv"   # <-- change if needed
df = pd.read_csv(PATH)


# Basic checks
required = {"Subject", "Modo"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")


# Keep only Subject/Modo and z-rob metrics
zcols = [c for c in df.columns if c.endswith("_zrob")]
if not zcols:
    raise ValueError("No *_zrob columns found.")


keep = ["Subject","Modo"] + zcols
df = df[keep].copy()


# Normalize Modo ordering if desired
order = [m for m in ["RE","XR","XS"] if m in df["Modo"].unique()]
if order:
    df["Modo"] = pd.Categorical(df["Modo"], categories=order, ordered=True)


# Average to one row per Subject×Modo (if multiple blocks per mode)
by_sm = df.groupby(["Subject","Modo"], as_index=False)[zcols].mean()


print("Rows:", len(by_sm), "Subjects:", by_sm["Subject"].nunique(), "Modes:", by_sm["Modo"].nunique())
display(by_sm.head())


# -----------------------------
# 2) Define metric sets
# -----------------------------
HRV = ["SDNN_zrob","RMSSD_zrob","lnRMSSD_zrob","CV_zrob","ShEn_zrob"]
EDA = ["SCL_mean_zrob","SCL_slope_zrob","SCR_count/min_zrob","SCR_amp_mean_zrob",
       "SCR_AUC_per_min_zrob","SCR_amp_mean_log1p_zrob","SCR_AUC_per_min_log1p_zrob"]


HRV = [m for m in HRV if m in by_sm.columns]
EDA = [m for m in EDA if m in by_sm.columns]


print("\nHRV metrics:", HRV)
print("EDA metrics:", EDA)


# -----------------------------
# 3) Per-mode Spearman (cross-subject) + Holm
# -----------------------------
def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    idx = np.argsort(pvals)
    out = np.empty_like(pvals, dtype=float)
    m = len(pvals)
    running = 0.0
    for k, i in enumerate(idx):
        adj = pvals[i] * (m - k)
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out


rows = []
modes = list(by_sm["Modo"].cat.categories) if hasattr(by_sm["Modo"], "cat") else sorted(by_sm["Modo"].unique())
for mode in modes:
    sub = by_sm[by_sm["Modo"] == mode]
    for h in HRV:
        for e in EDA:
            x = sub[h]
            y = sub[e]
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            if n >= 5:
                rho, p = spearmanr(x[mask], y[mask])
            else:
                rho, p = np.nan, np.nan
            rows.append({"scope":"per_mode", "mode":mode, "x":h, "y":e, "n":n, "rho":rho, "p_raw":p})


per_mode = pd.DataFrame(rows)


# Holm per mode
per_mode["p_holm"] = np.nan
for mode in modes:
    mask = per_mode["mode"] == mode
    ps = per_mode.loc[mask, "p_raw"].to_numpy()
    ok = np.isfinite(ps)
    if ok.sum() > 0:
        per_mode.loc[mask, "p_holm"] = np.where(ok, holm_adjust(ps[ok]), np.nan)


print("\n=== Spearman (per mode) with Holm adjustment ===")
display(per_mode.sort_values(["mode","p_holm","x","y"]).reset_index(drop=True))


# -----------------------------
# 4) Repeated-measures correlation across modes (rmcorr)
# -----------------------------
rm_rows = []
for h in HRV:
    for e in EDA:
        sub = by_sm[["Subject","Modo",h,e]].dropna()
        # Need >= 2 observations per subject to compute rmcorr
        counts = sub.groupby("Subject").size()
        valid_subjects = counts[counts >= 2].index
        sub = sub[sub["Subject"].isin(valid_subjects)]
        if sub["Subject"].nunique() >= 5:  # avoid unstable fits with too few subjects
            try:
                out = pg.rm_corr(data=sub, x=h, y=e, subject="Subject")
                rm_rows.append({
                    "x": h, "y": e,
                    "r": out.loc[0, "r"],
                    "CI95_low": out.loc[0, "CI95%"][0],
                    "CI95_high": out.loc[0, "CI95%"][1],
                    "p": out.loc[0, "pval"],
                    "df": out.loc[0, "dof"],
                    "n_subjects": sub["Subject"].nunique()
                })
            except Exception as err:
                rm_rows.append({"x":h,"y":e,"r":np.nan,"CI95_low":np.nan,"CI95_high":np.nan,"p":np.nan,"df":np.nan,"n_subjects":sub['Subject'].nunique(),"error":str(err)})
        else:
            rm_rows.append({"x":h,"y":e,"r":np.nan,"CI95_low":np.nan,"CI95_high":np.nan,"p":np.nan,"df":np.nan,"n_subjects":sub['Subject'].nunique(),"error":"<5 subjects with >=2 observations>"})


rmcorr = pd.DataFrame(rm_rows)


print("\n=== Repeated-measures correlation across modes (pooled) ===")
display(rmcorr.sort_values(["p","x","y"]).reset_index(drop=True))


# -----------------------------
# 5) Save outputs
# -----------------------------
per_mode.to_csv("/content/corr_per_mode_spearman.csv", index=False)
rmcorr.to_csv("/content/corr_rmcorr_across_modes.csv", index=False)
print("\nSaved files:")
print("- /content/corr_per_mode_spearman.csv")
print("- /content/corr_rmcorr_across_modes.csv")