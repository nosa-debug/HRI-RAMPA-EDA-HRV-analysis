# @title 🧠 Funciones EDA: filtrado, separación SCL/SCR y métricas
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional


# ===== Parámetros por defecto =====
FS_DEFAULT = 256.0 # Frecuencia de muestreo (Hz)
EDA_COLNAME = "Analog AUX [1]-ch1" # Nombre esperado de columna EDA
LOWPASS_CUTOFF = 2.0 # Hz
ROLL_WIN_SEC = 5.0 # ventana (s) para SCL simple
PEAK_THR_SD = 0.05 # umbral prominencia = 0.05 * SD(phasic)


def butter_lowpass_filt(x: np.ndarray, cutoff=LOWPASS_CUTOFF, fs=FS_DEFAULT, order=2):
b, a = butter(order, cutoff/(fs/2.0), btype='low')
return filtfilt(b, a, x)


def split_tonic_phasic(eda_f: np.ndarray, fs=FS_DEFAULT, roll_win_sec=ROLL_WIN_SEC):
win = int(max(1, roll_win_sec * fs))
tonic = pd.Series(eda_f).rolling(win, min_periods=1, center=True).mean().values
phasic = eda_f - tonic
return tonic, phasic


def compute_metrics(t: np.ndarray, tonic: np.ndarray, phasic: np.ndarray, peak_thr_sd=PEAK_THR_SD) -> Dict[str, float]:
# picos SCR
thr = peak_thr_sd * float(np.nanstd(phasic))
peaks, props = find_peaks(phasic, prominence=thr if thr>0 else None)
duration_min = t[-1] / 60.0 if len(t) > 1 else np.nan


SCL_mean = float(np.nanmean(tonic))
SCL_slope = float(np.polyfit(t, tonic, 1)[0]) if len(t) > 10 else np.nan
SCR_count_per_min = float(len(peaks) / duration_min) if duration_min and duration_min>0 else np.nan
SCR_amp_mean = float(np.nanmean(phasic[peaks])) if len(peaks) > 0 else np.nan
SCR_AUC = float(np.trapz(np.abs(phasic), t) / duration_min) if duration_min and duration_min>0 else np.nan


return {
"SCL_mean": SCL_mean,
"SCL_slope": SCL_slope,
"SCR_count_per_min": SCR_count_per_min,
"SCR_amp_mean": SCR_amp_mean,
"SCR_AUC": SCR_AUC,
"n_peaks": int(len(peaks)),
"duration_min": duration_min
}, peaks


def plot_eda(t: np.ndarray, eda_f: np.ndarray, tonic: np.ndarray, phasic: np.ndarray, peaks: np.ndarray,
title: str, out_png: Optional[str]=None):
plt.figure(figsize=(13,5))
plt.plot(t, eda_f, label='EDA filtrada', alpha=0.45)
plt.plot(t, tonic, label='SCL (tónica)', linewidth=2)
if len(peaks) > 0:
plt.scatter(t[peaks], phasic[peaks] + tonic[peaks], s=12, color='red', label='Picos SCR')
plt.xlabel("Tiempo (s)"); plt.ylabel("Amplitud (unid. relativas)")
plt.title(title); plt.legend(); plt.tight_layout()
if out_png:
plt.savefig(out_png, dpi=160)
plt.show()


def process_one_series(sig: np.ndarray, fs=FS_DEFAULT,
lowpass_cutoff=LOWPASS_CUTOFF,
roll_win_sec=ROLL_WIN_SEC,
peak_thr_sd=PEAK_THR_SD) -> Tuple[Dict[str,float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
# tiempo
t = np.arange(len(sig)) / fs
# filtrado
eda_f = butter_lowpass_filt(sig, cutoff=lowpass_cutoff, fs=fs, order=2)
# separación
tonic, phasic = split_tonic_phasic(eda_f, fs=fs, roll_win_sec=roll_win_sec)
# métricas
metrics, peaks = compute_metrics(t, tonic, phasic, peak_thr_sd=peak_thr_sd)
return metrics, t, eda_f, tonic, phasic, peaks


# @title ⬆️ Subir archivos y procesar en lote
from google.colab import files


# --- Subir archivos CSV/XLSX ---
uploaded = files.upload() # selecciona todos los archivos del participante / lote


# --- Parámetros (ajústalos si hace falta) ---
fs = 256.0 # frecuencia de muestreo
eda_col_candidates = [EDA_COLNAME, "EDA", "GSR"] # alias posibles por si cambia el nombre
save_plots = True # guardar PNGs de cada archivo
plots_dir = "eda_plots"
os.makedirs(plots_dir, exist_ok=True)


rows = []
for fname in uploaded.keys():
try:
# Leer CSV o XLSX
if fname.lower().endswith((".xlsx", ".xls")):
df = pd.read_excel(fname)
else:
df = pd.read_csv(fname)


# Buscar columna EDA
eda_col = None
for c in eda_col_candidates:
if c in df.columns:
eda_col = c
break
if eda_col is None:
# último intento: tomar la única columna numérica si el archivo sólo tiene una
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(numeric_cols) == 1:
eda_col = numeric_cols[0]
else:
raise ValueError(f"No se encontró columna EDA en {fname}. Columnas: {list(df.columns)[:8]} ...")


sig = df[eda_col].astype(float).values


# Procesar
metrics, t, eda_f, tonic, phasic, peaks = process_one_series(
sig, fs=fs, lowpass_cutoff=LOWPASS_CUTOFF, roll_win_sec=ROLL_WIN_SEC, peak_thr_sd=PEAK_THR_SD
)


# Guardar métricas
row = {"file": fname, "eda_col": eda_col}
row.update(metrics)
rows.append(row)


# Plot
if save_plots:
out_png = os.path.join(plots_dir, os.path.splitext(os.path.basename(fname))[0] + "_eda.png")
plot_eda(t, eda_f, tonic, phasic, peaks, title=f"EDA: {fname}", out_png=out_png)
else:
plot_eda(t, eda_f, tonic, phasic, peaks, title=f"EDA: {fname}", out_png=None)


except Exception as e:
print(f"[ERROR] {fname}: {e}")


# Exportar CSV resumen
summary = pd.DataFrame(rows)
summary_path = "eda_metrics_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"\n✅ Guardado resumen: {summary_path} ({len(summary)} archivos)")


try:
files.download(summary_path)
except:
pass


print("\nPrimeras filas:")
summary.head()