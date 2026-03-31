# PREPROCESADO + NORMALIZACIÓN Z ROBUSTO (INTRA-SUJETO)
# HRV + EDA (bloques por modo) — versión "todo en uno"
# ============================================


import pandas as pd
import numpy as np
import io, codecs, json
from pathlib import Path


# -----------------------------
# 0) CONFIGURACIÓN
# -----------------------------
DATA_PATH = "/content/metricas_experimentales.csv"   # <-- Cambia a tu ruta
OUT_DIR   = "/content"
OUT_CLEAN = f"{OUT_DIR}/preprocessed_normalized_zrob.csv"
OUT_PARAMS= f"{OUT_DIR}/normalization_params_zrob.csv"
OUT_MANIF = f"{OUT_DIR}/normalization_manifest.json"


# Columnas esperadas
ID_COLS    = ["Subject", "Modo", "Tarea"]
HRV_COLS   = ["SDNN", "RMSSD", "CV", "ShEn"]    # lnRMSSD se añade como derivada
EDA_COLS   = ["SCL_mean", "SCL_slope", "SCR_count/min", "SCR_amp_mean", "SCR_AUC"]
BEHAV_COLS = ["Tiempo(s)", "Errores", "Intentos"]


# Si quieres incluir n_Peaks en el "core", pon True (pasará a 36 columnas core)
INCLUDE_N_PEAKS_IN_CORE = False


# Columnas EDA a transformar con log(1+x) por asimetría
LOG_COLS = ["SCR_amp_mean"]  # añadiremos "SCR_AUC_per_min" tras derivarla


# Umbral de outlier en z_rob (|z| > THRESH se marca como atípico)
ZROB_OUTLIER = 4.0




# -----------------------------
# 1) LOADER ROBUSTO
# -----------------------------
def sniff_delimiter(sample_text, candidates=[',',';','\t','|']):
    lines = [l for l in sample_text.splitlines() if l.strip()][:5] or [sample_text]
    counts = {d: np.mean([l.count(d) for l in lines]) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ','


def normalize_header(cols):
    # Limpia y normaliza nombres de columnas
    norm = []
    for c in cols:
        x = str(c).strip().replace('\ufeff','')  # BOM
        x = x.replace('\xa0',' ')                # espacio duro
        x = ' '.join(x.split())                  # colapsa espacios
        norm.append(x)
    mapping = {
        'Sujeto':'Subject', 'sujeto':'Subject', 'subject':'Subject', 'SUBJECT':'Subject',
        'Modo':'Modo', 'modo':'Modo', 'MODE':'Modo', 'Mode':'Modo',
        'Tarea':'Tarea', 'tarea':'Tarea', 'Trial':'Tarea', 'trial':'Tarea'
    }
    norm = [mapping.get(c, c) for c in norm]
    return norm


def load_table_robusto(path):
    # lee como texto para olfatear separador y quitar BOM
    with open(path, 'rb') as f:
        raw = f.read()
    text = codecs.decode(raw, 'utf-8-sig', errors='ignore')
    delim = sniff_delimiter(text)


    # 1º intento con delimitador olfateado
    df = pd.read_csv(io.StringIO(text), sep=delim, engine='python')
    df.columns = normalize_header(df.columns)


    # si solo hay 1 columna, prueba otros separadores o whitespace
    if df.shape[1] == 1:
        for alt in [';', '\t', '|', ',']:
            if alt == delim:
                continue
            try:
                df_try = pd.read_csv(io.StringIO(text), sep=alt, engine='python')
                df_try.columns = normalize_header(df_try.columns)
                if df_try.shape[1] > 1:
                    df = df_try
                    break
            except Exception:
                pass
        if df.shape[1] == 1:
            df = pd.read_table(io.StringIO(text), delim_whitespace=True, engine='python')
            df.columns = normalize_header(df.columns)


    # Header pegado:
    if df.shape[1] == 1 and isinstance(df.columns[0], str):
        head = df.columns[0]
        for sp in [',',';','\t','|']:
            if sp in head:
                new_cols = [c.strip() for c in head.split(sp)]
                tmp = df.iloc[:,0].str.split(sp, expand=True)
                tmp.columns = normalize_header(new_cols)
                df = tmp
                break


    # normaliza numéricos coma->punto
    num_candidates = [
        "SDNN","RMSSD","CV","ShEn","SCL_mean","SCL_slope","SCR_count/min",
        "SCR_amp_mean","SCR_AUC","n_Peaks","Tiempo(s)","Errores","Intentos"
    ]
    for c in [c for c in num_candidates if c in df.columns]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")


    # tipa enteros si existen
    for c in ["Tarea","Errores","Intentos","n_Peaks"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")


    # orden parcial
    wanted = [c for c in ["Subject","Modo","Tarea"] if c in df.columns]
    others  = [c for c in df.columns if c not in wanted]
    df = df[wanted + others]


    print("Columnas detectadas:", list(df.columns))
    print("Dimensión:", df.shape)
    return df




# -----------------------------
# 2) LIMPIEZA DE VALORES + FORWARD-FILL (clave para evitar “0000”)
# -----------------------------
def canonicalize_str(s: str) -> str:
    s = str(s)
    s = s.replace('\ufeff', '')   # BOM
    s = s.replace('\xa0', ' ')    # espacio duro
    s = ' '.join(s.split())       # colapsa espacios internos
    return s.strip()


df = load_table_robusto(DATA_PATH)


# Si aún no aparece 'Subject', intenta renombrar desde alias
if "Subject" not in df.columns:
    cand = [c for c in df.columns if c.lower() in ("sujeto","subject","id","participant","pp","id_sujeto")]
    if cand:
        df = df.rename(columns={cand[0]: "Subject"})
    else:
        raise ValueError("No se encuentra columna de sujeto (Subject/Sujeto). Revisa encabezados del CSV.")


# 1) Convierte cadenas vacías y literales 'nan'/'NaN'/'None' en NaN reales
for col in ["Subject", "Modo"]:
    if col in df.columns:
        df[col] = df[col].astype(str)
        df[col] = df[col].replace({'': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan})


# 2) Limpia valores no nulos (BOM, espacios duros, etc.)
for col in ["Subject", "Modo"]:
    if col in df.columns:
        df[col] = df[col].map(lambda x: canonicalize_str(x) if pd.notna(x) else x)


# 3) Rellena hacia abajo (archivo debe estar ordenado por bloques)
df[["Subject","Modo"]] = df[["Subject","Modo"]].ffill()


# 4) Normaliza etiquetas de Modo por si hubiera variantes
modo_map = {"RE":"RE","XR":"XR","XS":"XS","R":"RE","X R":"XR","S":"XS"}
df["Modo"] = df["Modo"].str.upper().map(lambda m: modo_map.get(m, m))


# 5) Asegura Tarea numérica
df["Tarea"] = pd.to_numeric(df["Tarea"], errors="coerce").astype("Int64")


# Comprobaciones duras
print("\n>>> Tokens de Subject tras ffill:")
print(df["Subject"].map(repr).drop_duplicates().to_list())


rows_per_subject = df.groupby("Subject").size().sort_values()
print("\nFilas por Subject (esperado 9 si son 3x3):")
print(rows_per_subject)


# Orden consistente y aserción mínima
df = df.sort_values(["Subject","Modo","Tarea"], kind="mergesort").reset_index(drop=True)
assert (rows_per_subject >= 3).all(), "Hay sujetos con <3 filas; revisa que el CSV esté en orden y con bloques completos."




# -----------------------------
# 3) DERIVADOS DE EDA Y TRANSFORMACIONES
# -----------------------------
# Duración en minutos basada en Tiempo(s)
if "Tiempo(s)" in df.columns:
    df["Dur_min"] = df["Tiempo(s)"] / 60.0
else:
    df["Dur_min"] = np.nan


# AUC por minuto
if "SCR_AUC" in df.columns:
    df["SCR_AUC_per_min"] = df["SCR_AUC"] / df["Dur_min"]
else:
    df["SCR_AUC_per_min"] = np.nan


# Transformaciones log1p
def safe_log1p(series):
    x = series.copy()
    x = x.where(x > -1 + 1e-12, -1 + 1e-12)  # asegura x > -1
    return np.log1p(x)


LOG_COLS_DERIVED = LOG_COLS + ["SCR_AUC_per_min"]
for c in LOG_COLS_DERIVED:
    if c in df.columns:
        df[f"{c}_log1p"] = safe_log1p(df[c])


# lnRMSSD
if "RMSSD" in df.columns:
    df["lnRMSSD"] = np.log(np.maximum(df["RMSSD"], 1e-9))




# -----------------------------
# 4) NORMALIZACIÓN INTRA-SUJETO (Z ROBUSTO)
# -----------------------------
def robust_z_by_subject(data, cols, id_col="Subject"):
    """
    zrob = (x - mediana_sujeto) / (1.4826 * MAD_sujeto)
    Si MAD=0, usa desviación estándar poblacional como respaldo.
    """
    data = data.copy()
    params = []
    if id_col not in data.columns:
        raise ValueError(f"No existe columna '{id_col}' en el DataFrame.")
    subjects = data[id_col].astype(str).unique()


    for col in cols:
        if col not in data.columns:
            continue
        zcol = f"{col}_zrob"
        data[zcol] = np.nan
        for s in subjects:
            mask = data[id_col].astype(str) == str(s)
            x = data.loc[mask, col].astype(float).to_numpy()
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            scale = 1.4826 * mad
            if not np.isfinite(scale) or scale <= 0:
                std = np.nanstd(x)
                scale = std if (np.isfinite(std) and std > 0) else 1.0
            z = (x - med) / scale
            data.loc[mask, zcol] = z
            params.append({
                "Subject": s, "variable": col,
                "median": med, "MAD": mad, "scale_used": scale, "n": int(np.isfinite(x).sum())
            })
    return data, pd.DataFrame(params)


# Qué columnas normalizar
norm_cols = []
norm_cols += [c for c in HRV_COLS if c in df.columns]
if "lnRMSSD" in df.columns: norm_cols.append("lnRMSSD")
norm_cols += [c for c in (EDA_COLS + ["SCR_AUC_per_min"] + [f"{c}_log1p" for c in LOG_COLS_DERIVED]) if c in df.columns]
norm_cols = sorted(set(norm_cols))


df_norm, params = robust_z_by_subject(df, cols=norm_cols, id_col="Subject")


# -----------------------------
# 5) BANDERA DE OUTLIERS + DIAGNÓSTICO
# -----------------------------
z_cols = [c for c in df_norm.columns if c.endswith("_zrob")]
df_norm["outlier_any"] = df_norm[z_cols].abs().gt(ZROB_OUTLIER).any(axis=1)


# Diagnóstico de la primera fila por sujeto (deberían desaparecer "0000" sistemáticos)
cols_check = [c for c in ["SCL_mean_zrob","SCL_slope_zrob","SCR_count/min_zrob","SCR_amp_mean_zrob"] if c in df_norm.columns]
first_rows = (df_norm.sort_values(["Subject","Modo","Tarea"])
                      .groupby("Subject", sort=False)
                      .head(1))[["Subject","Modo","Tarea"] + cols_check]
print("\nPrimer bloque por Subject (comprobación z_rob):")
try:
    from IPython.display import display
    display(first_rows)
except Exception:
    print(first_rows.to_string(index=False))




# -----------------------------
# 6) ORDEN DE COLUMNAS (35 core; opcional incluir n_Peaks)
# -----------------------------
# Añade n_Peaks a EDA_COLS si quieres mantenerla junto a las originales
if "n_Peaks" in df_norm.columns and "n_Peaks" not in EDA_COLS:
    EDA_COLS.append("n_Peaks")


# BLOQUES
id_cols_present = [c for c in ID_COLS if c in df_norm.columns]
orig_cols = [c for c in (HRV_COLS + EDA_COLS + BEHAV_COLS) if c in df_norm.columns]
derived_cols = [c for c in ["Dur_min", "SCR_AUC_per_min", "lnRMSSD"] + [f"{c}_log1p" for c in LOG_COLS_DERIVED] if c in df_norm.columns]


# ZROB CORE (sin SCR_AUC_zrob, preferimos AUC/min)
z_core = [
    "SDNN_zrob","RMSSD_zrob","lnRMSSD_zrob","CV_zrob","ShEn_zrob",
    "SCL_mean_zrob","SCL_slope_zrob","SCR_count/min_zrob","SCR_amp_mean_zrob",
    # "SCR_AUC_zrob",  # excluida
    "Dur_min_zrob","SCR_AUC_per_min_zrob","SCR_amp_mean_log1p_zrob","SCR_AUC_per_min_log1p_zrob"
]
z_core = [c for c in z_core if c in df_norm.columns]


flag_cols = ["outlier_any"]


# Construye orden "core"
ordered = id_cols_present + orig_cols + derived_cols + z_core + flag_cols


# Si NO quieres n_Peaks en el core y estaba en orig_cols, sácalo a la cola
if not INCLUDE_N_PEAKS_IN_CORE and "n_Peaks" in ordered:
    ordered.remove("n_Peaks")  # quedará como leftover al final


# Añade el resto de columnas que no estén (incluida n_Peaks si no core)
leftovers = [c for c in df_norm.columns if c not in ordered]
ordered += leftovers


df_norm = df_norm[ordered]


# Si quieres exactamente 35 core, sin n_Peaks:
if not INCLUDE_N_PEAKS_IN_CORE:
    core_35 = [
        "Subject","Modo","Tarea",
        "SDNN","RMSSD","CV","ShEn","SCL_mean","SCL_slope","SCR_count/min","SCR_amp_mean","SCR_AUC",
        "Tiempo(s)","Errores","Intentos",
        "Dur_min","SCR_AUC_per_min","lnRMSSD","SCR_amp_mean_log1p","SCR_AUC_per_min_log1p",
        "SDNN_zrob","RMSSD_zrob","lnRMSSD_zrob","CV_zrob","ShEn_zrob","SCL_mean_zrob","SCL_slope_zrob","SCR_count/min_zrob","SCR_amp_mean_zrob",
        "Dur_min_zrob","SCR_AUC_per_min_zrob","SCR_amp_mean_log1p_zrob","SCR_AUC_per_min_log1p_zrob",
        "outlier_any"
    ]
    core_35 = [c for c in core_35 if c in df_norm.columns]
    df_norm = df_norm[core_35 + [c for c in df_norm.columns if c not in core_35]]


# -----------------------------
# 7) GUARDADO + MANIFEST (replicabilidad)
# -----------------------------
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
df_norm.to_csv(OUT_CLEAN, index=False)
params.to_csv(OUT_PARAMS, index=False)


manifest = {
    "normalization_method": "robust",  # mediana & 1.4826*MAD
    "group_col": "Subject",
    "scale_cols": norm_cols,
    "z_outlier_threshold": ZROB_OUTLIER,
    "include_n_peaks_in_core": INCLUDE_N_PEAKS_IN_CORE,
    "paths": {"data": DATA_PATH, "out_clean": OUT_CLEAN, "out_params": OUT_PARAMS},
    "timestamp_utc": pd.Timestamp.utcnow().isoformat()
}
with open(OUT_MANIF, "w") as f:
    json.dump(manifest, f, indent=2)


print(f"\nGuardado:\n- {OUT_CLEAN}\n- {OUT_PARAMS}\n- {OUT_MANIF}")


# Vista rápida por modo (no inferencial, solo sanity check)
if "Modo" in df_norm.columns:
    cols_for_summary = [c for c in df_norm.columns if c.endswith("_zrob") and any(k in c for k in ["RMSSD","SDNN","SCL_mean","SCR_count","SCR_amp_mean","SCR_AUC_per_min"])]
    summary = df_norm.groupby("Modo")[cols_for_summary].mean().round(3)
    print("\nMedia de z_rob por Modo (vista rápida, no inferencial):")
    try:
        from IPython.display import display
        display(summary)
    except Exception:
        print(summary.to_string())