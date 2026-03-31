# ============================================
# Cargar CSV "raro" (con líneas extra) + ANOVA RM o aviso si no hay Subject
# ============================================
import pandas as pd
import numpy as np
import io, re
from pathlib import Path


# ⬇️ CAMBIA AQUÍ
DATA = "/content/means_by_mode_zrob.csv"   # pon aquí el CSV que acabas de subir


def _normalize_header(cols):
    out = []
    for c in cols:
        x = str(c).replace("\ufeff","").replace("\xa0"," ")
        x = " ".join(x.split()).strip()
        out.append(x)
    mapping = {"Sujeto":"Subject","subject":"Subject","SUBJECT":"Subject",
               "Mode":"Modo","mode":"Modo","modo":"Modo","MODE":"Modo",
               "Tarea":"Tarea","trial":"Tarea","Trial":"Tarea"}
    return [mapping.get(c, c) for c in out]


def load_with_header_seek(path):
    """
    Lee un CSV que puede tener filas basura antes del header.
    - Busca la primera línea que contenga 'Subject' y 'Modo' (cualquier case, con o sin separadores).
    - Autodetecta el separador con engine='python'.
    - Forward-fill de Subject si hay celdas vacías en filas siguientes.
    """
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8-sig", errors="ignore")


    # 1) localizar la línea del header
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines[:200]):   # miramos primeras 200 líneas por seguridad
        line_clean = line.strip()
        # si la línea contiene ambos tokens (ignorando espacios/separadores)
        if re.search(r"\bsubject\b", line_clean, flags=re.I) and re.search(r"\bmodo\b", line_clean, flags=re.I):
            header_idx = i
            break


    if header_idx is None:
        # si no encontramos header, intentamos directamente inferir
        candidate = text
    else:
        candidate = "\n".join(lines[header_idx:])


    # 2) quitar líneas obvias basura como "Z values"
    cleaned_lines = []
    for line in candidate.splitlines():
        if line.strip().lower() == "z values":
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)


    # 3) intentar leer con autodetección de separador
    try:
        df = pd.read_csv(io.StringIO(cleaned), sep=None, engine="python")
    except Exception:
        # fallback: tab, luego coma, luego punto y coma
        for sep in ["\t", ",", ";", "|"]:
            try:
                df = pd.read_csv(io.StringIO(cleaned), sep=sep, engine="python")
                break
            except Exception:
                df = None
        if df is None:
            raise


    df.columns = _normalize_header(df.columns)


    # 4) si existe una primera columna vacía (a veces ocurre), la eliminamos
    bad_first = df.columns[0]
    if bad_first == "" or bad_first.lower().startswith("unnamed"):
        df = df.drop(columns=[bad_first])


    # 5) forward-fill de Subject si aparece y trae huecos
    if "Subject" in df.columns:
        df["Subject"] = df["Subject"].replace("", np.nan).ffill()


    # 6) recortar espacios en Modo
    if "Modo" in df.columns:
        df["Modo"] = df["Modo"].astype(str).str.strip()


    return df


d 

f = load_with_header_seek(DATA)
print("Columnas detectadas:", list(df.columns))
print("Primeras filas:")
display(df.head(10))


# -------------------------------
# ¿Trae columnas por sujeto (Subject presente)?
# -------------------------------
if "Subject" not in df.columns:
    print("\n⚠️ Este archivo NO contiene columna 'Subject'.")
    print("   Parece ser un archivo de medias por modo (p.ej., 'means_by_mode_zrob.csv').")
    print("   Para ANOVA de medidas repetidas necesitas el archivo con filas por sujeto y modo")
    print("   (las columnas terminadas en _zrob por fila), p.ej. 'preprocessed_normalized_zrob.csv'")
else:
    # Continuamos con el ANOVA RM (una media por Subject×Modo)
    z_cols = [c for c in df.columns if c.endswith("_zrob")]
    if not z_cols:
        raise ValueError("No se han encontrado columnas *_zrob en el archivo.")


    # Ordenar Modo si están los 3 niveles
    order = [m for m in ["RE","XR","XS"] if m in df["Modo"].unique()]
    if order:
        df["Modo"] = pd.Categorical(df["Modo"], categories=order, ordered=True)


    # Media por Subject×Modo (una observación por celda)
    by_sm = df.groupby(["Subject","Modo"], as_index=False)[z_cols].mean()


    # Mantener sujetos con todos los niveles de Modo
    k_levels = len(order) if order else by_sm["Modo"].nunique()
    counts = by_sm.groupby("Subject")["Modo"].nunique()
    keep_subjects = counts[counts==k_levels].index
    by_sm_bal = by_sm[by_sm["Subject"].isin(keep_subjects)].copy()
    print(f"\nSujetos con {k_levels} modos completos: {by_sm_bal['Subject'].nunique()} (de {by_sm['Subject'].nunique()})")


    # ANOVA RM
    from statsmodels.stats.anova import AnovaRM
    anova_rows = []
    for m in z_cols:
        try:
            aov = AnovaRM(data=by_sm_bal, depvar=m, subject="Subject", within=["Modo"]).fit()
            row = aov.anova_table.loc["Modo"]
            anova_rows.append({
                "metric": m,
                "subjects_N": by_sm_bal["Subject"].nunique(),
                "F": row.get("F Value", np.nan),
                "df_num": row.get("Num DF", np.nan),
                "df_den": row.get("Den DF", np.nan),
                "p": row.get("Pr > F", np.nan),
            })
        except Exception as e:
            anova_rows.append({"metric": m, "subjects_N": by_sm_bal["Subject"].nunique(),
                               "F": np.nan, "df_num": np.nan, "df_den": np.nan, "p": np.nan,
                               "error": str(e)})
    anova_df = pd.DataFrame(anova_rows).sort_values("metric")
    print("\n=== ANOVA RM (Modo como factor intra-sujeto) ===")
    display(anova_df)


    # Post-hoc pareados + Holm (XR-RE, XS-RE, XR-XS)
    from scipy import stats
    pairs = [("RE","XR"), ("RE","XS"), ("XR","XS")]
    posthoc_rows = []


    for m in z_cols:
        wide = by_sm_bal.pivot(index="Subject", columns="Modo", values=m)
        have = list(wide.columns)
        usable = [(a,b) for (a,b) in pairs if a in have and b in have]
        res = []
        for a,b in usable:
            sub = wide[[a,b]].dropna()
            if len(sub) < 2:
                res.append((a,b,np.nan,np.nan,len(sub)))
            else:
                t, p = stats.ttest_rel(sub[b], sub[a])  # b - a
                res.append((a,b,t,p,len(sub)))
        # Holm
        ps = [r[3] for r in res if not np.isnan(r[3])]
        order_idx = np.argsort(ps)
        mtests = len(ps)
        p_holm = [np.nan]*len(res)
        if mtests > 0:
            cummax = 0.0
            for k, idx in enumerate(order_idx):
                adj = ps[idx] * (mtests - k)
                cummax = max(cummax, adj)
                p_holm[idx] = min(1.0, cummax)
        for j,(a,b,t,p,n) in enumerate(res):
            ph = p_holm[j] if (j < len(p_holm) and not np.isnan(p)) else np.nan
            posthoc_rows.append({"metric": m, "contrast": f"{b} - {a}", "t": t, "p_raw": p, "p_holm": ph, "n": n})


    posthoc_df = pd.DataFrame(posthoc_rows).sort_values(["metric","contrast"])
    print("\n=== Post-hoc pareados (Holm) ===")
    display(posthoc_df)


    # Descriptivos por modo (por si los quieres)
    desc = by_sm_bal.groupby("Modo")[z_cols].mean().T.round(3)
    print("\n=== Medias por Modo (z-rob) ===")
    display(desc)


    # Guardar resultados
    Path("/content").mkdir(exist_ok=True)
    anova_df.to_csv("/content/anova_rm_by_metric.csv", index=False)
    posthoc_df.to_csv("/content/posthoc_paired_t_holm.csv", index=False)
    desc.to_csv("/content/means_by_mode_zrob_from_zrows.csv")
    print("\nListo. Ficheros en /content: anova_rm_by_metric.csv, posthoc_paired_t_holm.csv, means_by_mode_zrob_from_zrows.csv")