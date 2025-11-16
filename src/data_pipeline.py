# src/data_pipeline.py
"""
Pipeline de datos:
- Descarga dataset de Kaggle (Properati)
- Cleaning básico + corrección lat/lon
- Geofiltrado por polígono de CABA (caba.json)
- Limpieza numérica fuerte + outliers + solo residenciales

Guarda en SQLite:
    - datos_raw       -> dataset crudo tras concatenar CSVs y corrección básica
    - datos_limpios   -> df_res (lo que usás para modelar)
"""

import os
import glob
import shutil
import numpy as np
import pandas as pd
import kagglehub
import geopandas as gpd

from src.db_utils import get_connection, DB_PATH

# Para el mapa, si querés seguir generándolo
import folium
from folium.plugins import MarkerCluster


def _download_and_concat() -> pd.DataFrame:
    """
    BLOQUE 2 - Parte 1:
    1) Descarga el dataset Properati desde KaggleHub.
    2) Busca recursivamente todos los CSV dentro de la carpeta descargada.
    3) Copia esos CSV a ./data (carpeta del proyecto).
    4) Lee y concatena todos los CSV desde ./data.

    Si Kaggle no trae CSV (por cambios en el dataset),
    al menos podés copiar los CSV manualmente a ./data y
    este mismo código los va a usar.
    """
    print("[data_pipeline] Descargando dataset Properati desde KaggleHub...")

    # 1) Descargar (o usar cache existente)
    path = kagglehub.dataset_download("alejandroczernikier/properati-argentina-dataset")
    print("[data_pipeline] Ruta de descarga KaggleHub:", path)

    # 2) Carpeta local del proyecto
    dest = "data"
    os.makedirs(dest, exist_ok=True)

    # 3) Buscar CSV recursivamente en la ruta de Kaggle
    files_kaggle = []
    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(".csv"):
                files_kaggle.append(os.path.join(root, fname))

    if files_kaggle:
        print("[data_pipeline] CSV encontrados en la carpeta de Kaggle (recursivo):")
        for f in files_kaggle:
            print("   ->", f)

        # 4) Copiar todos los CSV encontrados a ./data
        copied = []
        for src in files_kaggle:
            filename = os.path.basename(src)
            dst = os.path.join(dest, filename)
            shutil.copy(src, dst)
            copied.append(dst)

        print("[data_pipeline] CSV copiados a ./data de tu proyecto:")
        for c in copied:
            print("   ->", c)
    else:
        print("[data_pipeline] ATENCIÓN: KaggleHub no devolvió ningún .csv (ni en subcarpetas).")
        print("[data_pipeline] Se continuará usando SOLO los CSV que ya estén en ./data.")

    # 5) Leer todos los CSV que haya en ./data (vengan de Kaggle o copiados a mano)
    files_local = glob.glob(os.path.join(dest, "*.csv"))

    if not files_local:
        raise RuntimeError(
            "No se encontraron archivos CSV en la carpeta './data'.\n"
            "Opciones:\n"
            "  - Revisá si el dataset de Kaggle trae archivos comprimidos (zip/gz) y descomprimilos a mano en ./data.\n"
            "  - O bien copiá vos mismo los CSV de Properati a la carpeta './data'."
        )

    print("[data_pipeline] Leyendo CSV desde ./data:")
    df_list = []
    for f in files_local:
        print("   ->", f)
        df_list.append(pd.read_csv(f))

    df = pd.concat(df_list, ignore_index=True)

    print("[data_pipeline] Shape inicial concat:", df.shape)
    print("[data_pipeline] Columnas:", df.columns.tolist())
    return df






def _basic_filters_and_swap(df: pd.DataFrame) -> pd.DataFrame:
    """
    BLOQUE 2 - Parte 2:
    - Swappear lat/lon
    - Filtros básicos: Venta, precio>0, l2 == Capital Federal
    - Limpieza de precios truchos, conversión ARS->USD 2019/2020
    - Drop de columnas que no se usan
    """
    # Swap lat/lon
    tmp_lat = df["lat"].copy()
    df["lat"] = df["lon"]
    df["lon"] = tmp_lat

    print("[data_pipeline] Rango lat:", df["lat"].min(), "→", df["lat"].max())
    print("[data_pipeline] Rango lon:", df["lon"].min(), "→", df["lon"].max())

    # Filtro por tipo de operación si existe la columna
    if "operation_type" in df.columns:
        df = df[df["operation_type"] == "Venta"]

    # Precio positivo
    df = df[df["price"] > 0]

    # Solo Capital Federal (CABA) según l2 si existe
    if "l2" in df.columns:
        df = df[df["l2"] == "Capital Federal"]

    # Limpieza de precios "truchos"
    precios_invalidos = [111111111, 11]
    df = df[~df["price"].isin(precios_invalidos)]

    # Conversión ARS→USD 2019/2020
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

        mask_2019_ars = (df["currency"] == "ARS") & (df["start_date"].dt.year == 2019)
        mask_2020_ars = (df["currency"] == "ARS") & (df["start_date"].dt.year == 2020)

        df.loc[mask_2019_ars, "price"] = df.loc[mask_2019_ars, "price"] / 80
        df.loc[mask_2020_ars, "price"] = df.loc[mask_2020_ars, "price"] / 195
        df.loc[mask_2019_ars | mask_2020_ars, "currency"] = "USD"

    print("[data_pipeline] Shape tras filtros básicos y conversión:", df.shape)

    cols_drop = [
        "l1", "ad_type", "l4", "l5", "l6",
        "title", "description", "price_period",
        "operation_type", "start_date", "end_date",
        "created_on", "currency", "l2"
    ]
    cols_to_drop_existing = [c for c in cols_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop_existing)

    print("[data_pipeline] Shape tras drop columnas:", df.shape)
    print("[data_pipeline] Columnas actuales:", df.columns.tolist())
    return df


def _geofilter_caba(df: pd.DataFrame) -> pd.DataFrame:
    """
    BLOQUE 2 - Parte 3:
    Geofiltrado: solo puntos dentro de CABA según caba.json
    """
    print("[data_pipeline] Filas antes de lat/lon:", len(df))
    df = df.dropna(subset=["lat", "lon"])
    print("[data_pipeline] Filas luego de eliminar lat/lon vacíos:", len(df))

    caba = gpd.read_file("caba.json")

    if caba.crs is not None and caba.crs.to_string() != "EPSG:4326":
        caba = caba.to_crs("EPSG:4326")

    try:
        caba_union = caba.unary_union
    except AttributeError:
        caba_union = caba.geometry.unary_union

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    gdf["inside_caba"] = gdf.within(caba_union)

    fuera = gdf[~gdf["inside_caba"]]
    print("[data_pipeline] Puntos fuera de CABA:", len(fuera))

    gdf_caba = gdf[gdf["inside_caba"]].copy()
    print("[data_pipeline] Filas finales dentro de CABA:", len(gdf_caba))

    gdf_caba_no_geom = gdf_caba.drop(columns=["geometry", "inside_caba"])
    gdf_caba_no_geom.to_csv("caba_geo_clean.csv", index=False)
    print("[data_pipeline] Nuevo dataset guardado: caba_geo_clean.csv")

    # Mapa (opcional, pero lo mantengo por si lo querés para la entrega)
    mapa = gdf_caba_no_geom.copy()
    m = folium.Map(location=[-34.60, -58.44], zoom_start=11, tiles="cartodbpositron")
    marker_cluster = MarkerCluster().add_to(m)
    if len(mapa) > 8000:
        mapa_sample = mapa.sample(n=8000, random_state=42)
    else:
        mapa_sample = mapa

    for _, row in mapa_sample.iterrows():
        if pd.notnull(row["lat"]) and pd.notnull(row["lon"]):
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=2,
                fill=True,
                fill_opacity=0.6,
                color="red",
            ).add_to(marker_cluster)

    m.save("mapa_validacion_caba.html")
    print("[data_pipeline] Mapa generado: mapa_validacion_caba.html")

    return gdf_caba_no_geom


def _numeric_cleaning_and_outliers(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    BLOQUE 7:
    Limpieza numérica fuerte + outliers por tipo + solo residenciales
    """
    df_clean = df_in.copy()
    print("[data_pipeline] Shape inicial para limpieza numérica:", df_clean.shape)

    num_cols = ["rooms", "bedrooms", "bathrooms", "surface_total", "surface_covered"]
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    before_incoh = len(df_clean)

    df_clean = df_clean[df_clean["surface_total"].isna() | (df_clean["surface_total"] > 0)]
    df_clean = df_clean[df_clean["surface_covered"].isna() | (df_clean["surface_covered"] >= 0)]

    mask_rel = (
        df_clean["surface_covered"].isna() |
        df_clean["surface_total"].isna() |
        (df_clean["surface_covered"] <= df_clean["surface_total"])
    )
    df_clean = df_clean[mask_rel]

    for col in ["rooms", "bedrooms", "bathrooms"]:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col].isna() | (df_clean[col] >= 0)]

    print("[data_pipeline] Eliminados por incoherencias:", before_incoh - len(df_clean))
    print("[data_pipeline] Shape tras incoherencias:", df_clean.shape)

    def remove_outliers_by_type(df_in, col, k=1.5):
        if df_in.empty:
            return df_in
        frames = []
        for ptype in df_in["property_type"].dropna().unique():
            subset = df_in[df_in["property_type"] == ptype].copy()

            Q1 = subset[col].quantile(0.25)
            Q3 = subset[col].quantile(0.75)
            IQR = Q3 - Q1

            if pd.isna(IQR) or IQR == 0:
                frames.append(subset)
                continue

            lower = Q1 - k * IQR
            upper = Q3 + k * IQR
            mask = subset[col].isna() | subset[col].between(lower, upper)
            frames.append(subset[mask])

        return pd.concat(frames, ignore_index=True) if frames else df_in

    before_out = len(df_clean)
    for col in num_cols:
        df_clean = remove_outliers_by_type(df_clean, col)
        print(f"[data_pipeline] Shape tras outliers en {col}: {df_clean.shape}")

    print("[data_pipeline] Eliminados total por outliers:", before_out - len(df_clean))

    residenciales = ["Departamento", "PH", "Casa", "Casa de campo"]
    df_res = df_clean[df_clean["property_type"].isin(residenciales)].copy()

    print("[data_pipeline] Filas residenciales (df_res):", len(df_res))
    print("[data_pipeline] Columnas df_res:", df_res.columns.tolist())

    return df_res


def run_data_pipeline():
    """
    Función principal de pipeline de datos:
      1) Descarga y concatena CSVs (df_raw) -> tabla 'datos_raw'
      2) Cleaning básico + CABA geográfica + outliers (df_res) -> tabla 'datos_limpios'
    """
    # 1) Descarga + concat
    df_raw = _download_and_concat()

    # Guardar df_raw en SQLite
    with get_connection(DB_PATH) as conn:
        df_raw.to_sql("datos_raw", conn, if_exists="replace", index=False)
        print("[data_pipeline] Tabla 'datos_raw' guardada en BD.")

    # 2) Cleaning + geofiltrado + outliers + residenciales
    df_basic = _basic_filters_and_swap(df_raw)
    df_caba = _geofilter_caba(df_basic)
    df_res = _numeric_cleaning_and_outliers(df_caba)

    # Guardar df_res en SQLite
    with get_connection(DB_PATH) as conn:
        df_res.to_sql("datos_limpios", conn, if_exists="replace", index=False)
        print("[data_pipeline] Tabla 'datos_limpios' guardada en BD.")
