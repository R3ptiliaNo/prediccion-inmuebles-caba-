# src/model_pipeline.py
"""
Pipeline de modelos:
- Lee 'datos_limpios' desde inmuebles.db
- Replica la lógica de BLOQUE 3b (CV estratificada) con XGBoost y CatBoost
- Calcula métricas (MAE, RMSE, R²) por fold y promedio
- Guarda métricas y modelo ganador en SQLite
"""

import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

from src.db_utils import get_connection, DB_PATH

SEED = 42


def _load_clean_data() -> pd.DataFrame:
    """
    Lee la tabla 'datos_limpios' de la BD y devuelve un DataFrame.
    """
    with get_connection(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM datos_limpios", conn)
    print("[model_pipeline] Datos limpios cargados desde BD. Filas:", len(df))
    return df


def _prepare_cv_data(df_res: pd.DataFrame):
    """
    Replica tu BLOQUE 3b:
    - df_cv
    - num_cols
    - base_features
    - X_all, y_all
    - y_bins (para StratifiedKFold)
    """
    df_cv = df_res.copy()

    num_cols = ["rooms", "bedrooms", "bathrooms",
                "surface_total", "surface_covered",
                "price", "lat", "lon"]

    for c in num_cols:
        if c in df_cv.columns:
            df_cv[c] = pd.to_numeric(df_cv[c], errors="coerce")

    df_cv["l3"] = df_cv["l3"].fillna("Desconocido")

    df_cv = df_cv[df_cv["surface_total"] > 0]
    df_cv = df_cv[df_cv["price"] > 0]

    precio_m2_all = df_cv["price"] / df_cv["surface_total"]
    low_q, high_q = precio_m2_all.quantile([0.01, 0.99])
    mask = (precio_m2_all >= low_q) & (precio_m2_all <= high_q)
    df_cv = df_cv[mask].copy()

    print("[model_pipeline] Filas luego de limpieza global para CV:", len(df_cv))

    base_features = [
        "lat", "lon",
        "rooms", "bedrooms", "bathrooms",
        "surface_total", "surface_covered",
        "property_type", "l3"
    ]

    X_all = df_cv[base_features].copy()
    y_all = df_cv["price"].copy()

    y_bins = pd.qcut(y_all, q=5, labels=False, duplicates="drop")
    print("[model_pipeline] Cantidad de bins (quintiles) para estratificación:",
          len(np.unique(y_bins)))

    return df_cv, X_all, y_all, y_bins


def _build_features_fold(X_train_base, y_train, X_val_base, n_clusters=6):
    """
    Copia literal de tu función build_features_fold, adaptada a módulo.
    """
    train_full = X_train_base.copy()
    train_full["price"] = y_train
    train_full = train_full[train_full["surface_total"] > 0].copy()
    train_full["precio_m2_real_train"] = train_full["price"] / train_full["surface_total"]

    precio_m2_barrio_train = (
        train_full.groupby("l3")["precio_m2_real_train"]
        .median()
    )

    X_train = X_train_base.copy()
    X_val = X_val_base.copy()

    global_med = precio_m2_barrio_train.median()
    X_train["precio_m2_barrio"] = X_train["l3"].map(precio_m2_barrio_train).fillna(global_med)
    X_val["precio_m2_barrio"] = X_val["l3"].map(precio_m2_barrio_train).fillna(global_med)

    barrio_zona = pd.qcut(
        precio_m2_barrio_train,
        q=4,
        labels=[0, 1, 2, 3]
    ).astype(int)

    X_train["zona_premium"] = X_train["l3"].map(barrio_zona).fillna(1).astype(int)
    X_val["zona_premium"] = X_val["l3"].map(barrio_zona).fillna(1).astype(int)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
    kmeans.fit(X_train[["lat", "lon"]])

    X_train["cluster_geo"] = kmeans.predict(X_train[["lat", "lon"]])
    X_val["cluster_geo"] = kmeans.predict(X_val[["lat", "lon"]])

    X_train_xgb = pd.get_dummies(
        X_train,
        columns=["property_type", "l3", "cluster_geo"],
        drop_first=True
    )
    X_val_xgb = pd.get_dummies(
        X_val,
        columns=["property_type", "l3", "cluster_geo"],
        drop_first=True
    )

    X_train_xgb, X_val_xgb = X_train_xgb.align(X_val_xgb, join="left", axis=1, fill_value=0)

    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    cat_features = [i for i, col in enumerate(X_train_cat.columns) if X_train_cat[col].dtype == "object"]

    return X_train_xgb, X_val_xgb, X_train_cat, X_val_cat, cat_features


def run_model_pipeline():
    """
    1) Carga datos_limpios de la BD.
    2) Prepara df_cv, X_all, y_all, y_bins.
    3) Ejecuta CV estratificada con XGBoost y CatBoost.
    4) Guarda métricas por fold y promedio en SQLite.
    5) Elige el mejor modelo por RMSE y lo deja registrado en config_modelo.
    """
    df_res = _load_clean_data()
    df_cv, X_all, y_all, y_bins = _prepare_cv_data(df_res)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    metrics_xgb = []
    metrics_cat = []

    rows_folds = []

    fold_num = 0

    for train_idx, val_idx in skf.split(X_all, y_bins):
        fold_num += 1
        print(f"\n[model_pipeline] ===== FOLD {fold_num} =====")

        X_train_base = X_all.iloc[train_idx].copy()
        X_val_base = X_all.iloc[val_idx].copy()
        y_train = y_all.iloc[train_idx].copy()
        y_val = y_all.iloc[val_idx].copy()

        X_train_xgb, X_val_xgb, X_train_cat, X_val_cat, cat_features = _build_features_fold(
            X_train_base, y_train, X_val_base, n_clusters=6
        )

        # --- XGBoost ---
        xgb_model_cv = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=SEED
        )

        xgb_model_cv.fit(X_train_xgb, y_train)
        pred_xgb_val = xgb_model_cv.predict(X_val_xgb)

        mae_x = mean_absolute_error(y_val, pred_xgb_val)
        rmse_x = np.sqrt(mean_squared_error(y_val, pred_xgb_val))
        r2_x = r2_score(y_val, pred_xgb_val)
        metrics_xgb.append((mae_x, rmse_x, r2_x))

        rows_folds.append({
            "modelo": "XGBoost",
            "fold": fold_num,
            "mae": mae_x,
            "rmse": rmse_x,
            "r2": r2_x
        })

        # --- CatBoost ---
        train_pool = Pool(X_train_cat, y_train, cat_features=cat_features)
        val_pool = Pool(X_val_cat, y_val, cat_features=cat_features)

        cat_model_cv = CatBoostRegressor(
            depth=8,
            learning_rate=0.05,
            n_estimators=400,
            random_state=SEED,
            loss_function="RMSE",
            verbose=False
        )

        cat_model_cv.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
        pred_cat_val = cat_model_cv.predict(val_pool)

        mae_c = mean_absolute_error(y_val, pred_cat_val)
        rmse_c = np.sqrt(mean_squared_error(y_val, pred_cat_val))
        r2_c = r2_score(y_val, pred_cat_val)
        metrics_cat.append((mae_c, rmse_c, r2_c))

        rows_folds.append({
            "modelo": "CatBoost",
            "fold": fold_num,
            "mae": mae_c,
            "rmse": rmse_c,
            "r2": r2_c
        })

        print(f"[model_pipeline] XGBoost  → MAE: {mae_x:,.0f} | RMSE: {rmse_x:,.0f} | R²: {r2_x:.3f}")
        print(f"[model_pipeline] CatBoost → MAE: {mae_c:,.0f} | RMSE: {rmse_c:,.0f} | R²: {r2_c:.3f}")

    metrics_xgb = np.array(metrics_xgb)
    metrics_cat = np.array(metrics_cat)

    def resumen_metrics(name, m):
        mae_mean, mae_std = m[:, 0].mean(), m[:, 0].std()
        rmse_mean, rmse_std = m[:, 1].mean(), m[:, 1].std()
        r2_mean, r2_std = m[:, 2].mean(), m[:, 2].std()
        return {
            "modelo": name,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std
        }

    res_xgb = resumen_metrics("XGBoost", metrics_xgb)
    res_cat = resumen_metrics("CatBoost", metrics_cat)

    print(f"\n[model_pipeline] ==== Resumen CV XGBoost ====")
    print(f"MAE: {res_xgb['mae_mean']:,.0f} ± {res_xgb['mae_std']:,.0f}")
    print(f"RMSE: {res_xgb['rmse_mean']:,.0f} ± {res_xgb['rmse_std']:,.0f}")
    print(f"R²: {res_xgb['r2_mean']:.3f} ± {res_xgb['r2_std']:.3f}")

    print(f"\n[model_pipeline] ==== Resumen CV CatBoost ====")
    print(f"MAE: {res_cat['mae_mean']:,.0f} ± {res_cat['mae_std']:,.0f}")
    print(f"RMSE: {res_cat['rmse_mean']:,.0f} ± {res_cat['rmse_std']:,.0f}")
    print(f"R²: {res_cat['r2_mean']:.3f} ± {res_cat['r2_std']:.3f}")

    # Elegimos mejor modelo por RMSE medio (menor es mejor)
    if res_xgb["rmse_mean"] <= res_cat["rmse_mean"]:
        mejor = "XGBoost"
        best_rmse = res_xgb["rmse_mean"]
    else:
        mejor = "CatBoost"
        best_rmse = res_cat["rmse_mean"]

    detalles = {
        "XGBoost": res_xgb,
        "CatBoost": res_cat
    }

    # Guardar todo en SQLite
    with get_connection(DB_PATH) as conn:
        # 1) métricas por fold
        df_folds = pd.DataFrame(rows_folds)
        df_folds.to_sql("resultados_por_fold", conn, if_exists="append", index=False)

        # 2) métricas promedio por modelo
        df_resumen = pd.DataFrame([
            {
                "modelo": "XGBoost",
                "rmse": res_xgb["rmse_mean"],
                "mae": res_xgb["mae_mean"],
                "r2": res_xgb["r2_mean"],
                "n_folds": metrics_xgb.shape[0],
                "criterio_seleccion": "RMSE",
                "timestamp": datetime.now().isoformat()
            },
            {
                "modelo": "CatBoost",
                "rmse": res_cat["rmse_mean"],
                "mae": res_cat["mae_mean"],
                "r2": res_cat["r2_mean"],
                "n_folds": metrics_cat.shape[0],
                "criterio_seleccion": "RMSE",
                "timestamp": datetime.now().isoformat()
            }
        ])
        df_resumen.to_sql("resultados_modelo", conn, if_exists="append", index=False)

        # 3) mejor modelo en config_modelo
        conn.execute(
            """
            INSERT INTO config_modelo (mejor_modelo, criterio, valor_criterio, detalles)
            VALUES (?, ?, ?, ?)
            """,
            (mejor, "RMSE", float(best_rmse), json.dumps(detalles))
        )

    print(f"\n[model_pipeline] Mejor modelo según RMSE medio: {mejor} (RMSE ≈ {best_rmse:,.0f})")
       # =====================================================
    
    # 6) Entrenar modelo XGBoost final con df_cv (71718 filas)
    #    y exportar artefactos (.pkl) para producción
    # =====================================================

    # Ojo: usamos X_all / y_all, que vienen de df_cv (no de datos_limpios completo)
    X_full_base = X_all.copy()
    y_full = y_all.copy()

    print("[model_pipeline] Filas usadas para entrenar el modelo final:", len(X_full_base))
    # Esto te tiene que imprimir 71718, igual que en el CV

    # Construimos las features completas (misma lógica que en tu BLOQUE 4)
    X_full_xgb, kmeans_final, precio_m2_barrio_full, barrio_zona_full = build_features_full(
        X_full_base,
        y_full,
        n_clusters=6
    )

    print("[model_pipeline] Shape X_full_xgb (dataset final para producción):", X_full_xgb.shape)

    # Entrenar modelo XGBoost final con TODO df_cv
    xgb_final = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=SEED
    )

    xgb_final.fit(X_full_xgb, y_full)

    # Exportar artefactos
    with open("modelo_xgboost_final.pkl", "wb") as f:
        pickle.dump(xgb_final, f)

    with open("kmeans_final.pkl", "wb") as f:
        pickle.dump(kmeans_final, f)

    with open("precio_m2_barrio_final.pkl", "wb") as f:
        pickle.dump(precio_m2_barrio_full, f)

    with open("zona_premium_map_final.pkl", "wb") as f:
        pickle.dump(barrio_zona_full, f)

    with open("xgb_feature_names.pkl", "wb") as f:
        pickle.dump(X_full_xgb.columns.tolist(), f)

    print("[model_pipeline] Modelos y artefactos exportados correctamente:")
    print(" - modelo_xgboost_final.pkl")
    print(" - kmeans_final.pkl")
    print(" - precio_m2_barrio_final.pkl")
    print(" - zona_premium_map_final.pkl")
    print(" - xgb_feature_names.pkl")
    print("[model_pipeline] Filas usadas para entrenar el modelo final:", len(X_full_base))
    # Esto te tiene que imprimir 71718, igual que en el CV


def build_features_full(X_base, y, n_clusters=6):
    """
    Versión completa (BLOQUE 4):
      - Crea precio_m2_barrio_full
      - Crea zona_premium_full
      - Ajusta KMeans final para cluster_geo
    Devuelve:
      - X_full_xgb: matriz lista para XGBoost (one-hot)
      - kmeans_final
      - precio_m2_barrio_full (Series por barrio)
      - barrio_zona_full (Series por barrio)
    """
    # ---- precio_m2_barrio_full ----
    aux = X_base.copy()
    aux["price"] = y
    aux = aux[aux["surface_total"] > 0].copy()
    aux["precio_m2_real"] = aux["price"] / aux["surface_total"]

    precio_m2_barrio_full = (
        aux.groupby("l3")["precio_m2_real"]
           .median()
    )

    global_med_full = precio_m2_barrio_full.median()

    X_feat = X_base.copy()
    X_feat["precio_m2_barrio"] = (
        X_feat["l3"].map(precio_m2_barrio_full)
                    .fillna(global_med_full)
    )

    # ---- zona_premium_full (0–3 por percentiles de precio_m2_barrio) ----
    barrio_zona_full = pd.qcut(
        precio_m2_barrio_full,
        q=4,
        labels=[0, 1, 2, 3]
    ).astype(int)

    X_feat["zona_premium"] = (
        X_feat["l3"].map(barrio_zona_full)
                    .fillna(1)
                    .astype(int)
    )

    # ---- cluster_geo final (KMeans sobre lat/lon de TODO el dataset) ----
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=SEED)
    kmeans_final.fit(X_feat[["lat", "lon"]])
    X_feat["cluster_geo"] = kmeans_final.labels_

    # ---- One-hot encoding para XGBoost ----
    X_full_xgb = pd.get_dummies(
        X_feat,
        columns=["property_type", "l3", "cluster_geo"],
        drop_first=True
    )

    return X_full_xgb, kmeans_final, precio_m2_barrio_full, barrio_zona_full

