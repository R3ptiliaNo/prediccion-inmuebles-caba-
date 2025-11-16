# visualize_metrics.py
"""
Script de visualización de métricas del TP Final.

Lee desde inmuebles.db:
    - resultados_modelo
    - resultados_por_fold

Hace:
    1) Imprime resumen de métricas promedio por modelo.
    2) Imprime resumen de métricas por fold.
    3) Genera gráficos en ./figures:
        - rmse_por_modelo.png
        - mae_por_modelo.png
        - r2_por_modelo.png
        - rmse_por_fold.png

Se ejecuta con:
    python visualize_metrics.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from src.db_utils import get_connection, DB_PATH


FIG_DIR = "figures"


def _load_metrics():
    """Carga tablas de métricas desde la BD."""
    with get_connection(DB_PATH) as conn:
        df_modelo = pd.read_sql("SELECT * FROM resultados_modelo", conn)
        df_fold = pd.read_sql("SELECT * FROM resultados_por_fold", conn)

    print("[visualize] Filas en resultados_modelo:", len(df_modelo))
    print("[visualize] Filas en resultados_por_fold:", len(df_fold))

    return df_modelo, df_fold


def _print_summary(df_modelo: pd.DataFrame, df_fold: pd.DataFrame):
    """Muestra por consola un resumen amigable de las métricas."""

    print("\n================= RESUMEN MÉTRICAS PROMEDIO POR MODELO =================")
    cols = ["modelo", "rmse", "mae", "r2", "n_folds", "timestamp"]
    cols_exist = [c for c in cols if c in df_modelo.columns]
    print(df_modelo[cols_exist].sort_values("rmse"))

    print("\n================= RESUMEN MÉTRICAS POR FOLD =================")
    cols_fold = ["modelo", "fold", "rmse", "mae", "r2"]
    cols_fold_exist = [c for c in cols_fold if c in df_fold.columns]
    print(df_fold[cols_fold_exist].sort_values(["modelo", "fold"]))


def _plot_rmse_por_modelo(df_modelo: pd.DataFrame):
    """Gráfico de barras de RMSE promedio por modelo."""

    os.makedirs(FIG_DIR, exist_ok=True)

    df_plot = df_modelo.groupby("modelo", as_index=False)["rmse"].mean()

    plt.figure()
    plt.bar(df_plot["modelo"], df_plot["rmse"])
    plt.title("RMSE promedio por modelo")
    plt.ylabel("RMSE")
    plt.xlabel("Modelo")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "rmse_por_modelo.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualize] Gráfico guardado: {out_path}")


def _plot_mae_por_modelo(df_modelo: pd.DataFrame):
    """Gráfico de barras de MAE promedio por modelo."""

    os.makedirs(FIG_DIR, exist_ok=True)

    df_plot = df_modelo.groupby("modelo", as_index=False)["mae"].mean()

    plt.figure()
    plt.bar(df_plot["modelo"], df_plot["mae"])
    plt.title("MAE promedio por modelo")
    plt.ylabel("MAE")
    plt.xlabel("Modelo")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "mae_por_modelo.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualize] Gráfico guardado: {out_path}")


def _plot_r2_por_modelo(df_modelo: pd.DataFrame):
    """Gráfico de barras de R² promedio por modelo."""

    os.makedirs(FIG_DIR, exist_ok=True)

    df_plot = df_modelo.groupby("modelo", as_index=False)["r2"].mean()

    plt.figure()
    plt.bar(df_plot["modelo"], df_plot["r2"])
    plt.title("R² promedio por modelo")
    plt.ylabel("R²")
    plt.xlabel("Modelo")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "r2_por_modelo.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualize] Gráfico guardado: {out_path}")


def _plot_rmse_por_fold(df_fold: pd.DataFrame):
    """
    Gráfico de RMSE por fold, diferenciando modelo.
    Queda como un gráfico de líneas (fold en X, RMSE en Y).
    """

    os.makedirs(FIG_DIR, exist_ok=True)

    plt.figure()

    for modelo, df_m in df_fold.groupby("modelo"):
        df_m = df_m.sort_values("fold")
        plt.plot(df_m["fold"], df_m["rmse"], marker="o", label=modelo)

    plt.title("RMSE por fold y por modelo")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "rmse_por_fold.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[visualize] Gráfico guardado: {out_path}")


def main():
    df_modelo, df_fold = _load_metrics()

    if df_modelo.empty or df_fold.empty:
        print("[visualize] No hay métricas cargadas en la BD. Corré primero main.py")
        return

    _print_summary(df_modelo, df_fold)
    _plot_rmse_por_modelo(df_modelo)
    _plot_mae_por_modelo(df_modelo)
    _plot_r2_por_modelo(df_modelo)   # ← acá sumamos el gráfico de R²
    _plot_rmse_por_fold(df_fold)

    print("\n[visualize] Listo. Gráficos generados en la carpeta 'figures/'.")


if __name__ == "__main__":
    main()
