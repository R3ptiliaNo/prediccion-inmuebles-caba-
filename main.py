# main.py
"""
TP Final Programación - Orquestador

Flujo:
1) Ejecuta data_pipeline: descarga datos, cleaning y guarda tablas en SQLite.
2) Ejecuta model_pipeline: entrena XGBoost + CatBoost, guarda métricas en SQLite.

Se corre con:
    python main.py
"""

from src.data_pipeline import run_data_pipeline
from src.model_pipeline import run_model_pipeline
from src.db_utils import DB_PATH, init_db


def main():
    # 0) Inicializar base de datos y tablas de resultados
    init_db()

    # 1) Extracción + limpieza -> guarda datos_raw y datos_limpios en BD
    run_data_pipeline()

    # 2) Modelado -> lee datos_limpios, entrena modelos, guarda métricas
    run_model_pipeline()

    print("\n[main] Proceso completo finalizado.")


if __name__ == "__main__":
    main()
