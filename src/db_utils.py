# src/db_utils.py
import sqlite3
import os

DB_PATH = "./data/inmuebles.db"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Devuelve una conexión a la base SQLite.
    """
    return sqlite3.connect(db_path)


def init_db(db_path: str = DB_PATH):
    """
    Crea la base de datos y las tablas de resultados si no existen.

    NOTA:
    - Las tablas de datos crudos y limpios ('datos_raw', 'datos_limpios')
      las crea pandas.to_sql según las columnas del DataFrame.
    - Acá definimos solo las tablas 'estructuradas' de resultados.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None

    with get_connection(db_path) as conn:
        cur = conn.cursor()

        # Tabla de métricas generales por modelo
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resultados_modelo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                modelo TEXT,
                rmse REAL,
                mae REAL,
                r2 REAL,
                n_folds INTEGER,
                criterio_seleccion TEXT,   -- ej: 'RMSE'
                timestamp TEXT
            );
            """
        )

        # Tabla de métricas por fold (opcional, pero útil)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS resultados_por_fold (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                modelo TEXT,
                fold INTEGER,
                mae REAL,
                rmse REAL,
                r2 REAL
            );
            """
        )

        # Tabla de configuración / mejor modelo elegido
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS config_modelo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mejor_modelo TEXT,
                criterio TEXT,         -- 'RMSE' o 'MAE'
                valor_criterio REAL,
                detalles TEXT           -- JSON con info adicional
            );
            """
        )

        conn.commit()
        print(f"[db_utils] Base inicializada en {db_path}")
