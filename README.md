
# 📌 **TP Final – Programación Avanzada**

## **Predicción del precio de propiedades en CABA usando XGBoost + CatBoost**

**Autor:** *Asado Analytics*
**Año:** 2025

**Integrantes:** 
  - Gastón Conessa
  - Fernando Nieto Benavidez
  - Alan Aramayo
  - Alejandro Sacndinaro
  - Juan Ignacio Failache

---

# 🧠 **Descripción general del proyecto**

Este trabajo práctico implementa un **pipeline completo de Ciencia de Datos y Machine Learning**, cuyo objetivo es **predecir el precio de propiedades residenciales en CABA** utilizando datos del dataset **Properati Argentina** (Kaggle).

El proyecto incluye:

✔ Descarga automática de datos desde Kaggle
✔ ETL + limpieza avanzada
✔ Geofiltrado con GeoPandas
✔ Detección de outliers por tipo de propiedad
✔ Construcción de features geoespaciales
✔ Cross-Validation estratificada por rangos de precio
✔ Comparación de modelos: **XGBoost vs. CatBoost**
✔ Entrenamiento final del mejor modelo (**XGBoost**)
✔ Exportación de artefactos `.pkl`
✔ Almacenamiento completo en **SQLite**
✔ Generación de gráficos de métricas
✔ Dashboard funcional en **Streamlit** para probar el modelo

---

# 🎯 **Objetivos del trabajo**

### ✔ Construcción del modelo

* Implementar **pipelines de preprocesamiento**.
* Comparar al menos **dos modelos de regresión** (XGBoost y CatBoost).
* Usar métricas: **MAE, RMSE, R²**.
* Realizar **Cross-Validation estratificada** para evitar fuga de información.

### ✔ Persistencia en Base de Datos

Toda la información se guarda en SQLite:

| Tabla                   | Descripción                      |
| ----------------------- | -------------------------------- |
| **datos_raw**           | CSV de Kaggle sin procesar       |
| **datos_limpios**       | Dataset filtrado y curado        |
| **resultados_por_fold** | Métricas de CV por fold          |
| **resultados_modelo**   | Métricas promedio por modelo     |
| **config_modelo**       | Parámetros de XGBoost y CatBoost |

### ✔ Visualizaciones

Se generan gráficos automáticos:

* RMSE por fold
* RMSE promedio por modelo
* MAE promedio por modelo
* R² promedio por modelo

### ✔ Exportación del modelo final

* `modelo_xgboost_final.pkl`
* `kmeans_final.pkl`
* `precio_m2_barrio_final.pkl`
* `zona_premium_map_final.pkl`
* `xgb_feature_names.pkl`

---

# 📁 **Estructura del proyecto**

```
tp_programacion/
│
├── data/                         # CSV descargados desde Kaggle
├── inmuebles.db                  # Base SQLite con todo el pipeline
│
├── src/
│   ├── db_utils.py               # Creación/lectura de SQLite
│   ├── data_pipeline.py          # ETL + limpieza + geofiltrado
│   ├── model_pipeline.py         # CV, métricas y modelos
│
├── main.py                       # Ejecuta TODO el pipeline end-to-end
├── visualize_metrics.py          # Gráficos de métricas
│
├── modelo_xgboost_final.pkl      # Modelo final exportado
├── kmeans_final.pkl
├── precio_m2_barrio_final.pkl
├── zona_premium_map_final.pkl
├── xgb_feature_names.pkl
│-- caba.json                    # geojson para hacer filtrado espacial
├── requirements.txt
└── README.md                     # Este archivo
```

---

# 🚀 **Flujo completo del pipeline**

El proyecto está diseñado para correr **con un solo comando**:

```
python main.py
```

### Lo que hace:

### **1) Descarga el Dataset**

Usa `kagglehub` para bajar:

```
properati-argentina-dataset
```

Luego **copia esos CSV a la carpeta local `/data`** del proyecto.

Si no trae el csv de Kaggle descargarlo de la pagina :https://www.kaggle.com/datasets/alejandroczernikier/properati-argentina-dataset

y dejarlo en carpeta DATA
---

### **2) ETL + Limpieza**

Incluye:

* Corrección de lat/lon invertidos
* Conversión ARS → USD 2019 / 2020
* Eliminación de columnas irrelevantes
* Validación geográfica contra `caba.json`
* Limpieza numérica fuerte
* Outliers por tipo de propiedad (IQR)
* Filtro: solo propiedades residenciales

El dataset final se guarda en:

📌 `datos_limpios` (≈ 93.000 filas)

---

### **3) Generación del dataset de CV**

Se aplica una limpieza adicional:

* Filtrado por precio/m² fuera del 1%-99%
* Solo filas confiables

📌 Resultado: **aprox. 71.718 registros**
Este dataset se usa tanto para **CV** como para el **modelo final**.

---

### **4) Cross-Validation estratificada**

* Estratificación por quintiles de precio.
* Construcción de features sin leakage:

  * `precio_m2_barrio`
  * `zona_premium (0–3)`
  * `cluster_geo` con KMeans
* Comparación de:

  * **XGBoost**
  * **CatBoost**

Cada fold genera: MAE, RMSE, R².

Se guardan en:

📌 `resultados_por_fold`
📌 `resultados_modelo`
📌 `config_modelo`

---

### **5) Entrenamiento final del mejor modelo**

El modelo ganador fue:

## ⭐ **XGBoost**

Se entrena con las mismas 71.718 filas usadas para CV.

Se exportan todos los artefactos a `.pkl` para producción.

---

### **6) Visualización de métricas**

Se corre con:

```
python visualize_metrics.py
```

Genera:

* `figures/rmse_por_fold.png`
* `figures/rmse_por_modelo.png`
* `figures/mae_por_modelo.png`
* `figures/r2_por_modelo.png`

---

# ▶️ **Cómo ejecutar localmente**

### 1) Crear entorno y activar

```
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

### 2) Instalar dependencias

```
pip install -r requirements.txt
```

### 3) Ejecutar todo el pipeline

```
python main.py
```

### 4) Visualizar métricas

```
python visualize_metrics.py
```

---

# 🌐 **Deploy en Streamlit (segundo repositorio)**

Este proyecto fue complementado con un repositorio aparte donde se implementó una **aplicación Streamlit** para consumir el modelo final entrenado.

### 🔗 **Repositorio del deploy**

👉 [https://github.com/R3ptiliaNo/prediccion-inmuebles](https://github.com/R3ptiliaNo/prediccion-inmuebles)

### 🔗 **App online funcionando**

👉 [https://prediccion-inmuebles-caba.streamlit.app/#prediccion-de-precio-de-propiedades-en-caba](https://prediccion-inmuebles-caba.streamlit.app/#prediccion-de-precio-de-propiedades-en-caba)

La app permite:

* Cargar ubicación, ambientes, superficie y tipo de propiedad
* Preprocesar la entrada con los mismos steps del pipeline
* Generar predicciones utilizando el modelo exportado `.pkl`


---

# 🏁 **Conclusiones**

El proyecto cumple con:

✔ Buenas prácticas de organización y modularización
✔ Pipelines claros de ETL y modelado
✔ Evaluación rigurosa con CV estratificada
✔ Comparación transparente de modelos
✔ Persistencia completa en SQLite
✔ Visualizaciones para análisis
✔ Modelo deployado en la web para validación real

El flujo desde datos en crudo → modelo final → app web está completamente integrado, profesional y reproducible.

---


