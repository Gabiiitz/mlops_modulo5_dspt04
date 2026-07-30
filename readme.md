# Proyecto Integrador: Modelo de Riesgo Crediticio y Despliegue en Producción

**Entidad Financiera Target:** Sector Fintech / Servicios Financieros  
**Autor:** Gabii  
**Estado del Proyecto:** Completado y listo para revisión  

---

## 📋 Descripción del Proyecto

El presente proyecto abarca la construcción integral de una solución analítica y predictiva de scoring crediticio para evaluar el cumplimiento de pago de los clientes (`Pago_atiempo`). El desarrollo va desde la estructuración del entorno de trabajo, limpieza y análisis exploratorio de datos (EDA), hasta la ingeniería de características, entrenamiento de modelos, monitoreo de *data drift* y contenedorización para despliegue en producción mediante Docker y FastAPI.

---

## 🛠️ Estructura del Proyecto y Flujo de Trabajo en Git

Para garantizar un desarrollo ordenado, modular y colaborativo, se estableció la siguiente metodología de control de versiones utilizando **VSCode** y **Git Bash**:

1. **Configuración Inicial:** Creación del repositorio local en VSCode y sincronización con GitHub mediante Git Bash usando comandos estándar (`git commit`, `git push`).
2. **Estrategia de Ramas (Git Flow):** 
   * Se trabajó en la rama **`developer`** para la construcción inicial de scripts y pruebas.
   * Una vez validados los componentes, se realizó el correspondiente **Push Request (PR)** de `developer` hacia **`certification`** para pruebas integrales.
   * Tras la aprobación en certificación, se ejecutó el **Push Request** final de `certification` a la rama **`main`**.
3. **Sincronización:** Cada actualización en el repositorio remoto fue sincronizada localmente ejecutando `git pull` directamente desde la consola integrada de VSCode.

---

## 📊 Análisis Exploratorio de Datos (EDA) y Corrección de Datos

Se configuró el módulo `cargar_datos.py` para la ingesta del dataset en formato Excel. En el notebook del EDA (`.ipynb`), se realizaron análisis univariables, bivariables y multivariables acompañados de gráficos explicativos. Durante esta etapa se aplicaron las siguientes reglas de negocio y correcciones sobre el dataset:

* **Tratamiento de Outliers en Edad:** Se eliminaron los registros de clientes mayores a 90 años, entendiendo este valor como el límite razonable de responsabilidad crediticia para la cartera.
* **Normalización de Tendencia de Ingresos:** Se imputaron los valores nulos con la categoría `'Sin Especificar'`. Los valores numéricos (positivos, negativos y cero) se estandarizaron a las categorías ordinales `'Creciente'`, `'Decreciente'` y `'Estable'` para facilitar su interpretación y visualización gráfica.
* **Estandarización de Promedio de Ingresos DataCrédito:** Se aplicó un ajuste similar para agrupar y representar de manera más concisa la variable en las visualizaciones.
* **Imputación en Saldo Mora Codeudor:** Se imputaron los datos faltantes con `0` tras corroborar que no tenían injerencia significativa en los datos ni introducían sesgos.
* **Imputación en Saldos (Total, Principal y Mora):** Se imputaron con `0` los valores nulos, ya que se verificó que correspondían a clientes sin historial crediticio previo en la entidad.
* **Puntaje DataCrédito:** No se imputaron sus nulos debido a que representaban una cantidad insignificante de registros sin injerencia en la distribución general.
* **Casteo de Variable Target:** Se convirtió la variable `Pago_atiempo` a tipo booleano/entero para adecuar los tipos de datos en el DataFrame al requerimiento de los modelos.

---

## ⚙️ Paso a Paso Técnico: Transformación, Modelado, Monitoreo y Despliegue

A continuación se detalla el flujo de trabajo implementado en los módulos Python para la futura revisión:

### Paso 1: Ingeniería de Características (`ft_engineering.py`)
* Consume la función `cargardatos` de `cargar_datos.py`.
* Elimina la variable `puntaje` para evitar fuga de información (*data leakage*) y afectaciones artificiales a la predictibilidad.
* Procesa la fecha extrayendo `anio_prestamo` y `mes_prestamo`.
* Configura pipelines de preprocesamiento usando Scikit-Learn:
  * **Variables numéricas:** Imputación por mediana (`SimpleImputer`) y escalado estándar (`StandardScaler`).
  * **Variables categóricas:** Imputación por moda y codificación One-Hot (`OneHotEncoder`).
  * **Variables ordinales:** Mapeo jerárquico (`'Decreciente'`, `'Estable'`, `'Creciente'`) con `OrdinalEncoder`.
* Incluye las funciones auxiliares `summarize_classification()` para reporte de métricas y `build_model()` para ensamblar el pipeline.

### Paso 2: Entrenamiento y Evaluación (`model_training_evaluation.py`)
* Entrena y evalúa tres algoritmos: **Logistic Regression**, **Random Forest** y **XGBoost**.
* Realiza validación cruzada (5-fold cross-validation) para garantizar estabilidad.
* Exporta gráficos comparativos con la **Curva ROC-AUC**, la **Curva Precision-Recall** y una comparativa de **F1-Score**.
* Selecciona automáticamente el mejor modelo en función del F1-Score y lo exporta a disco como `best_model.pkl`.

### Paso 3: Monitoreo de Datos (`model_monitoring.py`)
* Implementa la prueba estadística de Kolmogorov-Smirnov (`ks_2samp`) para comparar distribuciones de referencia contra nuevos datos de producción.
* Evalúa la presencia de **Data Drift** (p-valor < 0.05) en variables críticas como `capital_prestado`, `salario_cliente`, `puntaje_datacredito` y `saldo_total`, generando gráficos comparativos de densidad.

### Paso 4: Despliegue en API y Docker (`model_deploy.py` y `Dockerfile`)
* Construye una API REST con **FastAPI** que expone el endpoint `/predict` para recibir solicitudes JSON con los datos del cliente y retornar la predicción e inferencia de probabilidad.
* Incluye un archivo `Dockerfile` para empaquetar la aplicación junto a sus dependencias (`requirements.txt`), garantizando su despliegue portable en cualquier entorno de producción.

---

## 💡 Insights de Negocio

A partir del análisis de datos y la evaluación del comportamiento de los clientes, se concluyen tres recomendaciones clave para la entidad financiera:

1. **Flexibilización de Plazos y Comodidad de Cuota:** A mayor plazo en meses, el cliente demuestra una menor tasa de morosidad y mayor propensión al pago puntual. Ofrecer esquemas de cuotas más cómodas reduce la presión financiera sobre el deudor y asegura un flujo de caja constante para la entidad.
2. **Planes Crediticios Diferenciados para Empleados:** Los clientes con tipo laboral "Empleado" exhiben mayor estabilidad en sus ingresos y mejor comportamiento histórico de pago. Se recomienda diseñar líneas de crédito preferenciales (menor tasa de interés o aprobación ágil) orientadas a este segmento.
3. **Ecosistema de Captación e Captura de Ingresos:** Se sugiere desarrollar un ecosistema o herramienta digital propia que incentive tanto a trabajadores independientes como empleados a registrar y respaldar sus ingresos (mensuales y anuales). Esto permitirá reducir la proporción de registros "Sin Especificar" en variables de ingresos, mejorando la precisión del scoring crediticio en futuras colocaciones.