Predicción de Mortalidad por COVID-19 en Coahuila
Aprendizaje Supervisado y No Supervisado - ECBD UTP
Este repositorio contiene el desarrollo completo del caso práctico "Identificación de Perfiles de Riesgo y Predicción de Letalidad en Pacientes COVID-19: Estudio de Caso en el Estado de Coahuila".
El proyecto aplica técnicas de minería de datos siguiendo la metodología del Prof. José Francisco Espinosa Garita, integrando primero el análisis no supervisado para descubrir patrones ocultos y posteriormente alimentando modelos supervisados para predecir la mortalidad.
📂 Estructura del Repositorio
Aprendizaje-supervisado-no-supervisado/
├── README.md                  # Documentación del proyecto
├── .gitignore                 # Archivos ignorados (datos pesados, outputs)
├── ProyectoApsApns/           # FASE 1: Preparación y Análisis No Supervisado
│   ├── scrip1.py              # Limpieza, transformación e ingeniería de características
│   ├── scrip2.py              # PCA, K-Means y Métricas de Clustering
│   ├── scrip2_4_anomalias.py  # Detección de anomalías con Isolation Forest
│   └── scrip2_5_asociacion.py # Reglas de asociación con Apriori
└── ProyectoAps/               # FASE 2: Aprendizaje Supervisado
    └── scrip3_supervisado.py  # K-NN, Árbol de Decisión y Red Neuronal (MLP)

🗂️ Descripción Detallada de Archivos
📁 Carpeta: ProyectoApsApns (No Supervisado y Preparación)
Esta carpeta contiene los scripts que procesan la base de datos cruda desde MongoDB/CSV y ejecutan las 4 categorías de algoritmos no supervisados requeridas por la rúbrica.
scrip1.py (Fase I: Preparación de la Data)
Carga y filtra exclusivamente pacientes residentes en Coahuila (ENTIDAD_RES == 5).
Convierte variables a binario aplicando la regla oficial (1=Sí, 0=No/No aplica/Se ignora).
Escala la variable edad usando MinMaxScaler.
Crea variables de ingeniería de características: totalComorbilidades, riesgoMetabolico, gravedad y adultoMayor.
Genera la variable objetivo defuncion interpretando la columna FECHA_DEF.
Output: covid_coahuila_limpio.csv
scrip2.py (Clustering y Reducción Dimensional)
Divide los datos 70% entrenamiento / 30% prueba con muestreo estratificado.
PCA: Reduce 19 variables a 2 componentes principales para visualización.
K-Means: Identifica 4 perfiles clínicos de pacientes usando el Método del Codo.
Evalúa clusters con Silhouette Score y Davies-Bouldin Index.
Output: X_train_con_cluster.csv, gráficas de PCA y K-Means.
scrip2_4_anomalias.py (Detección de Anomalías)
Aplica Isolation Forest para detectar casos atípicos (ej. jóvenes graves o adultos mayores sanos que fallecen).
Genera Matriz de Confusión cruzando anomalías vs defunción real.
scrip2_5_asociacion.py (Reglas de Asociación)
Aplica Apriori para encontrar combinaciones frecuentes de comorbilidades.
Descubre el "Triángulo Metabólico" (Diabetes + Hipertensión + Obesidad).
Evalúa la regla más fuerte como predictor de riesgo.
Carpeta: ProyectoAps (Supervisado)
Contiene el script final que entrena y evalúa los 3 algoritmos supervisados solicitados, utilizando los datos ya limpios y enriquecidos con los clusters de la fase anterior.
scrip3_supervisado.py (Modelos Predictivos)
K-NN: Clasificación basada en similitud clínica (optimizado con GridSearchCV).
Árbol de Decisión: Modelo interpretable con reglas jerárquicas.
Red Neuronal (MLP): Aproximador universal con capas ocultas (100, 50) y early stopping.
Evalúa todos los modelos con Accuracy, Precision, Recall, F1-Score y Matriz de Confusión.
Determina el mejor modelo según métricas clínicas relevantes.
⚙️ Requisitos Técnicos
Para ejecutar estos scripts necesitas tener instalado Python 3.x y las siguientes librerías:

pip install polars pandas scikit-learn matplotlib seaborn mlxtend pymongo
