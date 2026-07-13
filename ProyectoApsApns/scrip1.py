import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("="*60)
print("FASE I: PREPARACIÓN DE LA DATA (Enfoque Clínico-Epidemiológico)")
print("="*60)

# =====================================================================
# 1. CARGA DE DATOS (Polars para máxima velocidad)
# =====================================================================
# Si lees desde CSV (recomendado para pruebas rápidas):
df = pl.read_csv("covid19.casos_coahuila_final_datos.csv", ignore_errors=True)

print(f"Registros cargados originalmente: {df.height:,}")

# =====================================================================
# FILTRO GEOGRÁFICO: Solo Coahuila (ENTIDAD_RES == 5)
# =====================================================================
df = df.filter(pl.col("ENTIDAD_RES") == 5)
print(f"Registros en Coahuila: {df.height:,}")

# =====================================================================
# INCISO d) Crear columna para interpretar datos (Variable Objetivo)
# Regla del Profe: Si FECHA_DEF tiene fecha -> 1 (Falleció), si es nulo/vacío -> 0
# =====================================================================
df = df.with_columns(
    pl.when(pl.col("FECHA_DEF").is_not_null() & (pl.col("FECHA_DEF").cast(pl.Utf8).str.strip_chars() != ""))
    .then(1)
    .otherwise(0)
    .alias("defuncion")
)

# =====================================================================
# INCISO a) Convertir variables a binario
# =====================================================================
# 1. SEXO: El catálogo dice 1=Mujer, 2=Hombre. La rúbrica pide 0=Mujer, 1=Hombre.
df = df.with_columns(
    pl.when(pl.col("SEXO") == 2).then(1).otherwise(0).alias("sexo_binario")
)

# 2. COMORBILIDADES Y VARIABLES CLÍNICAS:
# Regla del Profe: 1=Sí -> 1. Todo lo demás (2=No, 97=No aplica, 98=Se ignora, 99=No esp) -> 0
cols_si_no = [
    "INTUBADO", "NEUMONIA", "DIABETES", "EPOC", "ASMA", "INMUSUPR", 
    "HIPERTENSION", "CARDIOVASCULAR", "OBESIDAD", "RENAL_CRONICA", 
    "TABAQUISMO", "UCI", "EMBARAZO"
]

expresiones_binarias = []
for col in cols_si_no:
    if col in df.columns:
        expresiones_binarias.append(
            pl.when(pl.col(col) == 1).then(1).otherwise(0).alias(col.lower())
        )

df = df.with_columns(expresiones_binarias)

# =====================================================================
# 🛠️ CORRECCIÓN: RENOMBRAR VARIABLES RESTANTES A MINÚSCULAS
# =====================================================================
# Esto asegura que 'EDAD' y 'TIPO_PACIENTE' pasen a minúsculas para Pandas
df = df.with_columns([
    pl.col("EDAD").cast(pl.Int32).alias("edad"),
    pl.col("TIPO_PACIENTE").alias("tipo_paciente")
])

# =====================================================================
# INCISO c) Crear variables con sumas de valores (Ingeniería de Características)
# =====================================================================
df = df.with_columns([
    # Suma de todas las comorbilidades
    (pl.col("diabetes") + pl.col("epoc") + pl.col("asma") + pl.col("inmusupr") + 
     pl.col("hipertension") + pl.col("cardiovascular") + pl.col("obesidad") + 
     pl.col("renal_cronica") + pl.col("tabaquismo")).alias("totalComorbilidades"),
    
    # Suma de riesgo metabólico
    (pl.col("diabetes") + pl.col("hipertension") + pl.col("obesidad")).alias("riesgoMetabolico"),
    
    # Suma de gravedad respiratoria
    (pl.col("neumonia") + pl.col("intubado") + pl.col("uci")).alias("gravedad"),
    
    # Variable extra del Profe: adultoMayor (1 si >= 60, 0 si no)
    pl.when(pl.col("edad") >= 60).then(1).otherwise(0).alias("adultoMayor")
])

# =====================================================================
# SELECCIÓN FINAL DE COLUMNAS PARA EL MODELO
# =====================================================================
# Nos quedamos solo con las variables numéricas/binarias útiles para la minería
columnas_finales = [
    "edad", "sexo_binario", "adultoMayor", "totalComorbilidades", 
    "riesgoMetabolico", "gravedad", "neumonia", "diabetes", "epoc", "asma", 
    "inmusupr", "hipertension", "cardiovascular", "obesidad", "renal_cronica", 
    "tabaquismo", "intubado", "uci", "tipo_paciente", "defuncion"
]

# Filtrar solo las que existen en el dataframe
columnas_existentes = [c for c in columnas_finales if c in df.columns]
df_limpio_polars = df.select(columnas_existentes)

# =====================================================================
# EL PUENTE: Pasamos de Polars a Pandas para el Machine Learning
# =====================================================================
df_pandas = df_limpio_polars.to_pandas()

# =====================================================================
# INCISO b) Escalar variables (La Edad)
# Usamos MinMaxScaler para dejar la edad entre 0 y 1 (como pide el profe)
# =====================================================================
scaler = MinMaxScaler()
df_pandas["edad_escalada"] = scaler.fit_transform(df_pandas[["edad"]])

# Eliminamos la edad original para que no haya duplicidad en los modelos
df_pandas = df_pandas.drop(columns=["edad"])

print("\n✅ ¡Fase I Completada!")
print(f"Dimensiones finales del dataset: {df_pandas.shape}")
print("\nPrimeras 5 filas del dataset listo para Minería de Datos:")
print(df_pandas.head())

print("\nDistribución de la Variable Objetivo (Defunción):")
print(df_pandas["defuncion"].value_counts())

# =====================================================================
# GUARDAR EL DATASET LIMPIO (CHECKPOINT)
# =====================================================================
df_pandas.to_csv("covid_coahuila_limpio.csv", index=False, encoding='utf-8')
print("\n✅ ¡Listo! El archivo 'covid_coahuila_limpio.csv' se guardó en tu carpeta.")