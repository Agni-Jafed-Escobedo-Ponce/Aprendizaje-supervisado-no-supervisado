import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("NO SUPERVISADO 4: Reglas de Asociación (Apriori) - CORREGIDO")
print("="*60)

# 1. Cargar el dataset limpio
print("\n📂 Cargando datos...")
df = pd.read_csv("covid_coahuila_limpio.csv")
print(f"   Registros: {len(df):,}")

# 2. Seleccionar solo las columnas de comorbilidades
cols_comorbilidades = [
    'diabetes', 'epoc', 'asma', 'inmusupr', 'hipertension', 
    'cardiovascular', 'obesidad', 'renal_cronica', 'tabaquismo', 
    'neumonia', 'intubado', 'uci'
]

# Convertir a booleano (True/False)
df_bool = df[cols_comorbilidades].astype(bool)

# 3. Ver prevalencia de cada comorbilidad (para entender los datos)
print("\n📊 Prevalencia de cada comorbilidad:")
for col in cols_comorbilidades:
    prevalencia = df_bool[col].mean() * 100
    print(f"   - {col}: {prevalencia:.2f}%")

# 4. Ejecutar Apriori con soporte más bajo (2% en lugar de 5%)
print("\n🔍 Buscando conjuntos frecuentes (min_support=0.02)...")
frequent_itemsets = apriori(df_bool, min_support=0.02, use_colnames=True)
print(f"   Conjuntos frecuentes encontrados: {len(frequent_itemsets)}")

if len(frequent_itemsets) > 0:
    # 5. Generar reglas de asociación con lift >= 1.0
    print("🔗 Generando reglas de asociación (min_lift=1.0)...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if len(rules) > 0:
        # Mostrar las 15 reglas más importantes ordenadas por "Lift"
        print(f"\n📊 Top 15 Reglas de Asociación (Ordenadas por Lift):")
        top_rules = rules.sort_values(by='lift', ascending=False).head(15)
        
        for idx, row in top_rules.iterrows():
            ant = ', '.join(row['antecedents'])
            con = ', '.join(row['consequents'])
            print(f"   SI ({ant}) → ENTONCES ({con})")
            print(f"      Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}")
            print()
        
        # Guardar todas las reglas en CSV
        rules.to_csv("reglas_asociacion.csv", index=False)
        print("✅ Todas las reglas guardadas en 'reglas_asociacion.csv'")
    else:
        print("⚠️ No se encontraron reglas con lift >= 1.0")
else:
    print("⚠️ No se encontraron conjuntos frecuentes con support >= 0.02")

# =====================================================================
# EVALUACIÓN: Matriz de Confusión (Regla Clínica)
# =====================================================================
print("\n" + "="*60)
print("📊 EVALUACIÓN: Regla Clínica Diabetes + Hipertensión")
print("="*60)

# Usar operación vectorizada (MUY rápido, sin iterrows)
# Predicción: Si tiene Diabetes Y Hipertensión → Alto Riesgo (1)
df['prediccion_regla'] = ((df['diabetes'] == 1) & (df['hipertension'] == 1)).astype(int)

y_real = df['defuncion']
y_pred_regla = df['prediccion_regla']

# Estadísticas de la regla
total_con_regla = df['prediccion_regla'].sum()
total_sin_regla = len(df) - total_con_regla
print(f"\n   Pacientes con Diabetes + Hipertensión: {total_con_regla:,}")
print(f"   Pacientes sin esa combinación: {total_sin_regla:,}")

# Calcular Matriz de Confusión
cm_regla = confusion_matrix(y_real, y_pred_regla)
TP = cm_regla[1, 1]
TN = cm_regla[0, 0]
FP = cm_regla[0, 1]
FN = cm_regla[1, 0]

# Métricas
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📋 Matriz de Confusión:")
print(f"   Verdaderos Positivos (TP): {TP:,} (Tiene D+H y Falleció)")
print(f"   Verdaderos Negativos (TN): {TN:,} (No tiene D+H y Sobrevivió)")
print(f"   Falsos Positivos (FP): {FP:,} (Tiene D+H pero Sobrevivió)")
print(f"   Falsos Negativos (FN): {FN:,} (No tiene D+H pero Falleció)")

print(f"\n📈 Métricas:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Visualización
plt.figure(figsize=(6, 4))
sns.heatmap(cm_regla, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Bajo Riesgo (0)', 'Alto Riesgo (1)'],
            yticklabels=['Sobrevivió (0)', 'Falleció (1)'])
plt.ylabel('Realidad (Defunción)')
plt.xlabel('Regla: Diabetes + Hipertensión')
plt.title('Matriz de Confusión: Regla D+H vs Defunción')
plt.tight_layout()
plt.savefig("confusion_matrix_asociacion.png", dpi=150)
print("\n✅ Gráfico guardado como 'confusion_matrix_asociacion.png'")
plt.show()

print("\n🎉 ¡Script 2.5 completado!")