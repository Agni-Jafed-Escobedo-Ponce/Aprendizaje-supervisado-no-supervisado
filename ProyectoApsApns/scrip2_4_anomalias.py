import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

print("="*60)
print("NO SUPERVISADO 3: Detección de Anomalías (Isolation Forest)")
print("="*60)

# 1. Cargar el dataset limpio
df = pd.read_csv("covid_coahuila_limpio.csv")
X = df.drop(columns=["defuncion"])
y = df["defuncion"]

# 2. Entrenar Isolation Forest
# contamination=0.05 significa que esperamos que el 5% de los datos sean anomalías
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
y_pred_iso = iso_forest.fit_predict(X)

# Isolation Forest devuelve 1 para "Normal" y -1 para "Anomalía"
# Lo convertimos a 1 (Anomalía) y 0 (Normal) para facilitar la matriz
anomalias = (y_pred_iso == -1).astype(int)

print(f"\n✅ Modelo entrenado.")
print(f"   - Pacientes normales: {(anomalias == 0).sum():,}")
print(f"   - Pacientes atípicos (Anomalías): {(anomalias == 1).sum():,} ({(anomalias == 1).mean()*100:.2f}%)")

# 3. Evaluación: Matriz de Confusión (Cruzando Anomalías vs Defunción Real)
# Para cumplir la rúbrica del profesor, evaluamos si las anomalías coinciden con los fallecimientos
# TP: El modelo detectó anomalía y el paciente falleció (Caso atípico grave)
# TN: El modelo detectó normalidad y el paciente sobrevivió
# FP: El modelo detectó anomalía pero el paciente sobrevivió
# FN: El modelo detectó normalidad pero el paciente falleció (Caso atípico no detectado)

cm_anomalias = confusion_matrix(y, anomalias)
TP = cm_anomalias[1, 1]
TN = cm_anomalias[0, 0]
FP = cm_anomalias[0, 1]
FN = cm_anomalias[1, 0]

print("\n📊 Matriz de Confusión (Anomalías vs Defunción Real):")
print(f"   - Verdaderos Positivos (TP): {TP} (Anomalía detectada y Falleció)")
print(f"   - Verdaderos Negativos (TN): {TN} (Normal y Sobrevivió)")
print(f"   - Falsos Positivos (FP): {FP} (Anomalía detectada pero Sobrevivió)")
print(f"   - Falsos Negativos (FN): {FN} (Normal pero Falleció)")

# Visualización
plt.figure(figsize=(6, 4))
sns.heatmap(cm_anomalias, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Anomalía (1)'], 
            yticklabels=['Sobrevivió (0)', 'Falleció (1)'])
plt.ylabel('Realidad (Defunción)')
plt.xlabel('Predicción (Isolation Forest)')
plt.title('Matriz de Confusión: Detección de Casos Atípicos')
plt.tight_layout()
plt.savefig("confusion_matrix_anomalias.png", dpi=150)
print("✅ Gráfico guardado como 'confusion_matrix_anomalias.png'")
plt.show()