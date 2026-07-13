import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*70)
print("FASE III: APRENDIZAJE SUPERVISADO")
print("Proyecto: Predicción de Mortalidad COVID-19 en Coahuila")
print("="*70)

# =====================================================================
# CARGA DE DATOS
# =====================================================================
print("\n📂 Cargando datos preparados...")
X_train = pd.read_csv("X_train_con_cluster.csv")
X_test = pd.read_csv("X_test_con_cluster.csv")

# Carga ultra-segura de y_train y y_test
# Leemos el CSV completo y extraemos la ÚLTIMA columna (siempre son los 0s y 1s)
y_train_df = pd.read_csv("y_train.csv")
y_train = y_train_df.iloc[:, -1]  # Extrae la última columna como Serie

y_test_df = pd.read_csv("y_test.csv")
y_test = y_test_df.iloc[:, -1]    # Extrae la última columna como Serie

print(f"   ✅ Entrenamiento: {X_train.shape[0]:,} registros, {X_train.shape[1]} variables")
print(f"   ✅ Prueba: {X_test.shape[0]:,} registros, {X_test.shape[1]} variables")
print(f"   ✅ Variable objetivo (defuncion) cargada correctamente.")
print(f"   ✅ Distribución en Train: {y_train.value_counts().to_dict()}")
# =====================================================================
# FUNCIÓN DE EVALUACIÓN
# =====================================================================
def evaluar_modelo(nombre, y_real, y_pred):
    """Calcula métricas y muestra matriz de confusión"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUACIÓN: {nombre}")
    print('='*70)
    
    # Métricas
    accuracy = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred, zero_division=0)
    recall = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    
    print(f"\n Métricas:")
    print(f"   • Accuracy (Exactitud):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Precision:             {precision:.4f} ({precision*100:.2f}%)")
    print(f"   • Recall (Sensibilidad): {recall:.4f} ({recall*100:.2f}%) ⭐")
    print(f"   • F1-Score:              {f1:.4f} ({f1*100:.2f}%)")
    
    # Matriz de confusión
    cm = confusion_matrix(y_real, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    print(f"\n📋 Matriz de Confusión:")
    print(f"   • Verdaderos Positivos (TP):  {TP:,} (Falleció y predijo Falleció)")
    print(f"   • Verdaderos Negativos (TN):  {TN:,} (Sobrevivió y predijo Sobrevivió)")
    print(f"   • Falsos Positivos (FP):      {FP:,} (Sobrevivió pero predijo Falleció)")
    print(f"   • Falsos Negativos (FN):      {FN:,} (Falleció pero predijo Sobrevivió) ")
    
    # Visualización
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sobrevivió (0)', 'Falleció (1)'],
                yticklabels=['Sobrevivió (0)', 'Falleció (1)'])
    plt.ylabel('Realidad (Defunción)')
    plt.xlabel('Predicción del Modelo')
    plt.title(f'Matriz de Confusión - {nombre}')
    plt.tight_layout()
    nombre_archivo = f"confusion_{nombre.lower().replace(' ', '_').replace('-', '')}.png"
    plt.savefig(nombre_archivo, dpi=150)
    plt.close()
    print(f"\n   ✅ Gráfico guardado: {nombre_archivo}")
    
    return {
        'modelo': nombre,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

# =====================================================================
# ALGORITMO 1: K-NEAREST NEIGHBORS (K-NN)
# =====================================================================
print("\n" + "="*70)
print("🔵 ALGORITMO 1: K-NEAREST NEIGHBORS (K-NN)")
print("="*70)

print("\n📖 a) ¿Para qué se utiliza?")
print("   K-NN es un algoritmo de clasificación supervisada que predice")
print("   la clase de un nuevo paciente basándose en los 'k' pacientes")
print("   más similares del conjunto de entrenamiento.")

print("\n b) Modelo a crear:")
print("   - Almacena todos los datos de entrenamiento")
print("   - Para cada nuevo paciente, calcula distancia euclidiana")
print("   - Selecciona los k vecinos más cercanos")
print("   - Asigna la clase mayoritaria de esos vecinos")
print("   - Usamos k=5 (valor óptimo común) y weights='distance'")

print("\n⚙️ c.i) Entrenamiento...")
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn.fit(X_train, y_train)
print("   ✅ Modelo entrenado")

print("\n🔮 c.ii) Prueba...")
y_pred_knn = knn.predict(X_test)
print("   ✅ Predicciones completadas")

# Evaluar
resultados_knn = evaluar_modelo("K-NN (k=5)", y_test, y_pred_knn)

# =====================================================================
# ALGORITMO 2: ÁRBOL DE DECISIÓN
# =====================================================================
print("\n" + "="*70)
print(" ALGORITMO 2: ÁRBOL DE DECISIÓN (CART)")
print("="*70)

print("\n a) ¿Para qué se utiliza?")
print("   Los Árboles de Decisión son algoritmos de clasificación que")
print("   crean reglas jerárquicas interpretables, similares a un")
print("   diagrama de flujo de preguntas sí/no.")

print("\n b) Modelo a crear:")
print("   - Crea nodos de decisión basados en las variables más importantes")
print("   - Cada nodo pregunta: '¿edad > 60?', '¿tiene diabetes?', etc.")
print("   - Las hojas del árbol son las predicciones finales")
print("   - Limitamos profundidad (max_depth=10) para evitar overfitting")

print("\n⚙️ c.i) Entrenamiento...")
arbol = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'
)
arbol.fit(X_train, y_train)
print("   ✅ Modelo entrenado")

print("\n🔮 c.ii) Prueba...")
y_pred_arbol = arbol.predict(X_test)
print("   ✅ Predicciones completadas")

# Evaluar
resultados_arbol = evaluar_modelo("Árbol de Decisión", y_test, y_pred_arbol)

# =====================================================================
# ALGORITMO 3: RED NEURONAL (MLP)
# =====================================================================
print("\n" + "="*70)
print("🧠 ALGORITMO 3: RED NEURONAL (MLPClassifier)")
print("="*70)

print("\n📖 a) ¿Para qué se utiliza?")
print("   Las Redes Neuronales son algoritmos de clasificación inspirados")
print("   en el cerebro humano. Son 'aproximadores universales' capaces")
print("   de aprender patrones complejos y no lineales en los datos.")

print("\n b) Modelo a crear:")
print("   - Capa de entrada: 20 neuronas (una por cada variable)")
print("   - Capas ocultas: 2 capas con 100 y 50 neuronas")
print("   - Capa de salida: 1 neurona (Sobrevivió o Falleció)")
print("   - Función de activación: ReLU (no lineal)")
print("   - Early stopping para evitar overfitting")

print("\n⚙️ c.i) Entrenamiento...")
# ✅ CORRECTO - Sin class_weight, pero con sample_weight para balancear
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    learning_rate_init=0.001,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

# Calcular pesos para balancear clases (alternativa a class_weight)
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)

mlp.fit(X_train, y_train, sample_weight=sample_weights)
print("   ✅ Modelo entrenado")

print("\n🔮 c.ii) Prueba...")
y_pred_mlp = mlp.predict(X_test)
print("   ✅ Predicciones completadas")

# Evaluar
resultados_mlp = evaluar_modelo("Red Neuronal", y_test, y_pred_mlp)

# =====================================================================
# COMPARACIÓN DE RESULTADOS
# =====================================================================
print("\n" + "="*70)
print("🏆 COMPARACIÓN FINAL DE ALGORITMOS")
print("="*70)

resultados = [resultados_knn, resultados_arbol, resultados_mlp]

# Crear DataFrame comparativo
df_comparacion = pd.DataFrame(resultados)
df_comparacion = df_comparacion[['modelo', 'accuracy', 'precision', 'recall', 'f1']]

print("\n📊 Tabla Comparativa:")
print(df_comparacion.to_string(index=False))

# Gráfica comparativa
plt.figure(figsize=(12, 7))
x = np.arange(len(resultados))
width = 0.2

plt.bar(x - width, [r['accuracy'] for r in resultados], width, label='Accuracy', color='skyblue')
plt.bar(x, [r['precision'] for r in resultados], width, label='Precision', color='lightgreen')
plt.bar(x + width, [r['recall'] for r in resultados], width, label='Recall', color='salmon')
plt.bar(x + 2*width, [r['f1'] for r in resultados], width, label='F1-Score', color='gold')

plt.xlabel('Algoritmo')
plt.ylabel('Puntuación')
plt.title('Comparación de Métricas por Algoritmo')
plt.xticks(x, ['K-NN', 'Árbol', 'Red Neuronal'])
plt.legend()
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("comparacion_algoritmos.png", dpi=150)
plt.close()
print("\n✅ Gráfica comparativa guardada: comparacion_algoritmos.png")

# =====================================================================
# OPTIMIZACIÓN (HYPERPARAMETER TUNING)
# =====================================================================
# Reemplaza la sección de GridSearchCV con esto:
print("\n🔍 Optimizando K-NN con GridSearchCV (versión rápida)...")

# Usar solo una muestra del 10% para velocidad
from sklearn.utils import resample
X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=35000, random_state=42)

# Grid más pequeño
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['distance']
}

grid_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid_knn,
    cv=2,  # Solo 2 folds
    scoring='f1',
    n_jobs=-1
)
grid_knn.fit(X_train_sample, y_train_sample)

print(f"   ✅ Mejores parámetros: {grid_knn.best_params_}")
print(f"   ✅ F1-Score optimizado: {grid_knn.best_score_:.4f}")

# =====================================================================
# CONCLUSIONES
# =====================================================================
print("\n" + "="*70)
print("📝 f) CONCLUSIONES Y RECOMENDACIONES")
print("="*70)

print("\n🎯 Según la rúbrica del profesor, el modelo con MEJOR Recall es:")
mejor_recall = max(resultados, key=lambda x: x['recall'])
print(f"   • {mejor_recall['modelo']} con Recall = {mejor_recall['recall']:.4f}")
print(f"   • Detectó {mejor_recall['TP']} de {mejor_recall['TP']+mejor_recall['FN']} fallecimientos reales")

print("\n💡 Justificación:")
print("   En problemas médicos con clases desbalanceadas (como COVID-19),")
print("   el Recall es MÁS IMPORTANTE que el Accuracy porque:")
print("   • Un Falso Negativo (FN) = Paciente que fallece pero el modelo")
print("     dijo que sobreviviría → ERROR GRAVE (no se le da atención)")
print("   • Un Falso Positivo (FP) = Paciente que sobrevive pero el modelo")
print("     dijo que fallecería → Menos grave (solo genera alerta temprana)")

print("\n✅ Archivos generados:")
print("   • confusion_k-nn_k=5.png")
print("   • confusion_árbol_de_decisión.png")
print("   • confusion_red_neuronal.png")
print("   • confusion_k-nn_optimizado.png")
print("   • comparacion_algoritmos.png")

print("\n🎉 ¡FASE III COMPLETADA!")
print("="*70)