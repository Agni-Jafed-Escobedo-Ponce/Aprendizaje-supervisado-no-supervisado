import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("="*60)
print("FASE II: DIVISIÓN DE DATOS + APRENDIZAJE NO SUPERVISADO")
print("="*60)

# =====================================================================
# PASO 6 Y 7: PREPARACIÓN DEL CONJUNTO Y DIVISIÓN 70/30
# =====================================================================
# Cargar el dataset limpio de la Fase I
df_pandas = pd.read_csv("covid_coahuila_limpio.csv")

print(f"Dataset cargado: {df_pandas.shape}")

# Separar variables predictoras (X) de la variable objetivo (y)
X = df_pandas.drop(columns=["defuncion"])
y = df_pandas["defuncion"]

print(f"Variables predictoras (X): {X.shape}")
print(f"Variable objetivo (y): {y.shape}")

# División 70% entrenamiento / 30% prueba (como pide el profe)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42,  # Semilla para reproducibilidad
    stratify=y        # Mantiene la proporción de clases (2.5% fallecidos) en ambos sets
)

print(f"\n✅ División completada:")
print(f"   - Entrenamiento: {X_train.shape[0]:,} registros ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   - Prueba: {X_test.shape[0]:,} registros ({X_test.shape[0]/len(X)*100:.1f}%)")

# =====================================================================
# PASO 8: NORMALIZACIÓN / ESCALAMIENTO (Min-Max)
# =====================================================================
# El profe dice: "Necesario para: KNN, redes neuronales, SVM"
# Como ya escalamos la edad en la Fase I, escalamos TODO el dataset ahora

scaler = MinMaxScaler()

# Ajustar el escalador SOLO con los datos de entrenamiento (¡buena práctica!)
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Transformar los datos de prueba con el mismo escalador
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print(f"\n✅ Normalización Min-Max completada (rango 0 a 1)")
print(f"   Rango de valores: [{X_train_scaled.min().min():.2f}, {X_train_scaled.max().max():.2f}]")

# =====================================================================
# FASE NO SUPERVISADA 1: PCA (Reducción de Dimensionalidad)
# =====================================================================
print("\n" + "="*60)
print("NO SUPERVISADO 1: PCA - Reducción de Dimensionalidad")
print("="*60)

# Aplicar PCA para visualizar las comorbilidades en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Crear un DataFrame con los componentes principales
pca_df = pd.DataFrame(
    data=X_pca,
    columns=['PC1', 'PC2']
)

print(f"✅ PCA completado: {X_train_scaled.shape[1]} variables → 2 componentes")
print(f"   Varianza explicada por PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"   Varianza explicada por PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"   Varianza total explicada: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Visualizar los 2 primeros componentes principales
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    pca_df['PC1'], 
    pca_df['PC2'], 
    c=y_train.values, 
    cmap='coolwarm', 
    alpha=0.5,
    s=10
)
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: Visualización de Pacientes COVID-19 en Coahuila')
plt.colorbar(scatter, label='Defunción (0=Sobrevivió, 1=Falleció)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_covid_coahuila.png", dpi=150)
print("✅ Gráfico PCA guardado como 'pca_covid_coahuila.png'")
plt.show()

# =====================================================================
# FASE NO SUPERVISADA 2: MÉTODO DEL CODO (Encontrar k óptima)
# =====================================================================
print("\n" + "="*60)
print("NO SUPERVISADO 2: Método del Codo para encontrar K óptima")
print("="*60)

# Probar diferentes valores de k (de 2 a 10)
inercias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inercias.append(kmeans.inertia_)

# Graficar el método del codo
plt.figure(figsize=(10, 6))
plt.plot(K_range, inercias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia (Suma de Errores Cuadrados)')
plt.title('Método del Codo: Encontrando el número óptimo de clusters')
plt.grid(True, alpha=0.3)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig("metodo_codo.png", dpi=150)
print("✅ Gráfico del codo guardado como 'metodo_codo.png'")
plt.show()

# =====================================================================
# FASE NO SUPERVISADA 3: K-MEANS (Clustering)
# =====================================================================
print("\n" + "="*60)
print("NO SUPERVISADO 3: K-Means - Identificación de Perfiles de Pacientes")
print("="*60)

# Basado en el codo, elegimos k=4 (puedes ajustar si ves otro valor en tu gráfica)
k_optimo = 4
print(f"Usando k={k_optimo} clusters (ajusta este valor según tu gráfica del codo)")

kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(X_train_scaled)
clusters_test = kmeans.predict(X_test_scaled)

# Agregar la variable "perfil_cluster" a nuestros datasets
X_train_con_cluster = X_train_scaled.copy()
X_train_con_cluster['perfil_cluster'] = clusters_train

X_test_con_cluster = X_test_scaled.copy()
X_test_con_cluster['perfil_cluster'] = clusters_test

print(f"✅ K-Means completado con {k_optimo} clusters")

# =====================================================================
# EVALUACIÓN DEL MODELO NO SUPERVISADO (OPTIMIZADO PARA 3GB DE RAM)
# =====================================================================
print(f"\n Evaluación del Clustering:")

# ️ CAMBIO IMPORTANTE: Bajamos a 10,000 para que quepa en tus 3GB de RAM libres
sample_size = 10000 

if len(X_train_scaled) > sample_size:
    # Seleccionar muestra aleatoria
    indices_muestra = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    X_muestra = X_train_scaled.iloc[indices_muestra]
    clusters_muestra = clusters_train[indices_muestra]
    
    print(f"   Calculando Silhouette Score con muestra de {sample_size:,} registros (para cuidar tu RAM)...")
    silhouette_avg = silhouette_score(X_muestra, clusters_muestra)
else:
    silhouette_avg = silhouette_score(X_train_scaled, clusters_train)

print(f"   - Silhouette Score: {silhouette_avg:.4f}")
print(f"     (Rango: -1 a 1. Más alto = mejor separación entre clusters)")

# Davies-Bouldin (Este sí se puede con todo el dataset, es muy ligero)
from sklearn.metrics import davies_bouldin_score
print(f"   Calculando Davies-Bouldin Index...")
davies_bouldin = davies_bouldin_score(X_train_scaled, clusters_train)
print(f"   - Davies-Bouldin Index: {davies_bouldin:.4f}")

# Distribución de pacientes
print(f"\n📊 Distribución de pacientes por cluster (Entrenamiento):")
cluster_counts = pd.Series(clusters_train).value_counts().sort_index()
for cluster, count in cluster_counts.items():
    porcentaje = count / len(clusters_train) * 100
    print(f"   - Cluster {cluster}: {count:,} pacientes ({porcentaje:.1f}%)")

# Tasa de mortalidad por cluster
print(f"\n📊 Tasa de mortalidad por cluster (¡Análisis Clínico!):")
df_analisis = X_train.copy()
df_analisis['cluster'] = clusters_train
df_analisis['defuncion'] = y_train.values

for cluster in range(k_optimo):
    pacientes_cluster = df_analisis[df_analisis['cluster'] == cluster]
    total = len(pacientes_cluster)
    fallecidos = pacientes_cluster['defuncion'].sum()
    tasa_mortalidad = (fallecidos / total) * 100 if total > 0 else 0
    print(f"   - Cluster {cluster}: {tasa_mortalidad:.2f}% de mortalidad ({fallecidos}/{total})")

# =====================================================================
# GRÁFICA DE CLUSTERS (OPTIMIZADA PARA NO TRABAR MATPLOTLIB)
# =====================================================================
# Graficar 352k puntos puede saturar la RAM de Matplotlib. 
# Usamos una muestra de 30,000 puntos para la visualización.
print("\n🎨 Generando gráfica de clusters (usando muestra para no saturar RAM)...")
sample_plot = 30000
indices_plot = np.random.choice(len(pca_df), sample_plot, replace=False)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    pca_df['PC1'].iloc[indices_plot], 
    pca_df['PC2'].iloc[indices_plot], 
    c=clusters_train[indices_plot], 
    cmap='viridis', 
    alpha=0.5,
    s=15
)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title(f'K-Means: {k_optimo} Perfiles de Pacientes COVID-19 en Coahuila')
plt.colorbar(scatter, label='Cluster asignado')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_clusters.png", dpi=150)
print("✅ Gráfico de clusters guardado como 'kmeans_clusters.png'")
plt.show()

# Guardar datasets para el siguiente paso
X_train_con_cluster = X_train_scaled.copy()
X_train_con_cluster['perfil_cluster'] = clusters_train
X_test_con_cluster = X_test_scaled.copy()
X_test_con_cluster['perfil_cluster'] = clusters_test

X_train_con_cluster.to_csv("X_train_con_cluster.csv", index=False)
X_test_con_cluster.to_csv("X_test_con_cluster.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n" + "="*60)
print("✅ ¡FASE II COMPLETADA!")
print("="*60)

# =====================================================================
# GUARDAR LOS DATOS LISTOS PARA EL APRENDIZAJE SUPERVISADO
# =====================================================================
# Guardar los datasets con la nueva variable "perfil_cluster"
X_train_con_cluster.to_csv("X_train_con_cluster.csv", index=False)
X_test_con_cluster.to_csv("X_test_con_cluster.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n" + "="*60)
print("✅ ¡FASE II COMPLETADA!")
print("="*60)
print("Archivos generados:")
print("   - X_train_con_cluster.csv")
print("   - X_test_con_cluster.csv")
print("   - y_train.csv")
print("   - y_test.csv")
print("   - pca_covid_coahuila.png")
print("   - metodo_codo.png")
print("   - kmeans_clusters.png")
print("\n🚀 ¡Listo para la FASE III: Aprendizaje Supervisado!")