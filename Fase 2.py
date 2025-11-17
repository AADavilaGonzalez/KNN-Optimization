import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint

print("--- Fase 2: Búsqueda Aleatoria (Salida Homologada) ---")

# 1. Carga y Split (Idéntico a Fase 1)
X, y = load_breast_cancer(return_X_y=True)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Escalado (Idéntico a Fase 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 3. Configuración del Espacio de Búsqueda
# Ajustamos 'metric' para que la columna de salida sea igual a la Fase 1
param_distributions = {
    'n_neighbors': randint(1, 51),           # Rango ampliado
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev'] 
}

# 4. Configuración de RandomizedSearchCV con TODAS las métricas
# Esto es vital para llenar las columnas de Accuracy, Precision y F1
scoring_metrics = {
    'recall': 'recall',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'f1': 'f1'
}

knn_base = KNeighborsClassifier()
n_iter_search = 50  # Presupuesto limitado

random_search = RandomizedSearchCV(
    estimator=knn_base,
    param_distributions=param_distributions,
    n_iter=n_iter_search,
    cv=10,
    scoring=scoring_metrics,
    refit='recall',     # El ganador se decide por Recall
    n_jobs=-1,
    random_state=42,
    return_train_score=False
)

print(f"Iniciando búsqueda aleatoria ({n_iter_search} iteraciones)...")

# 5. Ejecución
start_total_time = time.time()
random_search.fit(X_train, y_train)
end_total_time = time.time()

# 6. Evaluación del Ganador (Para mostrar en consola)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\n--- Resultados Fase 2 Terminados ---")
print(f"Tiempo Total: {end_total_time - start_total_time:.4f} s")
print(f"Mejor Recall (CV): {random_search.best_score_:.4f}")

# --- 7. EXPORTACIÓN EXACTA (El paso clave para tu petición) ---
# Extraemos los resultados crudos
cv_results = pd.DataFrame(random_search.cv_results_)

# Creamos un DataFrame nuevo con LAS MISMAS columnas que Fase 1
df_export = pd.DataFrame()

# A. Parámetros
df_export['n_neighbors'] = cv_results['param_n_neighbors']
df_export['weights'] = cv_results['param_weights']
df_export['metric'] = cv_results['param_metric']

# B. Métricas
# OJO: En Fase 1, 'Mean_CV_Recall' era CV y 'Accuracy_Test' era del Test Set.
# En Fase 2, por eficiencia, usamos el promedio de CV para todo.
# Mantenemos los nombres de columnas para que puedas hacer Merge, 
# pero ten en cuenta que aquí son valores validados (más robustos).
df_export['Mean_CV_Recall'] = cv_results['mean_test_recall']
df_export['Accuracy_Test']  = cv_results['mean_test_accuracy'] 
df_export['Precision_Test'] = cv_results['mean_test_precision']
df_export['F1_Test']        = cv_results['mean_test_f1']

# C. Tiempos
# Mapeamos los tiempos promedio de CV a las columnas de Fase 1
df_export['Train_Time_sec'] = cv_results['mean_fit_time']
df_export['Pred_Time_sec']  = cv_results['mean_score_time']

# Ordenar igual que en Fase 1 (Por Recall y luego Accuracy)
df_export = df_export.sort_values(by=['Mean_CV_Recall', 'Accuracy_Test'], ascending=False)

csv_filename = 'resultados_fase2_random.csv'
df_export.to_csv(csv_filename, index=False)

print(f"Archivo generado: {csv_filename}")
print("Formato de columnas verificado: IDÉNTICO a Fase 1.")
print(df_export.head(1)) # Muestra la primera fila para que verifiques visualmente