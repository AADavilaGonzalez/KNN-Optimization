import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint

print("--- Fase 2: Búsqueda Aleatoria ---")

# 1. Carga y Split
X, y = load_breast_cancer(return_X_y=True)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 3. Configuración
param_distributions = {
    'n_neighbors': randint(1, 51),           
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev'] 
}

scoring_metrics = {
    'recall': 'recall',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'f1': 'f1'
}

knn_base = KNeighborsClassifier()
n_iter_search = 50  

# Configuramos la búsqueda mediante RandomizedSearchCV para optimizar en paralelo y no hacer una búsqueda exhaustiva
random_search = RandomizedSearchCV(
    estimator=knn_base,
    param_distributions=param_distributions,
    n_iter=n_iter_search,
    cv=10,
    scoring=scoring_metrics,
    refit='recall',
    n_jobs=-1,
    random_state=42,
    return_train_score=False
)

print(f"Total de modelos a entrenar y validar: {n_iter_search}")
print("Nota: Esto tomará un momento (optimizando en paralelo)...")

start_total_time = time.time()

random_search.fit(X_train, y_train)

end_total_time = time.time()

# Extraemos resultados
cv_results = pd.DataFrame(random_search.cv_results_)
df_export = pd.DataFrame()

# A. Parámetros
df_export['n_neighbors'] = cv_results['param_n_neighbors']
df_export['weights'] = cv_results['param_weights']
df_export['metric'] = cv_results['param_metric']

# B. Métricas 
df_export['Mean_CV_Recall'] = cv_results['mean_test_recall']
df_export['Accuracy_Test']  = cv_results['mean_test_accuracy'] 
df_export['Precision_Test'] = cv_results['mean_test_precision']
df_export['F1_Test']        = cv_results['mean_test_f1']

# C. Tiempos
df_export['Train_Time_sec'] = cv_results['mean_fit_time']
df_export['Pred_Time_sec']  = cv_results['mean_score_time']

df_export = df_export.sort_values(by=['Mean_CV_Recall', 'Accuracy_Test'], ascending=False)

csv_filename = 'resultados_fase2_random.csv'
df_export.to_csv(csv_filename, index=False)

print(f"\n--- Proceso Terminado en {end_total_time - start_total_time:.2f} segundos ---")
print(f"Mejor modelo (Top 1):")
# Imprimimos exactamente las mismas columnas clave que en Fase 1
print(df_export.iloc[0][['n_neighbors', 'weights', 'metric', 'Mean_CV_Recall']])
print(f"\nResultados guardados en: {csv_filename}")