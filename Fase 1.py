import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

print("--- Fase 1: Búsqueda Exhaustiva con GridSearchCV (Alineación Final) ---")
X, y = load_breast_cancer(return_X_y=True)

# 1. Split y Escalado (Igual que Fase 2)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 2. Definición del Espacio de Búsqueda (Grid)
param_grid = {
    'n_neighbors': list(range(1, 150)), # Lista en lugar de randint
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

# 3. Métricas de Puntuación (Igual que Fase 2)
scoring_metrics = {
    'recall': 'recall',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'f1': 'f1'
}

knn_base = KNeighborsClassifier()
cv = 50
# 4. Generador de CV Fijo (Igual que Fase 2)
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

# --- Ejecución de GridSearchCV ---
grid_search = GridSearchCV(
    estimator=knn_base,
    param_grid=param_grid,
    cv=skf,                 # Se usa el CV fijo
    scoring=scoring_metrics,
    refit='recall',         # Se optimiza por Recall
    n_jobs=-1,              # Mantener -1 para comparar eficiencia (o usar 1 para mayor certeza)
    return_train_score=False
)

print(f"Iniciando búsqueda exhaustiva ({len(grid_search.param_grid.keys())} combinaciones)...")

start_total_time = time.time()
grid_search.fit(X_train, y_train)
end_total_time = time.time()

# --- Procesamiento y Exportación (Igual que Fase 2) ---
print(f"\n--- Resultados Fase 1 Terminados ---")
print(f"Tiempo Total: {end_total_time - start_total_time:.4f} s")
print(f"Mejor Recall (CV): {grid_search.best_score_:.4f}")

cv_results = pd.DataFrame(grid_search.cv_results_)

df_export = pd.DataFrame()
# Parámetros
df_export['n_neighbors'] = cv_results['param_n_neighbors']
df_export['weights'] = cv_results['param_weights']
df_export['metric'] = cv_results['param_metric']

# Métricas (Ahora son los promedios de CV, igual que la Fase 2)
df_export['Mean_CV_Recall'] = cv_results['mean_test_recall']
df_export['Accuracy_Test'] = cv_results['mean_test_accuracy']
df_export['Precision_Test'] = cv_results['mean_test_precision']
df_export['F1_Test'] = cv_results['mean_test_f1']

# Tiempos (Ahora son los promedios de CV, igual que la Fase 2)
df_export['Train_Time_sec'] = cv_results['mean_fit_time']
df_export['Pred_Time_sec'] = cv_results['mean_score_time']

df_export = df_export.sort_values(by=['Mean_CV_Recall', 'Accuracy_Test'], ascending=False)

csv_filename = 'resultados_fase1_completa.csv'
df_export.to_csv(csv_filename, index=False)

print(f"Archivo generado: {csv_filename}")
print(df_export.head(1))
