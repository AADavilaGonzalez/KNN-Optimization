import time
from math import prod
from typing import cast
from sklearn.model_selection import GridSearchCV

from config import *
from datos import *
from utils import exportar_resultados_a_csv

# Definición del Espacio de búsqueda para Fase 1
param_grid = {
    'n_neighbors': list(range(1, 151)), #Espacio de busqueda lineal [1,150]
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

# Configuraion del espacio de busqueda exhaustivo (por cuadricula)
grid_search = GridSearchCV(
    estimator=knn_base,
    param_grid=param_grid,
    cv=cv_config,
    scoring=scoring_metrics,
    refit=cast(bool,'recall'),         # Se optimiza por Recall
    n_jobs=-1,              # Activar uso de ejecucion en paralelo
    return_train_score=False
)

print("--- Fase 1: Búsqueda Exhaustiva ---")
combinaciones = prod(len(i) for i in param_grid.values())
print(f"Iniciando búsqueda exhaustiva ({combinaciones} combinaciones)...")

# Entrenar los modelos en una sola llamada a .fit()
start_total_time = time.time()
grid_search.fit(X_train, y_train)
end_total_time = time.time()

print(f"\n--- Resultados Fase 1 Terminados ---")
print(f"Tiempo Elapsado: {end_total_time - start_total_time:.4f} s")
print(f"Mejor Recall (CV): {grid_search.best_score_:.4f}")
print(f"Mejores Parametros: {grid_search.best_params_}")

csv_filename = '../resultados_fase1_completa.csv'
exportar_resultados_a_csv(grid_search, csv_filename)

print(f"Archivo generado: {csv_filename}")
