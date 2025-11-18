import time
from typing import cast
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from datos import *
from config import *
from utils import exportar_resultados_a_csv

N_ITER = 225  # Presupuesto limitado

# Definición del Espacio de busqueda para Fase 2
param_random = {
    'n_neighbors': randint(1, 151),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

random_search = RandomizedSearchCV(
    estimator=knn_base,
    param_distributions=param_random,
    n_iter=N_ITER,
    cv=cv_config,
    scoring=scoring_metrics,
    refit=cast(bool,'recall'), # El ganador se decide por Recall
    n_jobs=-1,
    random_state=42,
    return_train_score=False
)

print("--- Fase 2: Búsqueda Aleatoria ---")
print(f"Iniciando búsqueda aleatoria ({N_ITER} iteraciones)...")

# 3. Ejecución
start_total_time = time.time()
random_search.fit(X_train, y_train)
end_total_time = time.time()

# best_model = random_search.best_estimator_
# y_pred = best_model.predict(X_test)

print(f"\n--- Resultados Fase 2 Terminados ---")
print(f"Tiempo Total: {end_total_time - start_total_time:.4f} s")
print(f"Mejor Recall (CV): {random_search.best_score_:.4f}")
print(f"Mejores Parametros: {random_search.best_params_}")

csv_filename = '../resultados_fase2_random.csv'
exportar_resultados_a_csv(random_search, csv_filename)

print(f"Archivo generado: {csv_filename}")
