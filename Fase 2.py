import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

print("--- Fase 2: Búsqueda Aleatoria ---")

# 1. Carga y Split
X, y = load_breast_cancer(return_X_y=True)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y #stratifícación para mantener proporciones
)
# 2. Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


param_distributions = {
    'n_neighbors': randint(1, 150),
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
n_iter_search = 225  # Presupuesto limitado

cv = 50
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=knn_base,
    param_distributions=param_distributions,
    n_iter=n_iter_search,
    cv=skf,
    scoring=scoring_metrics,
    refit='recall',     # El ganador se decide por Recall
    n_jobs=-1,
    random_state=42,
    return_train_score=False
)

print(f"Iniciando búsqueda aleatoria ({n_iter_search} iteraciones)...")

# 3. Ejecución
start_total_time = time.time()
random_search.fit(X_train, y_train)
end_total_time = time.time()

# 4. Evaluación del Ganador (Para mostrar en consola)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\n--- Resultados Fase 2 Terminados ---")
print(f"Tiempo Total: {end_total_time - start_total_time:.4f} s")
print(f"Mejor Recall (CV): {random_search.best_score_:.4f}")



cv_results = pd.DataFrame(random_search.cv_results_)

df_export = pd.DataFrame()

# Parámetros
df_export['n_neighbors'] = cv_results['param_n_neighbors']
df_export['weights'] = cv_results['param_weights']
df_export['metric'] = cv_results['param_metric']

# Métricas
df_export['Mean_CV_Recall'] = cv_results['mean_test_recall']
df_export['Accuracy_Test']  = cv_results['mean_test_accuracy']
df_export['Precision_Test'] = cv_results['mean_test_precision']
df_export['F1_Test']        = cv_results['mean_test_f1']

#  Tiempos
df_export['Train_Time_sec'] = cv_results['mean_fit_time']
df_export['Pred_Time_sec']  = cv_results['mean_score_time']

df_export = df_export.sort_values(by=['Mean_CV_Recall', 'Accuracy_Test'], ascending=False)

csv_filename = 'resultados_fase2_random.csv'
df_export.to_csv(csv_filename, index=False)

print(f"Archivo generado: {csv_filename}")
print(df_export.head(1))
