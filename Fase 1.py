import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("--- Fase 1: Búsqueda Exhaustiva con Validación Cruzada ---")
X, y = load_breast_cancer(return_X_y=True)

# Split básico para medición de tiempos (Train 70% / Test 30%)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# stratifiy=y asegura que la proporción de clases se mantiene en ambos conjuntos

# scaler normaliza características para KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
X_full_scaled = scaler.fit_transform(X)

# --- Definición del Espacio de Búsqueda ---
k_values = range(1, 51)
weights_options = ['uniform', 'distance']  # uniform = votos iguales, distance = votos ponderados por distancia
metric_options = ['euclidean', 'manhattan',
                  'chebyshev']  # euclidean = distancia directa, manhattan = distancia en bloques, chebyshev = max diferencia en cualquier dimensión
cv=50
results_list = []
total_iters = len(k_values) * len(weights_options) * len(metric_options)
print(f"Total de modelos a entrenar y validar: {total_iters}")
print(f"Nota: Esto tomará un momento porque haremos {cv} validaciones por cada modelo...")
# se hacen 10 validaciones cruzadas por cada combinación de parámetros

start_total_time = time.time()
counter = 0

# ---  Ejecución ---
#  probando todas las combinaciones de parámetros
for k in k_values:
    for w in weights_options:
        for m in metric_options:
            counter += 1
            # Instanciar modelo
            p_val = 2 if m == 'euclidean' else 1 if m == 'manhattan' else 3  # p=2 para euclidean, p=1 para manhattan, p=3 para chebyshev
            knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m, p=p_val)
            # ---  VALIDACIÓN CRUZADA ---
            # Evaluamos el modelo 50 veces con diferentes partes de los datos
            # scoring='recall' prioriza minimizar falsos negativos
            cv_scores = cross_val_score(knn, X_full_scaled, y, cv=cv, scoring='recall')
            mean_cv_recall = cv_scores.mean()  # Promedio de recall en validación cruzada

            # --- Medición de TIEMPOS ---
            start_train = time.time()
            knn.fit(X_train, y_train)  # Entrenamiento en el conjunto de entrenamiento
            train_time = time.time() - start_train

            start_pred = time.time()
            y_pred = knn.predict(X_test)
            pred_time = time.time() - start_pred

            # ---  Métricas ---
            acc = accuracy_score(y_test, y_pred)  # Precisión en el conjunto de prueba
            prec = precision_score(y_test, y_pred)  # Calidad de positivos predichos
            f1 = f1_score(y_test, y_pred)  # Balance entre precisión y recall

            # Guardar
            results_list.append({
                'n_neighbors': k,
                'weights': w,
                'metric': m,
                'Mean_CV_Recall': mean_cv_recall,
                'Accuracy_Test': acc,
                'Precision_Test': prec,
                'F1_Test': f1,
                'Train_Time_sec': train_time,
                'Pred_Time_sec': pred_time
            })

            # Barra de progreso simple en consola
            if counter % 20 == 0:
                print(f"Progreso: {counter}/{total_iters} modelos evaluados...")

end_total_time = time.time()

# --- Procesamiento y Exportación ---
df_results = pd.DataFrame(results_list)

# Ordenar: Primero por el mejor Recall validado, luego por Accuracy
df_results = df_results.sort_values(by=['Mean_CV_Recall', 'Accuracy_Test'], ascending=False)

csv_filename = 'resultados_fase1_completa.csv'
df_results.to_csv(csv_filename, index=False)

print(f"\n--- Proceso Terminado en {end_total_time - start_total_time:.2f} segundos ---")
print(f"Mejor modelo (Top 1):")
print(df_results.iloc[0][['n_neighbors', 'weights', 'metric', 'Mean_CV_Recall']])
print(f"\nResultados guardados en: {csv_filename}")