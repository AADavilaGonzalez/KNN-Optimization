import pandas as pd

df = pd.read_csv("../resultados_fase1_completa.csv")

filas_euclidean = df[ df["metric"] == "euclidean" ]
tiempos_entrenamiento_eucl = filas_euclidean["train_time"]
tiempos_prediccion_eucl = filas_euclidean["pred_time"]

filas_manhattan = df[ df["metric"] == "manhattan" ]
tiempos_entrenamiento_manh = filas_manhattan["train_time"]
tiempos_prediccion_manh = filas_manhattan["pred_time"]

filas_chebyshev = df[ df["metric"] == "chebyshev" ]
tiempos_entrenamiento_cheb = filas_chebyshev["train_time"]
tiempos_prediccion_cheb = filas_chebyshev["pred_time"]

print(
    "Tiempos de entrenamiento promedio por medida:",
    f"Euclidean: {tiempos_entrenamiento_eucl.mean()}",
    f"Manhattan: {tiempos_entrenamiento_manh.mean()}",
    f"Chebyshev: {tiempos_entrenamiento_cheb.mean()}",
    "",
    "Tiempos de prediccion promedio por medida:",
    f"Euclidean: {tiempos_prediccion_eucl.mean()}",
    f"Manhattan: {tiempos_prediccion_manh.mean()}",
    f"Chebyshev: {tiempos_prediccion_cheb.mean()}",
    sep = "\n"
)
