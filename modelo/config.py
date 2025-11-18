
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Semilla fija para tener reproducibilidad
SEED = 42

# Modelo utilizado en las pruebas (K nearest neighbors)
knn_base = KNeighborsClassifier()

# Métricas de Puntuación
scoring_metrics = {
    'recall': 'recall',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'f1': 'f1'
}

# Configracion de pruebas Cross Validation
cv_config = StratifiedKFold(n_splits=50, shuffle=True, random_state=SEED)

# Configuracion de parametros exportados a csv
nombre_parametros_exportados= {
    "param_n_neighbors"     : "n_neighbors",
    "param_weights"         : "weigths",
    "param_metric"          : "metric",
    "mean_test_recall"      : "recall",
    "mean_test_accuracy"    : "accuracy",
    "mean_test_precision"   : "precision",
    "mean_test_f1"          : "f1",
    "mean_fit_time"         : "train_time", # segundos
    "mean_score_time"       : "pred_time"
}

orden_parametros_exportados = [
    "mean_test_recall",
    "mean_test_accuracy"
]
