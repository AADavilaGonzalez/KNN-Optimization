
from itertools import product
import pandas as pd
from matplotlib import pyplot as plt

tests = ["recall", "accuracy", "precision", "f1"]
legend_loc = ["lower right", "lower left", "lower left", "lower left"]

weights = ["uniform", "distance"]
metrics = ["euclidean", "manhattan", "chebyshev"]

df = pd.read_csv("../resultados_fase1_completa.csv")
df = df.sort_values(by="n_neighbors", ascending=True)

for test, legend_loc in zip(tests, legend_loc):

    fig, ax = plt.subplots()

    for weight, metric in product(weights, metrics):
        
        label = f"{weight}, {metric}"

        view  = df[
            (df["weights"] == weight) &
            (df["metric"] == metric)
        ]

        if not isinstance(view, pd.DataFrame): raise TypeError()
        if view.empty: continue

        X = view["n_neighbors"].to_numpy()
        Y = view[test].to_numpy()
        ax.plot(X,Y,label=label, linestyle="-")
        
    ax.set_title(f"K vs {test.upper()}")
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel(f"{test.capitalize()} Score")

    ax.legend(
        loc=legend_loc,
        fontsize="small"
    )

    file_path = f"output/k_vs_{test}_perf.png"
    fig.savefig(file_path, dpi=300)

    plt.close(fig)


