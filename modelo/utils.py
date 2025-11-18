
import pandas as pd

from config import (
    nombre_parametros_exportados,
    orden_parametros_exportados
)

from typing import Any

def exportar_resultados_a_csv(
    busqueda: Any,
    nombre_archivo: str,
    nombres_parametros: dict[str, str] = nombre_parametros_exportados,
    orden_parametros: list[str] = orden_parametros_exportados
) -> None:

    # Dataframe de los resultados internos de la busqueda
    df_results = pd.DataFrame(busqueda.cv_results_)
    # Dataframe al que acoplar los datos en base al formato deseado
    df_export = pd.DataFrame()

    for parametro, nombre in nombres_parametros.items():
        df_export[nombre] = df_results[parametro]

    if orden_parametros:
        df_export.sort_values(
            by=[nombres_parametros[param] for param in orden_parametros],
            ascending=False,
            inplace=True
        )
        
    df_export.to_csv(nombre_archivo, index = False)

