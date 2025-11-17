# PIA Algoritmia y Optimizacion
## Optimizacion de algoritmo KNearestNeighbors

## Integrantes
* Rodrigo Zacatenco Olive               ???????
* Roberto Sanchez Santoyo               ???????
* Luis Fernando Segobia Torres          ???????
* Aldo Adrian Davila Gonzales           1994122
* Esmeralda Gabriela Mendieta Gonzalez  2064574

## Porcedimiento

1. Entrenar n  modelos KNN diferentes variando sus parametros (k, w, p)
    - k : numero de vecinos mas cercanos [1, 2, 3, ...]
    - w : peso asignado en base a distancia o igual para todos ["uniform", "distance"]
    - p : tipo de medida de distancia {1: "euclidean", 2: "manhattan"}

2. Para cada uno de los n modelos entrenados evaluarlos en m medidas de desempeño
    - las medidas de desempeño se eligen a nuestro criterio
    m1 : "???"
    m2 : "???"
    ...

3. Guardar todas las medidas de desempeño en un diccionario gigante
   {(k,w,p): (m1, m2, ...)}. Tal vez convenga generar un CSV con los resultados
   para no hacer a el usuario correr los modelos

4. Hacer una applicacion para presentar los datos obtenidos en cada ejecucion
   (basicamente presentar el CSV) y una seccion dedicada a la combinacion de
   parameros con mayor desempeño (en cada una de las categorias o como promedio?

## Dependencias
* python
    - numpy
    - scikit-learn


