import numpy as np
from pandas import Series

def calcular_entropia(s: Series) -> float:
    if s.empty:
        return 0

    # Contar cantidad de instancias para cada clase
    cant_instancias_por_clase = s.value_counts()
    # Cantidad total de instancias
    cant_instancias_total = len(s)
    if cant_instancias_total == 0:
        return 0
    # Probabilidad de cada una de las clases
    probabilidades_por_clase = cant_instancias_por_clase / cant_instancias_total
    # Calculo de entropía como sumatoria de probabilidades por log2 de la probabilidad
    entropia = np.sum([p * np.log2(p) for p in probabilidades_por_clase if p > 0])
    if entropia != 0:
        entropia *= -1

    print("\nSerie:")
    print(s)
    print("Entropia: ", entropia)

    return float(entropia)