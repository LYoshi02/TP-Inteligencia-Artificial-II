import numpy as np
from pandas import DataFrame, Series

class ArbolDecision:
    def __init__(self, umbral_ganancia: float = 0.001):
        self.raiz_arbol: Nodo | None = None
        self.df: DataFrame | None = None
        self.nombre_objetivo: str = ''
        self.umbral_ganancia = umbral_ganancia

    def entrenar(self, df: DataFrame, nombre_objetivo: str):
        print("----- FASE DE ENTRENAMIENTO -----")
        self.df: DataFrame = df
        self.nombre_objetivo: str = nombre_objetivo
        lista_atributos = list(df.drop(nombre_objetivo, axis=1).columns)
        self.raiz_arbol = self._construir_arbol(df, lista_atributos)

    def _construir_arbol(self, df: DataFrame, atributos_disponibles: list[str]):
        columna_clases = df[self.nombre_objetivo]
        clase_mas_comun = columna_clases.mode()[0] if not columna_clases.empty else None

        print("\n--- DataFrame ---")
        print(df)

        if self._tiene_una_sola_clase(columna_clases):
            print("Criterio de parada 1: hay una sola clase en este conjunto")
            print("Se crea un nodo hoja con clase: " + clase_mas_comun)
            return Nodo(valor=clase_mas_comun, clase_mas_comun=clase_mas_comun)
        elif not self._hay_atributos_disponibles(atributos_disponibles):
            print("Criterio de parada 2: no hay más atributos disponibles para esta rama")
            print("Se crea un nodo hoja con clase: " + clase_mas_comun)
            return Nodo(valor=clase_mas_comun, clase_mas_comun=clase_mas_comun)
        else:
            print("Se expande el árbol")
            # Entropia del conjunto de datos (p0)
            entropia_conjunto = calcular_entropia(columna_clases)
            print("\nEntropia del conjunto de datos (p0): " + str(entropia_conjunto))

            entropia_atributos: dict[str, float] = dict()
            for atributo in atributos_disponibles:
                entropia_atributos[atributo] = self._calcular_entropia_atributo(df, atributo)
            print("\nEntropia de cada atributo (pi): " + str(entropia_atributos))

            # Nombre del mejor atributo (Ag)
            mejor_atributo = self._elegir_mejor_atributo(entropia_conjunto, entropia_atributos)
            # Entropia del mejor atributo (pg)
            entropia_mejor_atributo = entropia_atributos[mejor_atributo]
            ganancia_mejor_atributo = entropia_conjunto - entropia_mejor_atributo

            print("Mejor atributo (ganancia=" + str(ganancia_mejor_atributo) + "): " + mejor_atributo)

            if ganancia_mejor_atributo < self.umbral_ganancia:
                print("El atributo '" + mejor_atributo + "' no reduce significativamente la impureza " + entropia_conjunto)
                print("Se crea un nodo hoja con clase: " + clase_mas_comun)
                return Nodo(valor=clase_mas_comun, clase_mas_comun=clase_mas_comun)

            # Crear un nodo de decisión y construir los hijos
            print("Se crea un nuevo nodo de decision con el atributo: " + mejor_atributo)
            nodos_hijos = {}
            print("Valores posibles del atributo: ")
            print(df[mejor_atributo].unique())
            # Se particiona el dataframe por cada valor único del mejor atributo seleccionado
            for valor_atributo, df_valor_atributo in df.groupby(mejor_atributo):
                print(f"\n-- Subconjunto del dataframe para valor: {valor_atributo} --")
                print(df_valor_atributo)
                if len(df_valor_atributo) == 0:
                    continue

                nuevos_atributos_disponibles = atributos_disponibles.copy()
                nuevos_atributos_disponibles.remove(mejor_atributo)
                print("Atributos disponibles: ", atributos_disponibles)
                print("Nuevos atributos disponibles: ", nuevos_atributos_disponibles)
                nodos_hijos[valor_atributo] = self._construir_arbol(df_valor_atributo, nuevos_atributos_disponibles)

            return Nodo(atributo=mejor_atributo, nodos_hijos=nodos_hijos, clase_mas_comun=clase_mas_comun)

    def _tiene_una_sola_clase(self, columna_clases: Series):
        return columna_clases.nunique() == 1

    def _hay_atributos_disponibles(self, atributos_disponibles: list[str]):
        return len(atributos_disponibles) > 0

    def _calcular_entropia_atributo(self, df: DataFrame, nombre_atributo: str):
        columna_clases = df[self.nombre_objetivo]
        valores_unicos_atributo = df[nombre_atributo].unique()

        entropia_atributo = 0
        for valor in valores_unicos_atributo:
            subconjunto_clases = df[df[nombre_atributo] == valor][self.nombre_objetivo]
            probabilidad_valor_atributo = len(subconjunto_clases) / len(columna_clases)
            entropia_atributo += probabilidad_valor_atributo * calcular_entropia(subconjunto_clases)

        return entropia_atributo

    def _elegir_mejor_atributo(self, entropia_conjunto: float, entropia_atributos: dict[str, float]):
        mayor_ganancia: float = -1
        mejor_atributo: str = ''

        for atributo, entropia_atributo in entropia_atributos.items():
            ganancia_atributo = entropia_conjunto - entropia_atributo
            if ganancia_atributo > mayor_ganancia:
                mayor_ganancia = ganancia_atributo
                mejor_atributo = atributo

        return mejor_atributo


class Nodo:
    def __init__(self, atributo=None, nodos_hijos=None, valor=None, clase_mas_comun=None):
        # Nombre del atributo para la decisión
        self.atributo: str = atributo
        # Diccionario para guardar las ramas hacia nodos de menor nivel en el árbol
        self.nodos_hijos: dict = nodos_hijos if nodos_hijos is not None else {}
        # Se setea un valor cuando el nodo es un nodo hoja
        self.valor = valor
        # Clase más común en este nodo
        self.clase_mas_comun = clase_mas_comun

    def es_nodo_hoja(self):
        return self.valor is not None


def calcular_entropia(s: Series):
    # Contar cantidad de instancias para cada clase
    cant_instancias_por_clase = s.value_counts()
    # Cantidad total de instancias
    cant_instancias_total = len(s)
    # Probabilidad de cada una de las clases
    probabilidades_por_clase = cant_instancias_por_clase / cant_instancias_total
    # Calculo de entropía como sumatoria de probabilidades por log2 de la probabilidad
    entropia = np.sum([p * np.log2(p) for p in probabilidades_por_clase if p > 0])
    if entropia != 0:
        entropia *= -1

    print("\nSerie:")
    print(s)
    print("Entropia: ", entropia)

    return entropia
