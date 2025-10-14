import pandas as pd
from pandas import DataFrame, Series

from mi_arbol_decision.entropia import calcular_entropia
from mi_arbol_decision.nodo import Nodo

# TODO 1: implementar metodo para predecir la clase de una instancia dada
# TODO 2: flexibilizar el algoritmo para poder usar diferentes funciones de impureza

class ArbolDecision:
    def __init__(self, umbral_ganancia: float = 0.001):
        self.raiz_arbol: Nodo | None = None
        self.df: DataFrame | None = None
        self.nombre_objetivo: str = ''
        self.umbral_ganancia = umbral_ganancia

    def entrenar(self, df: DataFrame, nombre_objetivo: str) -> None:
        print("----- FASE DE ENTRENAMIENTO -----")
        self.df: DataFrame = df
        self.nombre_objetivo: str = nombre_objetivo
        lista_atributos = list(df.drop(nombre_objetivo, axis=1).columns)
        self.raiz_arbol = self._construir_arbol(df, lista_atributos)

    def _construir_arbol(self, df: DataFrame, atributos_disponibles: list[str]) -> Nodo:
        columna_clases: Series = df[self.nombre_objetivo]
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
            entropia_conjunto: float = calcular_entropia(columna_clases)
            print("\nEntropia del conjunto de datos (p0): " + str(entropia_conjunto))

            mejor_atributo, ganancia_mejor_atributo, umbral_mejor_atributo = (
                self._encontrar_mejor_atributo(df, atributos_disponibles, entropia_conjunto)
            )

            if ganancia_mejor_atributo < self.umbral_ganancia:
                print(
                    "El atributo '" + mejor_atributo + "' no reduce significativamente la impureza " + str(
                        entropia_conjunto))
                print("Se crea un nodo hoja con clase: " + clase_mas_comun)
                return Nodo(valor=clase_mas_comun, clase_mas_comun=clase_mas_comun)

            # Crear un nodo de decisión y construir los hijos
            nuevos_atributos_disponibles: list[str] = atributos_disponibles.copy()
            es_atributo_categorico: bool = umbral_mejor_atributo is None
            nodos_hijos: dict[str, Nodo] = {}

            print(f"Mejor atributo para dividir: '{mejor_atributo}' (Ganancia={ganancia_mejor_atributo:.4f})")
            if es_atributo_categorico:
                # Los atributos categóricos se pueden usar una sola vez en una rama
                # Los atributos continuos pueden volver a usarse en sub-ramas con otros umbrales
                nuevos_atributos_disponibles.remove(mejor_atributo)

                print("Valores posibles del atributo: " + str(df[mejor_atributo].unique()))
                # Se particiona el dataframe por cada valor único del mejor atributo seleccionado
                for valor_atributo, df_valor_atributo in df.groupby(mejor_atributo):
                    print(f"\n-- Subconjunto del dataframe para valor: {valor_atributo} --")
                    print(df_valor_atributo)
                    if len(df_valor_atributo) == 0:
                        continue

                    nodos_hijos[str(valor_atributo)] = self._construir_arbol(df_valor_atributo,
                                                                             nuevos_atributos_disponibles)

                return Nodo(atributo=mejor_atributo, nodos_hijos=nodos_hijos, clase_mas_comun=clase_mas_comun)
            else:
                print(f"Umbral de división: {umbral_mejor_atributo}")
                # Dividir el conjunto en dos ramas: <= umbral y > umbral
                df_menor_igual = df[df[mejor_atributo] <= umbral_mejor_atributo]
                df_mayor = df[df[mejor_atributo] > umbral_mejor_atributo]

                print(f"\n-- Subconjunto del dataframe para <= {umbral_mejor_atributo}: --")
                print(df_menor_igual)
                print(f"\n-- Subconjunto del dataframe para < {umbral_mejor_atributo}: --")
                print(df_mayor)

                nodos_hijos['<= ' + str(umbral_mejor_atributo)] = self._construir_arbol(df_menor_igual,
                                                                                        nuevos_atributos_disponibles)
                nodos_hijos['> ' + str(umbral_mejor_atributo)] = self._construir_arbol(df_mayor,
                                                                                       nuevos_atributos_disponibles)

                return Nodo(atributo=mejor_atributo, nodos_hijos=nodos_hijos, clase_mas_comun=clase_mas_comun,
                            umbral=umbral_mejor_atributo)

    def _tiene_una_sola_clase(self, columna_clases: Series) -> bool:
        return columna_clases.nunique() == 1

    def _hay_atributos_disponibles(self, atributos_disponibles: list[str]) -> bool:
        return len(atributos_disponibles) > 0

    #  Itera sobre los atributos y encuentra el que tiene la mayor ganancia.
    #  Maneja tanto atributos continuos como categóricos.
    def _encontrar_mejor_atributo(self, df: DataFrame, atributos_disponibles: list[str], entropia_conjunto: float) -> \
            tuple[str, float, float | None]:
        mejor_atributo_nombre: str = ''
        mejor_ganancia: float = -1
        mejor_umbral_atributo: float | None = None

        for atributo in atributos_disponibles:
            if self._es_atributo_continuo(df[atributo]):
                ganancia, umbral = self._encontrar_mejor_umbral_continuo(df, atributo, entropia_conjunto)
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_atributo_nombre = atributo
                    mejor_umbral_atributo = umbral
            else:
                entropia_atributo = self._calcular_entropia_atributo_categorico(df, atributo)
                ganancia = entropia_conjunto - entropia_atributo
                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_atributo_nombre = atributo
                    # El atributo es categórico, por lo que no tiene umbral
                    mejor_umbral_atributo = None

        return mejor_atributo_nombre, mejor_ganancia, mejor_umbral_atributo

    def _es_atributo_continuo(self, serie_atributo: Series) -> bool:
        return pd.api.types.is_numeric_dtype(serie_atributo)

    # Calcula la mejor división binaria para un atributo continuo.
    def _encontrar_mejor_umbral_continuo(self, df: DataFrame, atributo: str, entropia_conjunto: float) -> tuple[
        float, float]:
        mejor_ganancia: float = -1
        mejor_umbral: float | None = None

        valores_unicos = sorted(df[atributo].unique())
        # Generar puntos de corte candidatos
        puntos_corte: list[float] = [(valores_unicos[i] + valores_unicos[i + 1]) / 2 for i in
                                     range(len(valores_unicos) - 1)]

        for umbral in puntos_corte:
            # Dividir el DataFrame en 2 usando el umbral
            df_menor_igual = df[df[atributo] <= umbral]
            df_mayor = df[df[atributo] > umbral]

            # Calcular la entropía ponderada de la división
            prob_menor_igual: float = len(df_menor_igual) / len(df)
            prob_mayor: float = len(df_mayor) / len(df)

            entropia_atributo: float = (prob_menor_igual * calcular_entropia(df_menor_igual[self.nombre_objetivo]) +
                                        prob_mayor * calcular_entropia(df_mayor[self.nombre_objetivo]))

            ganancia_actual: float = entropia_conjunto - entropia_atributo
            if ganancia_actual > mejor_ganancia:
                mejor_ganancia = ganancia_actual
                mejor_umbral = umbral

        return mejor_ganancia, mejor_umbral

    def _calcular_entropia_atributo_categorico(self, df: DataFrame, nombre_atributo: str) -> float:
        columna_clases = df[self.nombre_objetivo]
        valores_unicos_atributo = df[nombre_atributo].unique()

        entropia_atributo: float = 0
        for valor in valores_unicos_atributo:
            subconjunto_clases = df[df[nombre_atributo] == valor][self.nombre_objetivo]
            probabilidad_valor_atributo = len(subconjunto_clases) / len(columna_clases)
            entropia_atributo += probabilidad_valor_atributo * calcular_entropia(subconjunto_clases)

        return entropia_atributo
