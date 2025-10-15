import pandas as pd
from pandas import DataFrame, Series

from mi_arbol_decision.funcion_impureza.atributo import Atributo
from mi_arbol_decision.funcion_impureza.entropia import calcular_entropia
from mi_arbol_decision.funcion_impureza.funcion import FuncionImpureza


class GananciaDeInformacion(FuncionImpureza):
    def __init__(self, nombre_objetivo: str = ''):
        self.nombre_objetivo: str = nombre_objetivo
        pass

    #  Itera sobre los atributos y encuentra el que tiene la mayor ganancia.
    #  Maneja tanto atributos continuos como categóricos.
    def encontrar_mejor_atributo(self, df: DataFrame, atributos_disponibles: list[str]) -> Atributo:
        mejor_atributo: Atributo = Atributo()

        for nombre_atributo in atributos_disponibles:
            atributo = self.calcular_ganancia_atributo(df, nombre_atributo)
            if atributo.ganancia > mejor_atributo.ganancia:
                mejor_atributo = atributo

        return mejor_atributo

    def calcular_ganancia_atributo(self, df: DataFrame, nombre_atributo: str) -> Atributo:
        # Entropia del conjunto de datos (p0)
        columna_clases: Series = df[self.nombre_objetivo]
        entropia_conjunto: float = calcular_entropia(columna_clases)

        atributo = Atributo(nombre=nombre_atributo)
        if self._es_atributo_continuo(df[nombre_atributo]):
            ganancia, umbral = self._encontrar_mejor_umbral_continuo(df, nombre_atributo, entropia_conjunto)
            atributo.ganancia = ganancia
            atributo.umbral = umbral
        else:
            ganancia = self._calcular_ganancia_atributo_categorico(df, nombre_atributo, entropia_conjunto)
            atributo.ganancia = ganancia
            atributo.umbral = None

        return atributo

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

            entropia_atributo: float = (
                    prob_menor_igual * calcular_entropia(df_menor_igual[self.nombre_objetivo]) +
                    prob_mayor * calcular_entropia(df_mayor[self.nombre_objetivo]))

            ganancia_actual: float = entropia_conjunto - entropia_atributo
            if ganancia_actual > mejor_ganancia:
                mejor_ganancia = ganancia_actual
                mejor_umbral = umbral

        return mejor_ganancia, mejor_umbral

    def _calcular_ganancia_atributo_categorico(self, df: DataFrame, nombre_atributo: str,
                                               entropia_conjunto: float) -> float:
        columna_clases = df[self.nombre_objetivo]
        valores_unicos_atributo = df[nombre_atributo].unique()

        entropia_atributo: float = 0
        for valor in valores_unicos_atributo:
            subconjunto_clases = df[df[nombre_atributo] == valor][self.nombre_objetivo]
            probabilidad_valor_atributo = len(subconjunto_clases) / len(columna_clases)
            entropia_atributo += probabilidad_valor_atributo * calcular_entropia(subconjunto_clases)

        ganancia: float = entropia_conjunto - entropia_atributo
        return ganancia
