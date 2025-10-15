from abc import ABC, abstractmethod

from pandas import DataFrame

from mi_arbol_decision.funcion_impureza.atributo import Atributo


class FuncionImpureza(ABC):
    @abstractmethod
    def encontrar_mejor_atributo(self, df: DataFrame, atributos_disponibles: list[str]) -> Atributo:
        pass


class FUNCIONES_IMPUREZA:
    ganancia_informacion = 'ganancia_de_informacion'
    tasa_ganancia_informacion = 'tasa_de_ganancia_de_informacion'
