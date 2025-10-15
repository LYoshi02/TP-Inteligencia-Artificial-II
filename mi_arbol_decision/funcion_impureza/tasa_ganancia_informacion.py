from pandas import DataFrame

from mi_arbol_decision.funcion_impureza.atributo import Atributo
from mi_arbol_decision.funcion_impureza.entropia import calcular_entropia
from mi_arbol_decision.funcion_impureza.funcion import FuncionImpureza
from mi_arbol_decision.funcion_impureza.ganancia_informacion import GananciaDeInformacion


class TasaGananciaDeInformacion(FuncionImpureza):
    def __init__(self, nombre_objetivo: str = ''):
        self.nombre_objetivo: str = nombre_objetivo
        pass

    def encontrar_mejor_atributo(self, df: DataFrame, atributos_disponibles: list[str]) -> Atributo:
        ganancia_informacion = GananciaDeInformacion(self.nombre_objetivo)
        mejor_atributo: Atributo = Atributo()
        mejor_tasa_ganancia = -1

        for nombre_atributo in atributos_disponibles:
            atributo = ganancia_informacion.calcular_ganancia_atributo(df, nombre_atributo)
            entropia_atributo: float = calcular_entropia(df[atributo.nombre])
            if entropia_atributo != 0:
                tasa_ganancia = atributo.ganancia / entropia_atributo
            else:
                tasa_ganancia = 0

            print("ganancia (*): " + str(atributo.ganancia))
            print("entropia_atributo (*): " + str(entropia_atributo))
            print("tasa_ganancia (*): " + str(tasa_ganancia))
            if tasa_ganancia > mejor_tasa_ganancia:
                mejor_atributo = atributo
                mejor_tasa_ganancia = tasa_ganancia

        return mejor_atributo
