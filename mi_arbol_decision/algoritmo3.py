from pandas import DataFrame, Series

from mi_arbol_decision.funcion_impureza.funcion import FUNCIONES_IMPUREZA, FuncionImpureza
from mi_arbol_decision.funcion_impureza.ganancia_informacion import GananciaDeInformacion
from mi_arbol_decision.funcion_impureza.tasa_ganancia_informacion import TasaGananciaDeInformacion
from mi_arbol_decision.nodo import Nodo


class ArbolDecision:
    def __init__(self, umbral_ganancia: float = 0.001, funcion_impureza: str = ''):
        self.raiz_arbol: Nodo | None = None
        self.df: DataFrame | None = None
        self.nombre_objetivo: str = ''
        self.umbral_ganancia: float = umbral_ganancia
        self.nombre_funcion_impureza: str = funcion_impureza
        self.funcion_impureza: FuncionImpureza | None = None

    def entrenar(self, df: DataFrame, nombre_objetivo: str) -> None:
        print("----- FASE DE ENTRENAMIENTO -----")
        self.df: DataFrame = df
        self.nombre_objetivo: str = nombre_objetivo
        self.funcion_impureza = self._obtener_funcion_impureza(nombre_objetivo)
        self.raiz_arbol = self._construir_arbol(df, self._obtener_lista_atributos())

    def _obtener_funcion_impureza(self, nombre_objetivo: str) -> FuncionImpureza:
        if self.nombre_funcion_impureza == FUNCIONES_IMPUREZA.ganancia_informacion:
            return GananciaDeInformacion(nombre_objetivo)
        elif self.nombre_funcion_impureza == FUNCIONES_IMPUREZA.tasa_ganancia_informacion:
            return TasaGananciaDeInformacion(nombre_objetivo)
        else:  # Uso ganancia de informacion por defecto
            return GananciaDeInformacion(nombre_objetivo)

    def _obtener_lista_atributos(self) -> list[str]:
        if self.df is None or self.nombre_objetivo.strip() == '':
            return list()
        return list(self.df.drop(self.nombre_objetivo, axis=1).columns)

    def _construir_arbol(self, df: DataFrame, atributos_disponibles: list[str]) -> Nodo:
        columna_clases: Series = df[self.nombre_objetivo]
        clase_mas_comun = columna_clases.mode()[0] if not columna_clases.empty else None

        print("\n--- DataFrame ---")
        print(df)

        if self._tiene_una_sola_clase(columna_clases):
            print("Criterio de parada 1: hay una sola clase en este conjunto")
            print("Se crea un nodo hoja con clase: " + str(clase_mas_comun))
            return Nodo(valor=clase_mas_comun, clase_mas_comun=clase_mas_comun)
        elif not self._hay_atributos_disponibles(atributos_disponibles):
            print("Criterio de parada 2: no hay más atributos disponibles para esta rama")
            print("Se crea un nodo hoja con clase: " + str(clase_mas_comun))
            return Nodo(valor=clase_mas_comun, clase_mas_comun=clase_mas_comun)
        else:
            print("Se expande el árbol")
            mejor_atributo = self.funcion_impureza.encontrar_mejor_atributo(df, atributos_disponibles)

            if mejor_atributo.ganancia < self.umbral_ganancia:
                print(
                    "El atributo '" + mejor_atributo.nombre + "' no reduce significativamente la impureza")
                print("Se crea un nodo hoja con clase: " + str(clase_mas_comun))
                return Nodo(valor=clase_mas_comun, clase_mas_comun=clase_mas_comun)

            # Crear un nodo de decisión y construir los hijos
            nuevos_atributos_disponibles: list[str] = atributos_disponibles.copy()
            nodos_hijos: dict[str, Nodo] = {}

            print(f"Mejor atributo para dividir: '{mejor_atributo.nombre}' (Ganancia={mejor_atributo.ganancia:.4f})")
            if mejor_atributo.es_categorico():
                # Los atributos categóricos se pueden usar una sola vez en una rama
                # Los atributos continuos pueden volver a usarse en sub-ramas con otros umbrales
                nuevos_atributos_disponibles.remove(mejor_atributo.nombre)

                print("Valores posibles del atributo: " + str(df[mejor_atributo.nombre].unique()))
                # Se particiona el dataframe por cada valor único del mejor atributo seleccionado
                for valor_atributo, df_valor_atributo in df.groupby(mejor_atributo.nombre):
                    print(f"\n-- Subconjunto del dataframe para valor: {valor_atributo} --")
                    print(df_valor_atributo)
                    if len(df_valor_atributo) == 0:
                        continue

                    nodos_hijos[str(valor_atributo)] = self._construir_arbol(df_valor_atributo,
                                                                             nuevos_atributos_disponibles)

                return Nodo(atributo=mejor_atributo.nombre, nodos_hijos=nodos_hijos, clase_mas_comun=clase_mas_comun)
            else:
                print(f"Umbral de división: {mejor_atributo.umbral}")
                # Dividir el conjunto en dos ramas: <= umbral y > umbral
                df_menor_igual = df[df[mejor_atributo.nombre] <= mejor_atributo.umbral]
                df_mayor = df[df[mejor_atributo.nombre] > mejor_atributo.umbral]

                print(f"\n-- Subconjunto del dataframe para <= {mejor_atributo.umbral}: --")
                print(df_menor_igual)
                print(f"\n-- Subconjunto del dataframe para < {mejor_atributo.umbral}: --")
                print(df_mayor)

                nodos_hijos['<= ' + str(mejor_atributo.umbral)] = self._construir_arbol(df_menor_igual,
                                                                                        nuevos_atributos_disponibles)
                nodos_hijos['> ' + str(mejor_atributo.umbral)] = self._construir_arbol(df_mayor,
                                                                                       nuevos_atributos_disponibles)

                return Nodo(atributo=mejor_atributo.nombre, nodos_hijos=nodos_hijos, clase_mas_comun=clase_mas_comun,
                            umbral=mejor_atributo.umbral)

    def _tiene_una_sola_clase(self, columna_clases: Series) -> bool:
        return columna_clases.nunique() == 1

    def _hay_atributos_disponibles(self, atributos_disponibles: list[str]) -> bool:
        return len(atributos_disponibles) > 0

    # Predice la clase para una única instancia de datos.
    def predecir(self, instancia: dict):
        # Validaciones previas
        lista_atributos = self._obtener_lista_atributos()
        if self.raiz_arbol is None:
            raise RuntimeError("El árbol de decisión debe ser entrenado antes de poder predecir.")
        elif len(instancia.keys()) != len(lista_atributos):
            raise RuntimeError("Los atributos de la instancia no coinciden con los del árbol.")
        for atributo in instancia.keys():
            if atributo not in lista_atributos:
                raise RuntimeError("El atributo '" + atributo + "' no es válido")

        # Prediccion
        nodo_actual = self.raiz_arbol
        while not nodo_actual.es_nodo_hoja():
            # Obtener el atributo y el valor de la instancia para el nodo actual
            atributo_decision = nodo_actual.atributo
            valor_instancia = instancia[atributo_decision]

            # Lógica para decidir la siguiente rama
            es_atributo_categorico = nodo_actual.umbral is None
            if es_atributo_categorico:
                # Si el valor de la instancia existe como una rama en el árbol
                if valor_instancia in nodo_actual.nodos_hijos:
                    nodo_actual = nodo_actual.nodos_hijos[valor_instancia]
                else:
                    # Caso de respaldo:
                    # Si el valor no se vio durante el entrenamiento en esta rama, predice la clase más común de este nodo.
                    return nodo_actual.clase_mas_comun
            else:
                if valor_instancia <= nodo_actual.umbral:
                    # La clave para la rama "menor o igual"
                    llave_hijo = f"<= {nodo_actual.umbral}"
                else:
                    # La clave para la rama "mayor que"
                    llave_hijo = f"> {nodo_actual.umbral}"

                nodo_actual = nodo_actual.nodos_hijos[llave_hijo]

        # Cuando se llega a un nodo hoja, se retorna su valor como predicción
        return nodo_actual.valor

    ## Representación visual del árbol generado
    def imprimir_arbol(self, indent: str = "") -> None:
        if self.raiz_arbol is None:
            print("Aun no se generó el árbol")
            return

        self._imprimir_nodo(self.raiz_arbol, indent)

    # Recorre de manera recursiva el nodo recibido e imprime en consola el árbol generado
    def _imprimir_nodo(self, nodo: Nodo, indent: str) -> None:
        # Caso base: si el nodo es una hoja, imprime la predicción final de esa rama.
        if nodo.es_nodo_hoja():
            print(f"{indent}Predicción -> Clase: {nodo.valor}")
            return

        # Comprueba si el nodo corresponde a un atributo categórico (si no tiene un umbral).
        if nodo.umbral is None:
            # Para atributos categóricos, la pregunta es sobre el valor del atributo.
            pregunta = f"¿Cuál es el valor de '{nodo.atributo}'?"
        else:
            # La pregunta se basa en si el valor del atributo es menor o igual al umbral.
            pregunta = f"¿'{nodo.atributo}' <= {nodo.umbral}?"

        print(f"{indent}Nodo de decisión: {pregunta}")
        # Itera sobre cada rama (hijo) del nodo actual.
        for condicion, nodo_hijo in nodo.nodos_hijos.items():
            print(f"{indent}├─ Si es '{condicion}':")
            self._imprimir_nodo(nodo_hijo, indent + "│  ")
