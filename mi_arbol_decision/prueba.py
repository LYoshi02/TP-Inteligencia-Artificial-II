import pandas as pd

# from mi_arbol_decision.algoritmo import ArbolDecision, Nodo
from mi_arbol_decision.algoritmo2 import ArbolDecision, Nodo

def imprimir_arbol(nodo, indent=""):
    if nodo.es_nodo_hoja():
        print(f"{indent}Predicción -> Clase: {nodo.valor}")
        return

    print(f"{indent}Nodo de decisión: ¿Cuál es el valor de '{nodo.atributo}'?")
    for valor, nodo_hijo in nodo.nodos_hijos.items():
        print(f"{indent}├─ Si es '{valor}':")
        imprimir_arbol(nodo_hijo, indent + "│  ")

def imprimir_arbol2(nodo: Nodo, indent=""):
    """
    Imprime recursivamente el árbol de decisión de una manera visualmente clara.
    Maneja tanto nodos de atributos categóricos como continuos.
    """
    # Caso base: si el nodo es una hoja, imprime la predicción final de esa rama.
    if nodo.es_nodo_hoja():
        print(f"{indent}Predicción -> Clase: {nodo.valor}")
        return

    # Comprueba si el nodo corresponde a un atributo continuo (si tiene un umbral).
    if nodo.umbral is not None:
        # La pregunta se basa en si el valor del atributo es menor o igual al umbral.
        pregunta = f"¿'{nodo.atributo}' <= {nodo.umbral}?"
    else:
        # Para atributos categóricos, la pregunta es sobre el valor del atributo.
        pregunta = f"¿Cuál es el valor de '{nodo.atributo}'?"

    print(f"{indent}Nodo de decisión: {pregunta}")

    # Itera sobre cada rama (hijo) del nodo actual.
    # La variable 'condicion' ya contiene la descripción de la rama,
    # ya sea un valor categórico ('soleado') o una condición continua ('<= 30.5').
    for condicion, nodo_hijo in nodo.nodos_hijos.items():
        print(f"{indent}├─ Si es '{condicion}':")
        imprimir_arbol(nodo_hijo, indent + "│  ")

data_ejercicio_2 = {
    'Formación secundaria': ['Técnica', 'Técnica', 'No Técnica', 'No Técnica', 'Técnica', 'Técnica'],
    'Programa': ['SI', 'SI', 'SI', 'NO', 'SI', 'NO'],
    'Menos de 3 inasistencias a clase': ['SI', 'SI', 'SI', 'NO', 'NO', 'NO'],
    'Condicion': ['Regular', 'Regular', 'Regular', 'Libre', 'Regular', 'Libre']
}

data_ejercicio_3 = {
    'Hipertenso': ['NO', 'SI', 'SI', 'SI', 'NO', 'NO', 'SI', 'SI', 'SI', 'NO'],
    'Colesterol': ['Bajo', 'Bajo', 'Bajo', 'Medio', 'Medio', 'Medio', 'Alto', 'Alto', 'Alto', 'Alto'],
    'Triglicéridos': ['Normal', 'Elevado', 'Elevado', 'Elevado', 'Elevado', 'Normal', 'Normal', 'Normal', 'Elevado', 'Normal'],
    'Edad': ['Menor 40', 'Menor 40', 'Mayor a 60', 'Mayor a 60', 'Menor 40', 'Entre 40-60', 'Entre 40-60', 'Entre 40-60', 'Mayor a 60', 'Menor 40'],
    'Diabético': ['SI', 'SI', 'SI', 'NO', 'NO', 'SI', 'SI', 'SI', 'NO', 'NO'],
    'Clase': ['NO', 'NO', 'SI', 'SI', 'SI', 'NO', 'SI', 'SI', 'SI', 'NO'] # Problemas Cardiacos
}

data_video_yt = {
    'Loves Popcorn': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No'],
    'Loves Soda': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No'],
    'Age': [7, 12, 18, 35, 38, 50, 83],
    'Loves Cool As Ice': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No']
}

nombre_objetivo: str = 'Loves Cool As Ice'

pd_dataframe = pd.DataFrame(data_video_yt)

print("----- DATAFRAME ORIGINAL -----")
print(pd_dataframe)
print("\n\n")

arbol_decision = ArbolDecision()
arbol_decision.entrenar(pd_dataframe, nombre_objetivo)

print("\n\n")
imprimir_arbol2(arbol_decision.raiz_arbol)