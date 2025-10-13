import pandas as pd

from arbol_decision.algoritmo2 import C45

# =============================================================================
# PASO 1: CARGAR, PROCESAR Y ENTRENAR CON TUS DATOS
# =============================================================================

# --- Creación del DataFrame con los datos de la imagen ---
data1 = {
    'c': ['X', 'Y', 'Z', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Z'],
    'AtributoB': ['S', 'T', 'T', 'T', 'T', 'S', 'S', 'S', 'T', 'S'],
    'AtributoC': ['O', 'P', 'O', 'O', 'P', 'P', 'P', 'P', 'P', 'P'],
    'AtributoD': ['Q', 'R', 'R', 'R', 'Q', 'Q', 'Q', 'Q', 'R', 'Q'],
    'Clase': [2, 1, 3, 3, 3, 1, 1, 2, 2, 3]
}
data2 = {
    'Age': ['young', 'young', 'young', 'young', 'young', 'middle', 'middle', 'middle', 'middle', 'middle', 'old', 'old', 'old', 'old', 'old'],
    'Has_job': [False, False, True, True, False, False, False, True, False, False, False, False, True, True, False],
    'Own_house': [False, False, False, True, False, False, False, True, True, True, True, True, False, False, False],
    'Credit_rating': ['fair', 'good', 'good', 'fair', 'fair', 'fair', 'good', 'good', 'excellent', 'excellent', 'excellent', 'good', 'good', 'excellent', 'fair'],
    'Clase': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
data3 = {
    'Hipertenso': ['NO', 'SI', 'SI', 'SI', 'NO', 'NO', 'SI', 'SI', 'SI', 'NO'],
    'Colesterol': ['Bajo', 'Bajo', 'Bajo', 'Medio', 'Medio', 'Medio', 'Alto', 'Alto', 'Alto', 'Alto'],
    'Triglicéridos': ['Normal', 'Elevado', 'Elevado', 'Elevado', 'Elevado', 'Normal', 'Normal', 'Normal', 'Elevado', 'Normal'],
    'Edad': ['Menor 40', 'Menor 40', 'Mayor a 60', 'Mayor a 60', 'Menor 40', 'Entre 40-60', 'Entre 40-60', 'Entre 40-60', 'Mayor a 60', 'Menor 40'],
    'Diabético': ['SI', 'SI', 'SI', 'NO', 'NO', 'SI', 'SI', 'SI', 'NO', 'NO'],
    'Clase': ['NO', 'NO', 'SI', 'SI', 'SI', 'NO', 'SI', 'SI', 'SI', 'NO'] # Problemas Cardiacos
}
df = pd.DataFrame(data3)

print("--- Dataset Original ---")
print(df)
print("\n" + "=" * 25 + "\n")

# --- Separar características (X) y clase (y) ---
# X_categorical = df.drop('Clase', axis=1)
# y = df['Clase'].values
#
# # --- Preprocesamiento: Convertir datos categóricos a numéricos ---
# encoder = OrdinalEncoder()
# X = encoder.fit_transform(X_categorical)
#
# # Guardamos los nombres para la visualización
# feature_names = X_categorical.columns
# class_names = np.unique(y)  # Clases originales: [1, 2, 3]
#
# print("--- Dataset Preprocesado (Numérico) ---")
# print("Características (X):")
# print(X)
# print("\nClase (y):")
# print(y)
# print("\n" + "=" * 25 + "\n")

# --- Separar características (X) y clase (y) ---
X_train = df.drop('Clase', axis=1)
y_train = df['Clase']
y_train.name = 'Clase' # Es importante que la serie 'y' tenga un nombre

print("--- Usando el Dataset Original Directamente ---")
print(df)
print("\n" + "="*35 + "\n")

# --- ¡Entrenar el modelo multirama! ---
clf_multi = C45(max_depth=5)
clf_multi.fit(X_train, y_train)

# --- Entrenar el modelo C4.5 ---
clf = C45(max_depth=5)
# clf.fit(X, y)
clf.fit(X_train, y_train)


# =============================================================================
# PASO 3: FUNCIÓN PARA VISUALIZAR EL ÁRBOL EN TEXTO
# =============================================================================

def imprimir_arbol(nodo, feature_names, class_names, encoder, indent=""):
    """Función recursiva para imprimir el árbol de decisión en la consola."""
    if nodo.es_nodo_hoja():
        print(f"{indent}Predicción -> Clase: {nodo.value}")
        return

    # Obtenemos el nombre de la característica
    feature_name = feature_names[nodo.feature]

    # El umbral es numérico (ej: 0.5), lo que representa la división entre dos categorías
    # ej: si 'S' es 0 y 'T' es 1, un umbral de 0.5 separa S de T.
    # Para hacerlo más legible, identificamos qué categorías van a la izquierda
    feature_categories = encoder.categories_[nodo.feature]
    # El umbral está entre dos valores enteros. El valor entero es el umbral mismo.
    threshold_category_index = int(nodo.threshold)
    left_categories = feature_categories[:threshold_category_index + 1]

    print(f"{indent}¿{feature_name} está en {list(left_categories)}?")

    # Rama Izquierda (Sí)
    print(f"{indent}├─ Sí:")
    imprimir_arbol(nodo.left, feature_names, class_names, encoder, indent + "│  ")

    # Rama Derecha (No)
    print(f"{indent}└─ No:")
    imprimir_arbol(nodo.right, feature_names, class_names, encoder, indent + "   ")

def imprimir_arbol_multi(nodo, indent=""):
    if nodo.es_nodo_hoja():
        print(f"{indent}Predicción -> Clase: {nodo.value}")
        return

    print(f"{indent}Nodo de decisión: ¿Cuál es el valor de '{nodo.feature}'?")
    for value, child_node in nodo.children.items():
        print(f"{indent}├─ Si es '{value}':")
        imprimir_arbol_multi(child_node, indent + "│  ")

# --- Mostrar el árbol resultante ---
print("--- Árbol de Decisión C4.5 Resultante ---")
# imprimir_arbol(clf.root, feature_names, class_names, encoder)
imprimir_arbol_multi(clf.root)