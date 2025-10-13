class Nodo:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """Constructor para un nodo. Si 'value' no es None, es un nodo hoja."""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def es_nodo_hoja(self):
        return self.value is not None


class NodoMulti:
    def __init__(self, feature=None, children=None, *, value=None, most_common_class=None):
        """
        Constructor para un nodo multirama.
        - feature: El nombre de la característica en la que se divide.
        - children: Un diccionario {valor_categoria: NodoHijo}.
        - value: Si es un nodo hoja, esta es la predicción.
        - most_common_class: La clase más común en este nodo (para predicciones de respaldo).
        """
        self.feature = feature
        self.children = children if children is not None else {}
        self.value = value
        self.most_common_class = most_common_class

    def es_nodo_hoja(self):
        return self.value is not None