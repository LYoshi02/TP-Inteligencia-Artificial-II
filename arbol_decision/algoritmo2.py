import pandas as pd

from arbol_decision.entropia import calcular_entropia2
from arbol_decision.nodo import NodoMulti

class C45:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Entrena el árbol. X es un DataFrame de Pandas, y es una Serie de Pandas."""
        # Combinamos X e y para facilitar el filtrado
        self.dataset = pd.concat([X, y], axis=1)
        self.target_name = y.name
        self.root = self._construir_arbol(self.dataset)

    def _construir_arbol(self, df, depth=0):
        """Función recursiva para construir el árbol usando DataFrames."""
        X, y = df.drop(self.target_name, axis=1), df[self.target_name]
        n_samples, n_features = X.shape
        n_labels = len(y.unique())

        # Clase más común en este nodo (útil para hojas y predicciones)
        most_common = y.mode()[0] if not y.empty else None

        # Criterios de parada
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return NodoMulti(value=most_common, most_common_class=most_common)

        # Encontrar el mejor atributo para dividir
        best_feature = self._encontrar_mejor_atributo(df)

        # Si no hay ganancia de información, crear una hoja
        if best_feature is None:
            return NodoMulti(value=most_common, most_common_class=most_common)

        # Crear un nodo de decisión y construir los hijos
        children = {}
        for value in df[best_feature].unique():
            # Filtra el dataset para cada valor de la característica
            subset_df = df[df[best_feature] == value].drop(best_feature, axis=1)
            # ¡Llamada recursiva!
            children[value] = self._construir_arbol(subset_df, depth + 1)

        return NodoMulti(feature=best_feature, children=children, most_common_class=most_common)

    def _encontrar_mejor_atributo(self, df):
        """Encuentra la característica que maximiza la Ganancia de Información."""
        y = df[self.target_name]
        features = df.drop(self.target_name, axis=1).columns

        parent_entropy = calcular_entropia2(y)
        best_gain = -1
        best_feature = None

        for feature in features:
            # Calcular la entropía ponderada de los hijos
            unique_values = df[feature].unique()
            child_entropy = 0
            for value in unique_values:
                subset_y = df[df[feature] == value][self.target_name]
                weight = len(subset_y) / len(y)
                child_entropy += weight * calcular_entropia2(subset_y)

            # Calcular Ganancia de Información
            info_gain = parent_entropy - child_entropy

            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature

        # Si no hay ganancia (best_gain <= 0), no hay un buen atributo para dividir
        return best_feature if best_gain > 0 else None

    def predict(self, X_test):
        """Realiza predicciones para un conjunto de datos X_test (DataFrame)."""
        return X_test.apply(self._atravesar_arbol, axis=1, args=(self.root,))

    def _atravesar_arbol(self, x, nodo):
        """Navega el árbol para una sola muestra 'x' (una fila del DataFrame)."""
        if nodo.es_nodo_hoja():
            return nodo.value

        # Obtener el valor de la característica de la muestra de entrada
        feature_value = x.get(nodo.feature)

        # Buscar el siguiente nodo en los hijos
        child_node = nodo.children.get(feature_value)

        # Si el valor no se vio en el entrenamiento, no habrá una rama
        if child_node is None:
            # ¡Fallback! Devolvemos la clase más común del nodo actual.
            return nodo.most_common_class

        return self._atravesar_arbol(x, child_node)