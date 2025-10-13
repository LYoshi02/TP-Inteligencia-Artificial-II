from collections import Counter
import numpy as np

from arbol_decision.entropia import calcular_entropia
from arbol_decision.nodo import Nodo

class C45:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """Función principal para entrenar (construir) el árbol."""
        self.root = self._construir_arbol(X, y)

    def _construir_arbol(self, X, y, depth=0):
        """Función recursiva para construir el árbol."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Criterios de parada
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._clase_mas_comun(y)
            return Nodo(value=leaf_value)

        # Encontrar la mejor división usando Tasa de Ganancia
        best_feat, best_thresh = self._encontrar_mejor_division(X, y)

        # Si la ganancia es 0, no podemos dividir más
        if best_feat is None:
            leaf_value = self._clase_mas_comun(y)
            return Nodo(value=leaf_value)

        # Dividir los datos y construir sub-árboles
        left_idxs, right_idxs = self._dividir(X[:, best_feat], best_thresh)
        left = self._construir_arbol(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._construir_arbol(X[right_idxs, :], y[right_idxs], depth + 1)

        return Nodo(best_feat, best_thresh, left, right)

    def _encontrar_mejor_division(self, X, y):
        """
        Itera sobre todas las características y umbrales para encontrar la división
        que maximice la Tasa de Ganancia.
        """
        best_gain_ratio = -1
        split_idx, split_thresh = None, None
        n_features = X.shape[1]

        # Entropía del nodo actual (padre)
        parent_entropy = calcular_entropia(y)

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Calcular la Ganancia de Información
                gain = self._ganancia_informacion(y, X_column, threshold, parent_entropy)

                # Si la ganancia es 0, la Tasa de Ganancia también lo será
                if gain == 0:
                    continue

                # Calcular la Información de División
                split_info = self._split_info(y, X_column, threshold)

                # Evitar división por cero
                if split_info == 0:
                    continue

                # Calcular la Tasa de Ganancia
                gain_ratio = gain / split_info

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _ganancia_informacion(self, y, X_column, split_thresh, parent_entropy):
        """Calcula la Ganancia de Información."""
        # Dividir los datos
        left_idxs, right_idxs = self._dividir(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calcular la entropía ponderada de los hijos
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = calcular_entropia(y[left_idxs]), calcular_entropia(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Ganancia de información
        ig = parent_entropy - child_entropy
        return ig

    def _split_info(self, y, X_column, split_thresh):
        """Calcula la Información de División."""
        left_idxs, right_idxs = self._dividir(X_column, split_thresh)

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        # Si una rama está vacía, el split info es 0
        if n_l == 0 or n_r == 0:
            return 0

        p_l = n_l / n
        p_r = n_r / n

        # Fórmula de Split Info
        return - (p_l * np.log2(p_l) + p_r * np.log2(p_r))

    def _dividir(self, X_column, split_thresh):
        """Devuelve los índices para las ramas izquierda y derecha."""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _clase_mas_comun(self, y):
        """Devuelve la etiqueta más común en un array."""
        counter = Counter(y)
        if not counter: return None
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Realiza predicciones para un conjunto de datos X."""
        return np.array([self._atravesar_arbol(x, self.root) for x in X])

    def _atravesar_arbol(self, x, nodo):
        """Navega recursivamente el árbol para clasificar una sola muestra 'x'."""
        if nodo.es_nodo_hoja():
            return nodo.value

        if x[nodo.feature] <= nodo.threshold:
            return self._atravesar_arbol(x, nodo.left)
        return self._atravesar_arbol(x, nodo.right)