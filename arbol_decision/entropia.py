import numpy as np

# --- Función para calcular la entropía ---
def calcular_entropia(y):
    """Calcula la entropía de un conjunto de etiquetas."""
    # np.bincount es eficiente para contar ocurrencias de enteros no negativos
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def calcular_entropia2(s):
    """Calcula la entropía de una Serie de Pandas."""
    # value_counts() es perfecto para contar las clases
    counts = s.value_counts()
    total = len(s)
    ps = counts / total
    return -np.sum([p * np.log2(p) for p in ps if p > 0])