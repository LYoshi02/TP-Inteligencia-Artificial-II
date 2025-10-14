class Nodo:
    def __init__(self, atributo=None, nodos_hijos=None, valor=None, clase_mas_comun=None, umbral=None):
        # Nombre del atributo para la decisión
        self.atributo: str = atributo
        # Diccionario para guardar las ramas hacia nodos de menor nivel en el árbol
        self.nodos_hijos: dict = nodos_hijos if nodos_hijos is not None else {}
        # Se setea un valor cuando el nodo es un nodo hoja
        self.valor = valor
        # Clase más común en este nodo
        self.clase_mas_comun = clase_mas_comun
        # Umbral para dividir atributos continuos
        self.umbral: float | None = umbral

    def es_nodo_hoja(self):
        return self.valor is not None