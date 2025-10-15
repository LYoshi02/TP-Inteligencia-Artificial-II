class Atributo:
    def __init__(self, nombre: str = '', ganancia: float = -1, umbral: float | None = None):
        self.nombre: str = nombre
        self.ganancia: float = ganancia
        self.umbral: float | None = umbral

    def es_categorico(self) -> bool:
        return self.umbral is None

    def __str__(self):
        return self.nombre
