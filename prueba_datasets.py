import os
import sys
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from mi_arbol_decision.algoritmo3 import ArbolDecision, FUNCIONES_IMPUREZA

@contextmanager
def silenciar_output():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout

# === 1. Cargar el dataset desde un CSV local ===
nombre_csv_dataset: str = "data_cardiovascular_risk_LIMPIO_BALANCEADO.csv"
# nombre_csv_dataset: str = "data_cardiovascular_risk_LIMPIO_DESBALANCEADO.csv"
df_completo: DataFrame = pd.read_csv(f"datasets/{nombre_csv_dataset}")

# === 2. Dividir los datos 80% entrenamiento / 20% prueba ===
df_entrenamiento, df_prueba = train_test_split(df_completo, test_size=0.3, random_state=42)

# === 3. Entrenar el árbol con los datos de entrenamiento ===
nombre_objetivo: str = "TenYearCHD"

arbol_decision = ArbolDecision(funcion_impureza=FUNCIONES_IMPUREZA.ganancia_informacion)

print("Entrenando arbol...")
inicio_entrenamiento = time.time()
with silenciar_output():
    arbol_decision.entrenar(df_entrenamiento, nombre_objetivo)
fin_entrenamiento = time.time()
print("Fin entrenamiento arbol.")

tiempo_entrenamiento = fin_entrenamiento - inicio_entrenamiento
print(f"\n⏱️ Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} segundos")

# === 4. Predecir las clases del conjunto de prueba ===
predicciones = []
reales = []

atributos = list(df_completo.drop(nombre_objetivo, axis=1).columns)
for _, fila in df_prueba.iterrows():
    instancia = {col: fila[col] for col in atributos}
    prediccion = arbol_decision.predecir(instancia)
    predicciones.append(prediccion)
    reales.append(fila[nombre_objetivo])

# === 5. Calcular precisión general ===
print("\n--- RESULTADOS ---")
aciertos = sum(1 for r, p in zip(reales, predicciones) if r == p)
precision = aciertos / len(reales) * 100
print(f"\nPrecisión general del árbol de decisión: {precision:.2f}%")

# === 6. Mostrar métricas detalladas ===
print("\n📊 Matriz de confusión:")
print(confusion_matrix(reales, predicciones))
# cm = confusion_matrix(reales, predicciones)
# labels = [0, 1]
#
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.title("Matriz de Confusión del Árbol de Decisión")
# plt.xlabel("Clase Predicha")
# plt.ylabel("Clase Real")
# plt.tight_layout()
# plt.show()

print("\n📈 Reporte de clasificación:")
print(classification_report(reales, predicciones, digits=2))
# Convertir el reporte de clasificación en un DataFrame
reporte_dict = classification_report(reales, predicciones, output_dict=True)
df_reporte = pd.DataFrame(reporte_dict).transpose()

# Filtrar solo las clases (evitamos accuracy, macro avg, etc.)
# df_metricas = df_reporte.iloc[:2, :3]  # solo clases 0 y 1, y métricas precision, recall, f1-score
#
# plt.figure(figsize=(7, 5))
# df_metricas.plot(kind='bar')
# plt.title("Métricas por Clase (Precisión, Recall, F1-score)")
# plt.ylabel("Valor")
# plt.ylim(0, 1)
# plt.xticks(rotation=0)
# plt.legend(title="Métrica")
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()