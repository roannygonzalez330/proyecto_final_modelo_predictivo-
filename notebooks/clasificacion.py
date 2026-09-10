import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score, roc_curve)

import os

os.makedirs('../graficas', exist_ok=True)

# ============================================================================
# PARTE 3: Modelo de clasificacion -- predecir el abandono (Churn)
# ============================================================================

df = pd.read_csv('../data/telco_modelo_base.csv')

# Convertir Churn a binario: los modelos de sklearn necesitan numeros, no texto.
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Diagnostico de desbalance: si una clase domina mucho, accuracy solo no alcanza
# para evaluar el modelo (un modelo que siempre dijera "No" ya acertaria 73%).
print("Distribucion de Churn (proporcion):")
print(df['Churn'].value_counts(normalize=True).round(4))

# A diferencia del modelo de regresion (Parte 2), aqui SI se dejan MonthlyCharges
# y TotalCharges como predictores: la fuga de datos de la Parte 2 era especifica
# de predecir tenure, no aplica de la misma forma al predecir Churn.
y = df['Churn']
X = df.drop(columns=['Churn'])

# stratify=y mantiene la misma proporcion de Churn en train y test, para que el
# split no quede desbalanceado por casualidad.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTamaño del split:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print("Proporcion de Churn en train:", round(y_train.mean(), 4))
print("Proporcion de Churn en test: ", round(y_test.mean(), 4))

# Escalado: se ajusta (fit) SOLO con train, y se aplica (transform) a test.
# Si se ajustara con todos los datos, el modelo "veria" estadisticas del test
# antes de entrenar (data leakage).
columnas_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])
X_test[columnas_numericas] = scaler.transform(X_test[columnas_numericas])

# Modelo 1: Regresion logistica -- modelo lineal, rapido, facil de interpretar.
modelo_log = LogisticRegression(max_iter=1000, random_state=42)
modelo_log.fit(X_train, y_train)

# Modelo 2: Arbol de decision, A PROPOSITO sin limite de profundidad.
# Un arbol sin restricciones puede crecer hasta memorizar el set de entrenamiento
# casi perfectamente -- lo dejamos asi por ahora para poder DETECTAR ese problema
# en la Parte 5, en vez de ocultarlo limitando la profundidad desde el inicio.
modelo_arbol = DecisionTreeClassifier(random_state=42)
modelo_arbol.fit(X_train, y_train)

print("\nAccuracy Regresion Logistica -> train:", round(modelo_log.score(X_train, y_train), 4),
      "| test:", round(modelo_log.score(X_test, y_test), 4))
print("Accuracy Arbol (sin limite)  -> train:", round(modelo_arbol.score(X_train, y_train), 4),
      "| test:", round(modelo_arbol.score(X_test, y_test), 4))


# ============================================================================
# PARTE 4: Evaluacion completa
# ============================================================================

def evaluar(nombre, modelo, X_test, y_test, guardar_graficas=False, sufijo=''):
    """Calcula e imprime las metricas de clasificacion, y opcionalmente guarda
    la matriz de confusion como imagen."""
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    matriz = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n===== {nombre} =====")
    print("Matriz de confusion (formato [[VN, FP], [FN, VP]]):")
    print(matriz)
    print("Accuracy: ", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:   ", round(recall, 4))
    print("F1:       ", round(f1, 4))
    print("AUC:      ", round(auc, 4))

    if guardar_graficas:
        # Matriz de confusion como mapa de calor, mas facil de leer que numeros sueltos
        plt.figure(figsize=(5, 4))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predijo: No', 'Predijo: Si'],
                    yticklabels=['Real: No', 'Real: Si'])
        plt.title(f'Matriz de Confusion - {nombre}')
        plt.tight_layout()
        plt.savefig(f'../graficas/matriz_confusion_{sufijo}.png')
        plt.show()

    return {'nombre': nombre, 'accuracy': acc, 'precision': precision,
            'recall': recall, 'f1': f1, 'auc': auc, 'y_proba': y_proba}

print("\n\n--- Evaluacion inicial (el arbol SIN limite, todavia con overfitting) ---")
res_log = evaluar("Regresion Logistica", modelo_log, X_test, y_test,
                   guardar_graficas=True, sufijo='regresion_logistica')
res_arbol_roto = evaluar("Arbol de Decision (sin limite)", modelo_arbol, X_test, y_test,
                          guardar_graficas=True, sufijo='arbol_sin_limite')


# ============================================================================
# PARTE 5: Validacion
# ============================================================================

# Paso 1: comparar accuracy train vs test. Una brecha grande es señal de overfitting
# (el modelo memorizo el train en vez de aprender un patron generalizable).
print("\n\n--- Diagnostico de overfitting (accuracy train vs test) ---")
print("Regresion Logistica -> train:", round(modelo_log.score(X_train, y_train), 4),
      "| test:", round(modelo_log.score(X_test, y_test), 4),
      "| brecha:", round(modelo_log.score(X_train, y_train) - modelo_log.score(X_test, y_test), 4))
print("Arbol sin limite     -> train:", round(modelo_arbol.score(X_train, y_train), 4),
      "| test:", round(modelo_arbol.score(X_test, y_test), 4),
      "| brecha:", round(modelo_arbol.score(X_train, y_train) - modelo_arbol.score(X_test, y_test), 4))

# Paso 2: cross-validation (cv=5). Entrena y evalua el modelo 5 veces, cada vez
# con una porcion distinta como validacion. Si el rendimiento es consistente entre
# folds (desviacion estandar baja), da confianza de que el resultado no depende
# de "tener suerte" con un split en particular.
cv_scores = cross_val_score(modelo_log, X_train, y_train, cv=5)
print("\nCross-validation (Regresion Logistica), 5 folds:")
print("Scores por fold:", cv_scores.round(4))
print("Media:", round(cv_scores.mean(), 4), "| Desviacion estandar:", round(cv_scores.std(), 4))

# Paso 3: el arbol mostro overfitting severo. Se corrige limitando max_depth
# (que tan profundo puede crecer el arbol), y se compara antes/despues del ajuste.
modelo_arbol_fix = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_arbol_fix.fit(X_train, y_train)

print("\nArbol ANTES del ajuste (sin limite) -> train:", round(modelo_arbol.score(X_train, y_train), 4),
      "| test:", round(modelo_arbol.score(X_test, y_test), 4))
print("Arbol DESPUES del ajuste (max_depth=5) -> train:", round(modelo_arbol_fix.score(X_train, y_train), 4),
      "| test:", round(modelo_arbol_fix.score(X_test, y_test), 4))

# Comparacion final JUSTA: se evalua el arbol YA CORREGIDO (no el que tenia
# overfitting), para que la comparacion contra la regresion logistica sea valida.
print("\n\n--- Comparacion final (arbol YA CORREGIDO, max_depth=5) ---")
res_arbol_fix = evaluar("Arbol de Decision (max_depth=5, corregido)", modelo_arbol_fix, X_test, y_test,
                         guardar_graficas=True, sufijo='arbol_corregido')

# Tabla resumen de metricas, para comparar los dos modelos de un vistazo
tabla_comparativa = pd.DataFrame({
    'Regresion Logistica': [res_log['accuracy'], res_log['precision'], res_log['recall'],
                             res_log['f1'], res_log['auc']],
    'Arbol de Decision (corregido)': [res_arbol_fix['accuracy'], res_arbol_fix['precision'],
                                       res_arbol_fix['recall'], res_arbol_fix['f1'], res_arbol_fix['auc']]
}, index=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])

print("\nTabla comparativa final:")
print(tabla_comparativa.round(4))

# Curva ROC de ambos modelos superpuestas, para visualizar la separacion
# entre clases (que tan bien distingue "se va" de "no se va")
plt.figure(figsize=(6, 5))
for nombre, resultado in [('Regresion Logistica', res_log), ('Arbol corregido', res_arbol_fix)]:
    fpr, tpr, _ = roc_curve(y_test, resultado['y_proba'])
    plt.plot(fpr, tpr, label=f"{nombre} (AUC = {resultado['auc']:.3f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Modelo al azar (AUC = 0.5)')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
plt.title('Curva ROC - Comparacion de modelos')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('../graficas/curva_roc_comparacion.png')
plt.show()