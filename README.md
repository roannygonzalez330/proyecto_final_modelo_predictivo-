# proyecto_final_modelo_predictivo-

# Proyecto Final: Modelo Predictivo Funcional
## Predicción de abandono de clientes de telecomunicaciones — Mes 5, Machine Learning Supervisado

Este proyecto construye, evalúa y valida dos modelos de Machine Learning sobre el mismo dataset de clientes de una empresa de telecomunicaciones (Telco Customer Churn, 7,043 clientes): uno de **regresión** (predecir cuánto tiempo lleva un cliente) y uno de **clasificación** (predecir si va a cancelar el servicio).

## Estructura del proyecto

- `01_preparacion.py` — Diagnóstico del dataset, limpieza de `TotalCharges`, eliminación de `customerID`, codificación de variables categóricas. Genera `telco_modelo_base.csv`, usado por el resto de los archivos.
- `02_regresion.py` — Modelo de Regresión Lineal para predecir `tenure`.
- `03_clasificacion.py` — Modelos de clasificación para predecir `Churn` (Partes 3, 4 y 5: entrenamiento, evaluación completa y validación).
- `04_cliente_hipotetico.py` — Prueba de ambos modelos sobre un cliente nuevo simulado.

## Modelo de Regresión Lineal

Predice cuántos meses lleva un cliente (`tenure`) a partir de su perfil demográfico y los servicios que tiene contratados.

**Resultado:** R² de 0.66, MAE de 11.57 meses, RMSE de 14.56 meses — en promedio, el modelo se equivoca por poco más de 11 meses al estimar la antigüedad de un cliente.

**Fuga de datos evitada:** `TotalCharges` fue excluido de los predictores porque está calculado, casi exactamente, a partir de `MonthlyCharges × tenure` — es decir, contiene de forma indirecta la respuesta que se quiere predecir. Al entrenar sin esta variable, el R² se mantuvo en un rango razonable (0.66), confirmando que no había fuga de información.

## Modelo de Clasificación — Comparación de dos modelos

Se entrenaron y compararon dos modelos para predecir `Churn`: **Regresión Logística** y **Árbol de Decisión**.

| Métrica | Regresión Logística | Árbol de Decisión (corregido) |
|---|---|---|
| Accuracy | 80.6% | 79.8% |
| Precision | 65.7% | 63.5% |
| Recall | 55.9% | 56.7% |
| F1 | 60.4% | 59.9% |
| AUC | 0.842 | 0.830 |

**Overfitting detectado y corregido:** el árbol de decisión, entrenado sin límite de profundidad, memorizó el conjunto de entrenamiento (99.8% de accuracy en train) pero rendía mucho peor en datos nuevos (72.1% en test) — una brecha de 27.7 puntos. Al limitar su profundidad a `max_depth=5`, la brecha se redujo a menos de 1 punto (80.4% train vs 79.8% test), confirmando que el ajuste corrigió el problema sin sacrificar rendimiento. La validación por cross-validation (5 folds) sobre la regresión logística mostró un rendimiento consistente (media 80.4%, desviación estándar de solo 0.014), reforzando la confianza en que los resultados no dependen de un split particular de los datos.

**Modelo elegido: Árbol de Decisión.** Aunque la regresión logística tiene mejor accuracy, precision y AUC, la diferencia entre ambos modelos es mínima una vez corregido el overfitting del árbol. Para este problema de negocio, es más costoso no detectar a un cliente que realmente va a cancelar (falso negativo) que contactar innecesariamente a uno que se iba a quedar (falso positivo) — por eso se prioriza el **recall**, donde el árbol de decisión tiene una ligera ventaja (56.7% vs 55.9%).

## Cliente hipotético

Se probó un cliente simulado (6 meses de antigüedad, contrato mes a mes, fibra óptica, pago con cheque electrónico, $90/mes) contra ambos modelos:

- **Regresión:** antigüedad estimada de ~11 meses.
- **Clasificación:** riesgo de abandono del **72%**, clasificado como cliente propenso a cancelar.

Este perfil —contrato sin compromiso, método de pago asociado a mayor abandono, y poca antigüedad— coincide con los factores de riesgo identificados en el análisis exploratorio del proyecto anterior (Mes 4), reforzando la consistencia de ambos análisis.