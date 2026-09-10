import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# ============================================================================
# PARTE 6: Cliente hipotetico
# Se reentrenan ambos modelos ya elegidos (mismo random_state, mismo resultado
# que en 02_regresion.py y 03_clasificacion.py), para que este archivo sea
# independiente y no dependa de haber corrido los otros antes en la misma sesion.
# ============================================================================

df = pd.read_csv('../data/telco_modelo_base.csv')

# Lista de columnas categoricas usada en la Parte 1 para el encoding
columnas_categoricas = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                         'PaperlessBilling', 'PaymentMethod']

# --- Reentrenar el modelo de REGRESION (Parte 2) ---
y_reg = df['tenure']
X_reg = df.drop(columns=['tenure', 'TotalCharges', 'Churn'])
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
modelo_reg = LinearRegression()
modelo_reg.fit(X_train_reg, y_train_reg)

# --- Reentrenar el modelo de CLASIFICACION elegido (arbol corregido, Parte 3/5) ---
df_clf = df.copy()
df_clf['Churn'] = df_clf['Churn'].map({'Yes': 1, 'No': 0})
y_clf = df_clf['Churn']
X_clf = df_clf.drop(columns=['Churn'])
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
columnas_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train_clf[columnas_numericas] = scaler.fit_transform(X_train_clf[columnas_numericas])

modelo_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_clf.fit(X_train_clf, y_train_clf)

print("Modelos reentrenados y listos.")
print("Regresion  -> columnas de entrada:", X_train_reg.shape[1])
print("Clasificacion -> columnas de entrada:", X_train_clf.shape[1])


# --- Construir el cliente nuevo, con los mismos campos del dataset original ---
cliente_nuevo = pd.DataFrame([{
    'customerID': 'NEW-0001',
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 6,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 90.00,
    'TotalCharges': 540.00,
}])

print("\nPerfil del cliente hipotetico:")
print(cliente_nuevo.T)

# Codificar el cliente igual que se codifico el dataset de entrenamiento
cliente_enc = pd.get_dummies(cliente_nuevo, columns=columnas_categoricas, drop_first=False)
bool_cols = cliente_enc.select_dtypes(bool).columns
cliente_enc[bool_cols] = cliente_enc[bool_cols].astype(int)

# --- Prediccion de REGRESION: cuanto tiempo se estima que permanecera ---
# Se alinean las columnas del cliente con las que espera el modelo de regresion
# (sin tenure ni TotalCharges, igual que en el entrenamiento). get_dummies() solo
# genero columnas para las categorias que ESTE cliente tiene, por eso hace falta
# reindex: agrega con 0 cualquier columna que el cliente no disparo.
cliente_reg = cliente_enc.reindex(columns=X_train_reg.columns, fill_value=0)
tenure_predicho = modelo_reg.predict(cliente_reg)
print(f"\nTenure estimado (Parte 2): {tenure_predicho[0]:.2f} meses")

# --- Prediccion de CLASIFICACION: va a cancelar o no ---
# Aqui si se incluye tenure y TotalCharges (son predictores validos para Churn).
# Se reindexa contra las columnas del modelo de clasificacion, y se escala con
# el MISMO scaler ya ajustado con el train de clasificacion (no uno nuevo).
cliente_clf = cliente_enc.reindex(columns=X_train_clf.columns, fill_value=0)
cliente_clf[columnas_numericas] = scaler.transform(cliente_clf[columnas_numericas])

churn_predicho = modelo_clf.predict(cliente_clf)
probabilidad_churn = modelo_clf.predict_proba(cliente_clf)[:, 1]

print(f"\nPrediccion de Churn (Parte 3): {'Se va (Churn)' if churn_predicho[0] == 1 else 'Se queda (No Churn)'}")
print(f"Probabilidad de abandono: {probabilidad_churn[0]:.2%}")