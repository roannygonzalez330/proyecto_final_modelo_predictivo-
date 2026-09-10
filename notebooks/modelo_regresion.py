import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('../data/telco_modelo_base.csv')

# tenure es la variable objetivo.
# TotalCharges se excluye: es fuga de datos (viene de MonthlyCharges x tenure,
# literalmente lo que queremos predecir).
# Churn se excluye tambien: sigue como texto sin codificar (es el objetivo de la
# Parte 3, no un predictor aqui) y ademas LinearRegression no acepta texto.
y = df['tenure']
X = df.drop(columns=['tenure', 'TotalCharges', 'Churn'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMAE (Error Absoluto Medio):", round(mae, 2))
print("RMSE (Raiz del Error Cuadratico Medio):", round(rmse, 2))
print("R2 (Coeficiente de determinacion):", round(r2, 4))