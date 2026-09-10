import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

df = pd.read_csv('../data/telco_modelo_base.csv')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

y = df['Churn']
X = df.drop(columns=['Churn'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

columnas_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[columnas_numericas] = scaler.fit_transform(X_train[columnas_numericas])
X_test[columnas_numericas] = scaler.transform(X_test[columnas_numericas])

modelo_log = LogisticRegression(max_iter=1000, random_state=42)
modelo_log.fit(X_train, y_train)

modelo_arbol = DecisionTreeClassifier(random_state=42)
modelo_arbol.fit(X_train, y_train)

def evaluar(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print(f"\n===== {nombre} =====")
    print("Matriz de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:   ", round(recall_score(y_test, y_pred), 4))
    print("F1:       ", round(f1_score(y_test, y_pred), 4))
    print("AUC:      ", round(roc_auc_score(y_test, y_proba), 4))

evaluar("Regresion Logistica", modelo_log, X_test, y_test)
evaluar("Arbol de Decision", modelo_arbol, X_test, y_test)
