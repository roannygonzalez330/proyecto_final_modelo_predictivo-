import pandas as pd

df = pd.read_csv('../data/telco.csv')

print("Forma del dataset:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)
print("\nNulos:", df.isnull().sum().sum())

# TotalCharges se carga como texto y tiene filas en blanco
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nNulos en TotalCharges tras convertir a numerico:", df['TotalCharges'].isnull().sum())
print("Tenure de esos clientes:", df.loc[df['TotalCharges'].isnull(), 'tenure'].unique())

# Son clientes nuevos (tenure=0), sin cargo acumulado aun -> se rellenan con 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print("Nulos totales despues de resolver TotalCharges:", df.isnull().sum().sum())

# customerID no aporta valor predictivo
df = df.drop(columns=['customerID'])

# Columnas de texto a codificar como predictoras (Churn se deja para la Parte 3)
columnas_categoricas = df.drop(columns=['Churn']).select_dtypes(exclude='number').columns.tolist()
print("\nColumnas a codificar:", columnas_categoricas)

df_encoded = pd.get_dummies(df, columns=columnas_categoricas, drop_first=False)

bool_cols = df_encoded.select_dtypes(bool).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print("\nForma final:", df_encoded.shape)
print("Nulos:", df_encoded.isnull().sum().sum())
print("\nChurn (se deja sin codificar, se convertira en la Parte 3):", df_encoded['Churn'].unique())

df_encoded.to_csv('../data/telco_modelo_base.csv', index=False)
print("\nGuardado: data/telco_modelo_base.csv")