# utils_model.py
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Función para entrenar un modelo de clasificación
# y guardar el modelo y los codificadores de etiquetas
def entrenar_modelo(csv_path, modelo_id):
    df = pd.read_csv(csv_path)

    # Verificar que las columnas necesarias están en el DataFrame
    columnas_utiles = ["Age", "Type of Travel", "Class",
                       "Flight Distance", "Inflight entertainment",
                       "On-board service", "Cleanliness",
                       "Arrival Delay in Minutes", "Departure Delay in Minutes"]
    
    # Comprobar si las columnas existen en el DataFrame
    for col in columnas_utiles:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no está en el archivo CSV.")
    
    
    # Convertir la columna de satisfacción a binaria
    X = df[columnas_utiles]
    y = df["satisfaction"].apply(lambda x: 1 if x == "satisfied" else 0)
    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    

    # El algitmo analizado es RandomForestClassifier y entrena modelos de acuerdo a los # datos de entrada
    # y los guarda en la carpeta "models" y los codificadores en "encoders"
    model = RandomForestClassifier(n_estimators=100, random_state=42) #
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    os.makedirs("encoders", exist_ok=True)

    joblib.dump(model, f"models/model_{modelo_id}.pkl")
    joblib.dump(label_encoders, f"encoders/encoder_{modelo_id}.pkl")

    return True
