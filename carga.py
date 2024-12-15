import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Carga de datos
file_path = 'advertising.xlsx'
df = pd.read_excel(file_path)

# Mostrar las primeras filas del DataFrame
print(df.head())

# Preprocesamiento de datos

# Convertir la columna 'date' a un formato datetime si es necesario
df['date'] = pd.to_datetime(df['date'])

# Eliminar la columna 'date' si no se va a utilizar directamente en el modelo
df = df.drop(columns=['date'])

# Verificar si hay valores nulos y eliminarlos o imputarlos
df = df.dropna()

# Separar las características (X) y la variable objetivo (y)
X = df[['newspaper', 'tv', 'radio']]
y = df['sales']

# Entrenar un modelo de regresión lineal


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo


# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el R²
r2 = r2_score(y_test, y_pred)
print(f'R²: {r2}')

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Visualización
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs Predicciones')
plt.show()

