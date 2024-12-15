import numpy as np
import pandas as pd

# Generar datos sintéticos para el archivo de mantenimiento con 200 registros
np.random.seed(42)  # Fijar la semilla para reproducibilidad

data_mantenimiento_200 = {
    'Horas de operación': np.random.randint(500, 1500, size=200),
    'Temperatura del motor': np.random.randint(80, 130, size=200),
    'Vibración del motor': np.random.uniform(0.5, 1.5, size=200).round(2),
    'Presión hidráulica': np.random.randint(200000, 300000, size=200),
    'Falla': np.random.choice([0, 1], size=200)  # 0: No Falla, 1: Falla
}

# Crear DataFrame
df_mantenimiento_200 = pd.DataFrame(data_mantenimiento_200)

# Guardar en un archivo CSV
file_path = "mantenimiento_equipos_200.csv"
df_mantenimiento_200.to_csv(file_path, index=False)

file_path