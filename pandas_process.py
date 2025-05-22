import pandas as pd

# Función para normalizar un archivo de bitnetflow
def normalize_bitnetflow(file_path, output_path):
    # Leer el archivo sin especificar dtype manualmente
    df = pd.read_csv(file_path, header=0, low_memory=False)

    # Normalizar formatos de fecha y hora
    df["StartTime"] = pd.to_datetime(df["StartTime"], errors="coerce")
    df["StartTime"] = df["StartTime"].dt.strftime("%Y/%m/%d %H:%M:%S.%f")

    # Convertir el Protocolo a minúsculas
    df["Proto"] = df["Proto"].str.lower()

    # Manejar valores NA en los puertos y convertirlos a enteros solo si es posible
    df["Sport"] = pd.to_numeric(df["Sport"], errors="coerce").fillna(0).astype(int)
    df["Dport"] = pd.to_numeric(df["Dport"], errors="coerce").fillna(0).astype(int)

    # Asegurar que sTos y dTos sean consistentes
    df["sTos"] = pd.to_numeric(df["sTos"], errors="coerce").fillna(0).astype(float)
    df["dTos"] = pd.to_numeric(df["dTos"], errors="coerce").fillna(0).astype(float)

    # Guardar el archivo normalizado en formato binetflow
    df.to_csv(output_path, index=False)

    print(f"Archivo binetflow normalizado creado exitosamente en: {output_path}")

# Ejemplo de uso
file_path = "database/capturas/5K.binetflow"
output_path = "database/4.binetflow"

normalize_bitnetflow(file_path, output_path)
