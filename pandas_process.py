import pandas as pd

# Función para normalizar un archivo de bitnetflow


def normalize_bitnetflow(file_path, output_path):
    df = pd.read_csv(file_path)

    # # Imprimir los tipos de datos originales
    # print("Tipos de datos originales:")
    # print(df.dtypes)

    # Normalizar formatos de fecha y hora
    df["StartTime"] = pd.to_datetime(
        df["StartTime"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    df["StartTime"] = df["StartTime"].fillna(
        pd.to_datetime(df["StartTime"],
                       format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    )
    df["StartTime"] = df["StartTime"].dt.strftime("%Y/%m/%d %H:%M:%S.%f")

    # Convertir el Protocolo a minúsculas
    df["Proto"] = df["Proto"].str.lower()

    # Manejar valores NA en los puertos y convertirlos a enteros
    df["Sport"] = df["Sport"].fillna(0).astype(int)
    df["Dport"] = df["Dport"].fillna(0).astype(int)

    # # Imprimir los tipos de datos después de la conversión
    # print("Tipos de datos después de la conversión:")
    # print(df.dtypes)

    # Asegurar que sTos y dTos sean consistentes
    df["sTos"] = df["sTos"].astype(str).replace("0", "0.0")
    df["dTos"] = df["dTos"].astype(str).replace("0", "0.0")

    # Guardar el archivo normalizado en formato binetflow
    with open(output_path, "w") as f:
        # Escribir cabeceras
        headers = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
        f.write(headers)
        # Escribir cada fila de datos
        for index, row in df.iterrows():
            line = (
                f"{row['StartTime']},{row['Dur']},{row['Proto']},"
                f"{row['SrcAddr']},{row['Sport']},{row['Dir']},"
                f"{row['DstAddr']},{row['Dport']},{row['State']},"
                f"{row['sTos']},{row['dTos']},{row['TotPkts']},"
                f"{row['TotBytes']},{row['SrcBytes']},{row['Label']}\n"
            )
            f.write(line)
    print(f"Archivo binetflow normalizado creado exitosamente en: {
          output_path}")


# Ejemplo de uso de la función
#general
# file_path = "database/otras/flow_analysis.binetflow"
# output_path = "database/0.binetflow"

#especifico
file_path = "database/otras/flow_analysis_31_12_24.binetflow"
output_path = "database/1.binetflow"


normalize_bitnetflow(file_path, output_path)
