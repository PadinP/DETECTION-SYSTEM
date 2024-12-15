import pandas as pd

# Función para normalizar un archivo de bitnetflow


def normalize_bitnetflow(file_path, output_path):
    df = pd.read_csv(file_path)

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

    # Eliminar puntos decimales en puertos y asegurar valores enteros
    df["Sport"] = df["Sport"].apply(
        lambda x: int(float(x)) if pd.notnull(x) else x)
    df["Dport"] = df["Dport"].apply(
        lambda x: int(float(x)) if pd.notnull(x) else x)

    # Asegurar que sTos y dTos sean consistentes
    df["sTos"] = df["sTos"].replace("0", "0.0")
    df["dTos"] = df["dTos"].replace("0", "0.0")

    # Guardar el archivo normalizado en formato bitnetflow
    with open(output_path, "w") as f:
        # Escribir cabeceras
        headers = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
        f.write(headers)

        # Escribir cada fila de datos
        for index, row in df.iterrows():
            line = (
                f"{row['StartTime']},{row['Dur']},{row['Proto']},{
                    row['SrcAddr']},{row['Sport']},{row['Dir']},"
                f"{row['DstAddr']},{row['Dport']},{row['State']},{
                    row['sTos']},{row['dTos']},{row['TotPkts']},"
                f"{row['TotBytes']},{row['SrcBytes']},{row['Label']}\n"
            )
            f.write(line)

    print(f"Archivo bitnetflow normalizado creado exitosamente en: {
          output_path}")


# Ejemplo de uso de la función
file_path = "database/3.bitnetflow"
output_path = "database/0.bitnetflow"
normalize_bitnetflow(file_path, output_path)


# import pandas as pd

# # Función para normalizar un archivo de bitnetflow
# def normalize_bitnetflow(file_path, output_path):
#     df = pd.read_csv(file_path)

#     # Normalizar formatos de fecha y hora
#     df["StartTime"] = pd.to_datetime(
#         df["StartTime"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
#     )
#     df["StartTime"] = df["StartTime"].fillna(
#         pd.to_datetime(df["StartTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
#     )
#     df["StartTime"] = df["StartTime"].dt.strftime("%Y/%m/%d %H:%M:%S.%f")

#     # Convertir el Protocolo a minúsculas
#     df["Proto"] = df["Proto"].str.lower()

#     # Eliminar puntos decimales en puertos y asegurar valores enteros
#     df["Sport"] = df["Sport"].apply(lambda x: conv_port(x) if pd.notnull(x) else x)
#     df["Dport"] = df["Dport"].apply(lambda x: conv_port(x) if pd.notnull(x) else x)

#     # Asegurar que sTos y dTos sean consistentes
#     df["sTos"] = df["sTos"].replace("0", "0.0")
#     df["dTos"] = df["dTos"].replace("0", "0.0")

#     # Corregir duplicación en la etiqueta de Label
#     df["Label"] = df["Label"].str.replace("flow=flow=", "flow=")

#     # Convertir el Estado de `_` a un valor adecuado (por ejemplo, 'S_')
#     df["State"] = df["State"].replace("_", "S_")

#     # Guardar el archivo normalizado en formato bitnetflow
#     with open(output_path, "w") as f:
#         # Escribir cabeceras
#         headers = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
#         f.write(headers)

#         # Escribir cada fila de datos
#         for index, row in df.iterrows():
#             line = (
#                 f"{row['StartTime']},{row['Dur']},{row['Proto']},{row['SrcAddr']},{row['Sport']},{row['Dir']},"
#                 f"{row['DstAddr']},{row['Dport']},{row['State']},{row['sTos']},{row['dTos']},{row['TotPkts']},"
#                 f"{row['TotBytes']},{row['SrcBytes']},flow={row['Label']}\n"
#             )
#             f.write(line)

#     print(f"Archivo bitnetflow normalizado creado exitosamente en: {output_path}")

# # Ejemplo de uso de la función
# file_path = "database/3.bitnetflow"
# output_path = "database/0.bitnetflow"
# normalize_bitnetflow(file_path, output_path)
