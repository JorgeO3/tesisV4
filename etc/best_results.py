import pandas as pd

df = pd.read_csv("./results/gretel_70/11-16_12:54:10_TS.csv")
df = df[["user_attrs_mse", "user_attrs_r2", "params_layer_1"]]

x = 10

# Resultados por cada tamaño de capa
resultados_por_capa = {}

for capa in range(1, 25):
    # Filtrar por tamaño de capa
    df_filtrado = df[df["params_layer_1"] == capa]

    # Ordenar por MSE y tomar los primeros x registros
    mejores_resultados = df_filtrado.sort_values("user_attrs_r2", ascending=False).head(x)

    # Guardar los resultados
    resultados_por_capa[capa] = mejores_resultados

for capa, df_resultados in resultados_por_capa.items():
    print(f"Resultados para el tamaño de capa {capa}:")
    print(df_resultados.to_string(index=False))
    print("\n")
