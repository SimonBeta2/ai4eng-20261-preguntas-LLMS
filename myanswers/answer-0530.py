import pandas as pd
import numpy as np

# ── Función solución ──────────────────────────────────────────────
#def detectar_inestabilidad_fisiologica(df_vitales, ventana):
    #df = df_vitales.copy()
    #df["media_movil"] = df["frecuencia_cardiaca"].rolling(window=ventana).mean()
    #df["desviacion"]  = (df["frecuencia_cardiaca"] - df["media_movil"]).abs()
    #mascara = df["desviacion"] > (0.15 * df["media_movil"])
    #return df[mascara].dropna(subset=["desviacion"])


def detectar_inestabilidad_fisiologica(df_vitales, ventana):
    df = df_vitales.copy()
    df['media_movil'] = df['frecuencia_cardiaca'].rolling(window=ventana).mean()
    df['desviacion'] = (df['frecuencia_cardiaca'] - df['media_movil']).abs()
    df = df.dropna(subset=['media_movil'])
    return df[df['desviacion'] > (df['media_movil'] * 0.15)]

def generar_caso_de_uso_detectar_inestabilidad_fisiologica():
    n_puntos = 100
    base_hr = np.random.randint(60, 100, size=n_puntos).astype(float)
    base_hr[np.random.randint(0, 100, 5)] += 50 
    df = pd.DataFrame({'frecuencia_cardiaca': base_hr})
    v = 10
    
    entrada = {"df_vitales": df, "ventana": v}
    salida_esperada = detectar_inestabilidad_fisiologica(df, v)
    
    return entrada, salida_esperada

# ── Generador de casos de uso ─────────────────────────────────────
#def generar_caso_de_uso_detectar_inestabilidad_fisiologica():
    n_puntos = 100
    base_hr = np.random.randint(60, 100, size=n_puntos).astype(float)
    base_hr[np.random.randint(0, 100, 5)] += 50
    df = pd.DataFrame({'frecuencia_cardiaca': base_hr})
    v = 10

    input_data = {"df_vitales": df.copy(), "ventana": v}

    # Output esperado calculado de forma independiente
    df_out = df.copy()
    df_out["media_movil"] = df_out["frecuencia_cardiaca"].rolling(window=v).mean()
    df_out["desviacion"]  = (df_out["frecuencia_cardiaca"] - df_out["media_movil"]).abs()
    mascara = df_out["desviacion"] > (0.15 * df_out["media_movil"])
    output_data = df_out[mascara].dropna(subset=["desviacion"])

    return input_data, output_data


# ── Ejecución ─────────────────────────────────────────────────────
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_detectar_inestabilidad_fisiologica()

    print("=== INPUT ===")
    print(input_data["df_vitales"].to_string())
    print(input_data["ventana"])

    resultado = detectar_inestabilidad_fisiologica(
        **input_data
    )

    print(f"\nOutput esperado:\n{output_data.to_string()}")
    print(f"\nOutput obtenido:\n{resultado.to_string()}")

    ok = resultado.equals(output_data)
    print(f"\n¿Correcto? → {'✓ PASS' if ok else '✗ FAIL'}")