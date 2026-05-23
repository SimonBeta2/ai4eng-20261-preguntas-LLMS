import random
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def evaluar_clasificador_fraude(X, y, test_size, random_state):
    # 1. División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Escalado (fit solo sobre train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3. Entrenamiento
    clf = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    # 4. Predicción y métricas
    y_pred = clf.predict(X_test_scaled)

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
    }

def generar_caso_de_uso_evaluar_clasificador_fraude():
    """Genera un caso de prueba aleatorio para evaluar_clasificador_fraude.

    Construye una matriz X y un vector y de clasificación binaria
    con parámetros aleatorios, y calcula las métricas esperadas.

    Returns
    -------
    tuple
        (input_data, output_data) donde input_data es un dict con
        las claves 'X', 'y', 'test_size', 'random_state' y
        output_data es un dict con las métricas esperadas.
    """

    # 1. Configuración aleatoria de dimensiones
    n_samples = random.randint(100, 300)
    n_features = random.randint(3, 8)
    n_informative = random.randint(2, min(n_features, 5))
    n_redundant = random.randint(0, n_features - n_informative)

    # 2. Generar datos de clasificación binaria con sklearn
    generation_seed = random.randint(0, 9999)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=generation_seed,
    )

    # 3. Definir hiperparámetros aleatorios
    test_size = round(random.uniform(0.15, 0.4), 2)
    random_state = random.randint(0, 9999)

    # ---------------------------------------------------------
    # 4. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "test_size": test_size,
        "random_state": random_state,
    }

    # ---------------------------------------------------------
    # 5. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener
    #    evaluar_clasificador_fraude
    # ---------------------------------------------------------

    # A. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # B. Escalar características (fit solo sobre train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # C. Entrenar DecisionTreeClassifier con max_depth=5
    clf = DecisionTreeClassifier(
        max_depth=5, random_state=random_state,
    )
    clf.fit(X_train_scaled, y_train)

    # D. Predecir y calcular métricas
    y_pred = clf.predict(X_test_scaled)

    output_data = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    return input_data, output_data


test_input, expected_output = (
            generar_caso_de_uso_evaluar_clasificador_fraude()
        )
result = evaluar_clasificador_fraude(**test_input)

# --- Ejemplo de uso ---
if __name__ == "__main__":
    input_data, output_data = generar_caso_de_uso_evaluar_clasificador_fraude()
    print("=== INPUT ===")
    print(f"\nX:\n{input_data['X']}")
    print(f"\ny:\n{input_data['y']}")
    print(f"\ntest_size:\n{input_data['test_size']}")
    print(f"\nrandom_state:\n{input_data['random_state']}")
    print("\n=== OUTPUT ESPERADO ===")
    print(f"Output esperado:\n{output_data}\n Output obtenido:\n{evaluar_clasificador_fraude(**input_data)}")