"""Evaluación del clasificador de riesgo de marketplace.

Reproduce las cifras reportadas en el paper a partir de
train/dataset/dataset_fb_marketplace.csv. Ejecutar desde la raíz del repo:

    python validation/evaluate.py

Diferencias respecto de train/main.py:
- La primera columna del CSV es el índice de pandas y no se usa como característica.
- Se eliminan las filas idénticas (la fuente reúne capturas diarias).
- La evaluación usa validación cruzada estratificada de cinco particiones, porque la clase
  de riesgo alto tiene 34 registros y una sola partición dejaría siete en prueba.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier

DATASET = Path("train/dataset/dataset_fb_marketplace.csv")
RESULTS = Path("validation/results")
SEED = 42
CLASES = {1: "riesgo_alto", 2: "riesgo_moderado", 3: "riesgo_bajo", 4: "sin_verificar"}


def nuevo_modelo():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        criterion="log_loss",
        class_weight="balanced",
        random_state=SEED,
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    crudo = pd.read_csv(DATASET, index_col=0)
    dataset = crudo.drop_duplicates()
    labels = dataset["etiqueta_riesgo"]
    features = dataset.drop(columns=["etiqueta_riesgo"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Validación cruzada: métricas por partición y predicciones agregadas
    scores = cross_validate(
        nuevo_modelo(), features, labels, cv=cv,
        scoring=["accuracy", "f1_macro", "recall_macro"],
    )
    pred = cross_val_predict(nuevo_modelo(), features, labels, cv=cv)
    reporte = classification_report(labels, pred, digits=4, output_dict=True)

    # Importancias sobre una partición estratificada 80/20
    X_train, _, y_train, _ = train_test_split(
        features, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    modelo = nuevo_modelo().fit(X_train, y_train)
    importancias = dict(
        sorted(
            zip(features.columns, modelo.feature_importances_.round(4).tolist()),
            key=lambda kv: -kv[1],
        )
    )

    # Diagnóstico: un árbol poco profundo recupera la regla de etiquetado
    arbol = cross_validate(
        DecisionTreeClassifier(max_depth=5, random_state=SEED),
        features, labels, cv=cv, scoring=["accuracy", "f1_macro"],
    )
    arbol_completo = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(features, labels)
    usadas = sorted({features.columns[i] for i in arbol_completo.tree_.feature if i >= 0})
    base = cross_validate(
        DummyClassifier(strategy="most_frequent"),
        features, labels, cv=cv, scoring=["accuracy", "f1_macro"],
    )

    resumen = {
        "registros": {"crudos": len(crudo), "duplicados": len(crudo) - len(dataset), "usados": len(dataset)},
        "distribucion": {CLASES[k]: int(v) for k, v in labels.value_counts().sort_index().items()},
        "validacion_cruzada": {
            m: {"media": round(scores[f"test_{m}"].mean(), 4), "desviacion": round(scores[f"test_{m}"].std(), 4)}
            for m in ["accuracy", "f1_macro", "recall_macro"]
        },
        "por_clase": {
            CLASES[int(k)]: {m: round(v[m], 4) for m in ["precision", "recall", "f1-score", "support"]}
            for k, v in reporte.items() if k.isdigit()
        },
        "agregado": {
            "accuracy": round(reporte["accuracy"], 4),
            "macro": {m: round(reporte["macro avg"][m], 4) for m in ["precision", "recall", "f1-score"]},
        },
        "matriz_confusion": {
            "orden": [CLASES[k] for k in sorted(CLASES)],
            "valores": confusion_matrix(labels, pred, labels=sorted(CLASES)).tolist(),
        },
        "importancias": importancias,
        "diagnostico_arbol_profundidad_5": {
            "accuracy": round(arbol["test_accuracy"].mean(), 4),
            "f1_macro": round(arbol["test_f1_macro"].mean(), 4),
            "hojas": int(arbol_completo.get_n_leaves()),
            "variables_usadas": usadas,
        },
        "linea_base_clase_mayoritaria": {
            "accuracy": round(base["test_accuracy"].mean(), 4),
            "f1_macro": round(base["test_f1_macro"].mean(), 4),
        },
    }

    (RESULTS / "metrics.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    (RESULTS / "classification_report.txt").write_text(
        classification_report(labels, pred, digits=4, target_names=[CLASES[k] for k in sorted(CLASES)])
    )
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
