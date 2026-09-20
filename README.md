# Classification Model Marketplace

Modelo de clasificación para estimar el nivel de riesgo de publicaciones de Facebook Marketplace a partir de atributos del producto y del vendedor.

## Estructura

```text
.
├── train/
│   ├── main.py
│   ├── dataset/
│   │   └── dataset_fb_marketplace.csv
│   └── models/
│       └── model_fb_marketplace.pkl
└── requirements.txt
```

## Requisitos

- Python 3.10 o superior recomendado.
- Entorno virtual de Python.
- El archivo `train/dataset/dataset_fb_marketplace.csv`.

Las dependencias del proyecto están declaradas en `requirements.txt`:

- `pandas`: lectura y manipulación del dataset.
- `scikit-learn`: partición de datos, entrenamiento y evaluación.
- `joblib`: serialización del modelo entrenado.

## Instalación

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows PowerShell, activa el entorno con:

```powershell
.venv\Scripts\Activate.ps1
```

## Dataset

El dataset debe ser un archivo CSV separado por comas y contener estas columnas:

| Columna | Descripción |
| --- | --- |
| `discount` | Descuento aplicado al producto. |
| `ratingAvg` | Promedio de calificación del vendedor. |
| `ratingCount` | Cantidad de calificaciones recibidas. |
| `shopFollowerCount` | Cantidad de seguidores de la tienda o vendedor. |
| `shopItemCount` | Cantidad de artículos publicados. |
| `isShopeeVerified` | Indicador binario de verificación del vendedor. |
| `sellerSeniorityInDays` | Antigüedad del vendedor en días. |
| `imagesCount` | Número de imágenes de la publicación. |
| `cobertura_evidencia` | Cobertura de evidencia disponible para la publicación. |
| `etiqueta_riesgo` | Variable objetivo que representa la clase de riesgo. |

`etiqueta_riesgo` es la variable objetivo; todas las demás columnas se utilizan como características de entrada. El dataset actual contiene las clases `1`, `2`, `3` y `4`.

## Entrenamiento

Ejecuta el script desde la raíz del repositorio, ya que utiliza rutas relativas:

```bash
python train/main.py
```

El proceso realiza los siguientes pasos:

1. Lee el dataset.
2. Separa las características de `etiqueta_riesgo`.
3. Divide los datos en entrenamiento y prueba usando 80% y 20%, respectivamente.
4. Entrena un `RandomForestClassifier` con 100 árboles, `max_depth=10`, criterio `log_loss` y semilla `42`.
5. Imprime un `classification_report` con las métricas por clase.
6. Guarda el modelo en `train/models/model_fb_marketplace.pkl`.

Si la carpeta de salida no existe, créala antes de ejecutar el entrenamiento:

```bash
mkdir -p train/models
```

## Salida del modelo

El archivo `train/models/model_fb_marketplace.pkl` contiene el clasificador serializado con `joblib`. Puede cargarse en otro script de inferencia así:

```python
import joblib

model = joblib.load("train/models/model_fb_marketplace.pkl")
predicciones = model.predict(nuevas_caracteristicas)
```

`nuevas_caracteristicas` debe conservar las mismas nueve columnas de entrada y el mismo orden usado durante el entrenamiento. No debe incluir `etiqueta_riesgo`.

## Evaluación y consideraciones

El script muestra precisión, recall, F1-score y soporte mediante `classification_report`. La distribución actual de clases es desigual, por lo que conviene revisar especialmente el recall y el F1-score de las clases minoritarias, y no únicamente la exactitud global.

El conjunto de prueba se obtiene con una división aleatoria fija (`random_state=42`), pero el entrenamiento no usa estratificación ni validación cruzada. Para una evaluación experimental más robusta, se recomienda incorporar validación cruzada estratificada y comparar las métricas con una línea base.

## Limitaciones actuales

- El entrenamiento y la evaluación están concentrados en `train/main.py`.
- No existe todavía un script de inferencia o una API para consumir el modelo.
- El modelo serializado depende de versiones compatibles de Python, scikit-learn y joblib.
- Las rutas del script son relativas a la raíz del proyecto.
- El dataset puede contener una fuerte desproporción entre clases.
