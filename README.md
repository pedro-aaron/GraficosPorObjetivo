# Taller práctico: **Elección de gráficos según el objetivo**

Este proyecto en Python es un taller práctico para la asignatura **Elección de gráficos según el objetivo**.  
Trabaja con archivos CSV de ventas por país (Australia, Canada, Germany, Japan, Mexico, Nigeria, USA) y con catálogos (`products.csv`, `manufacturers.csv`, `geo.csv`).

El objetivo es:

1. Realizar **análisis exploratorio de datos (EDA)** por archivo y consolidado.
2. **Concatenar** los datos de ventas por país en una sola tabla de hechos.
3. Construir un **cubo de información** (tablas dinámicas/pivotes) para contestar preguntas de agregación.
4. Producir una **galería de visualizaciones** con _pandas_ y _matplotlib_ alineadas al objetivo comunicativo (comparación, tendencia, distribución, parte-todo, correlación, ranking y ubicación con proxies).

---

## Estructura del repo

```text
visualizacion/
├── data/
│   └── data.zip              # Archivos CSV comprimidos
├── outputs/                   # Se guardarán figuras y pivotes aquí
├── src/
│   ├── data_loader.py         # Carga y limpieza
│   ├── cube.py                # Lógica del cubo (pivotes MultiIndex)
│   ├── visuals.py             # Gráficos según el objetivo
│   └── workshop.py            # Script principal
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

---

## Preparación del entorno (Windows / macOS / Linux)

### 0. Instalar Python y pip (si no los tienes)

-   Instalar Python 3.8+ desde [python.org](https://www.python.org/downloads/).
-   `pip` viene incluido a partir de Python 3.4+.
-   Asegúrate de que `python` y `pip` estén en el PATH.

Verifica la instalación:

```bash
python --version
pip --version
```

Opcionalmente:

-   Usar [Anaconda](https://www.anaconda.com/products/distribution).
-   Editor recomendado: [VSCode](https://code.visualstudio.com/).

---

### 1. Crear entorno virtual

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 3. Descomprimir los datos

Los archivos `.csv` están dentro de `data/data.zip`.  
Debes descomprimirlo **antes de ejecutar** el taller:

```bash
# macOS / Linux
unzip data/data.zip -d data

# Windows (PowerShell)
Expand-Archive -Path data\data.zip -DestinationPath data
```

Esto creará archivos como:

```text
data/
   ├── Australia.csv
   ├── Canada.csv
   ├── Germany.csv
   ├── Japan.csv
   ├── Mexico.csv
   ├── Nigeria.csv
   ├── USA.csv
   ├── geo.csv
   ├── manufacturers.csv
   └── products.csv
```

---

### 4. Ejecutar el taller

```bash
python -m src.workshop --data-dir data --out-dir outputs
```

Parámetros opcionales:

-   `--countries "USA,Canada,Mexico"` → restringe análisis a ciertos países.
-   `--country "USA"` → fija país de enfoque para la galería individual.

El script guardará figuras en `outputs/fig_*.png` y exportará pivotes CSV en `outputs/`.

---

## Qué aprenderás (mini-tutorial integrado)

-   **Pandas**: lectura robusta de CSV, `astype`, `to_datetime`, `merge`, `groupby`, `pivot_table`, `resample`, `categorical`.
-   **Matplotlib**: figuras y ejes, `line`, `barh`, `hist`, `boxplot`, `scatter`, `imshow` (heatmaps), anotaciones, formato de ejes y leyendas.
-   **Diseño por objetivo**: elegir el gráfico según la meta comunicativa (comparación, tendencia, distribución, parte-todo, relación, ranking, ubicación-proxy).

El archivo `src/workshop.py` está **ampliamente comentado** para que sigas y adaptes cada paso.

---

## Notas sobre los datos

-   `Zip` se trata como **texto** (hay códigos alfanuméricos, p. ej. Canadá).
-   `Date` se convierte a `datetime64` y se derivan `Year`, `Month`, `Quarter`.
-   `products.csv` contiene `Price` como cadena (`"USD 412.13"`); el script lo parsea a numérico en USD.
-   Se calcula `UnitPrice = Revenue/Units` y se normaliza `Country` desde el nombre del archivo.

---

## Preguntas que el cubo permite responder

-   Ventas (unidades/ingresos) por **País × Año × Mes**.
-   Top-N **productos** o **categorías** por ingreso.
-   Participación **país → parte-todo** por año.
-   Tendencias por país y producto.
-   Heatmap **País × Mes** (estacionalidad).
-   Relación **precio vs. unidades** a nivel producto/orden.

---

## Licencia y uso en clase

Este material es educativo. Puedes modificar el código para tus prácticas y tareas.
