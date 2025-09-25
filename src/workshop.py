# -*- coding: utf-8 -*-
"""workshop.py (script principal)
============================================================
Este script guía un taller práctico de *pandas* + *matplotlib* para:
    1) EDA por archivo y consolidado.
    2) Construcción de tabla de hechos (ventas) a partir de archivos por país.
    3) Creación de un "cubo" (pivotes) para responder preguntas de agregación.
    4) Generación de una galería de visualizaciones **según el objetivo**.

Cómo ejecutar:
    python -m src.workshop --data-dir data --out-dir outputs \
        --countries "USA,Canada,Mexico,Australia,Germany,Japan,Nigeria" \
        --country "USA"

Nota: El script está **comentado** para funcionar como tutorial.
"""
from __future__ import annotations
import argparse
import os
from typing import List, Optional
import pandas as pd

from .data_loader import DataLoader
from .cube import SalesCube
from .visuals import VisualGallery, CountryVisualGallery, VisualGalleryCustom


# ------------------------------------------------------------------
# Utilidad: imprime una cabecera para separar secciones
# ------------------------------------------------------------------
def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)


# ------------------------------------------------------------------
# EDA breve por archivo (compatible con distintas versiones de pandas)
# ------------------------------------------------------------------
def quick_eda(df: pd.DataFrame, name: str, out_dir: str) -> None:
    """Genera un resumen de EDA por DataFrame y lo guarda como CSV.

    - Head (primeras filas)
    - Describe numérico/categórico (con fallback para pandas antiguos)
    - Conteo de nulos
    """
    os.makedirs(out_dir, exist_ok=True)
    df.head(10).to_csv(os.path.join(out_dir, f"eda_head_{name}.csv"), index=False)

    # Describe con fallback (por si falta datetime_is_numeric en pandas viejos)
    describe_path = os.path.join(out_dir, f"eda_describe_{name}.csv")
    try:
        desc = df.describe(include="all", datetime_is_numeric=True)  # pandas nuevos
        desc.to_csv(describe_path)
    except TypeError:
        df.describe(include="all").to_csv(describe_path)
        dt = df.select_dtypes(include=["datetime64[ns]", "datetimetz"])
        if not dt.empty:
            dt.agg(["min", "max"]).to_csv(
                os.path.join(out_dir, f"eda_describe_datetimes_{name}.csv")
            )

    df.isna().sum().to_csv(os.path.join(out_dir, f"eda_nulls_{name}.csv"))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(
    data_dir: str,
    out_dir: str,
    countries: Optional[List[str]] = None,
    country_focus: Optional[str] = None,
) -> None:
    header("1) CARGA DE DATOS")
    print(f"Pandas version: {pd.__version__}")
    loader = DataLoader(data_dir)

    # Carga por país para EDA rápida
    country_files = [
        "Australia.csv",
        "Canada.csv",
        "Germany.csv",
        "Japan.csv",
        "Mexico.csv",
        "Nigeria.csv",
        "USA.csv",
    ]
    for fn in country_files:
        # Uso intencional del método interno para fines pedagógicos del taller
        df_country = loader._read_country(fn)
        print(f"Archivo {fn}: filas={len(df_country):,}")
        quick_eda(df_country, name=os.path.splitext(fn)[0].lower(), out_dir=out_dir)

    header("2) TABLA DE HECHOS (CONSOLIDADA)")
    fact = loader.build_fact_table()
    print("Columnas de la tabla de hechos:", list(fact.columns))
    print("Filas totales:", len(fact))
    # Exporta una muestra para inspección
    fact.head(20).to_csv(os.path.join(out_dir, "fact_sample.csv"), index=False)

    header("3) CUBO (TABLAS DINÁMICAS)")
    cube = SalesCube(fact, out_dir=out_dir)

    # Ejemplos de pivotes (agregaciones frecuentes)
    rev_country_year = cube.revenue_by(index=["Country", "Year"])
    cube.export(rev_country_year, "cubo_revenue_country_year.csv")

    units_country_month = cube.units_by(index=["Country"], columns=["Month"])
    cube.export(units_country_month, "cubo_units_country_month.csv")

    top_products = cube.top_n_products_by_revenue(n=15)
    top_products.to_csv(
        os.path.join(out_dir, "top15_productos_ingreso.csv"), index=False
    )

    # ---------------------------
    # 4) VISUALIZACIONES GLOBALES
    # ---------------------------
    header("4) VISUALIZACIONES SEGÚN EL OBJETIVO (GLOBAL)")
    gallery = VisualGallery(fact, out_dir=out_dir)

    # Si no se pasan países, usa todos los del dataset
    if not countries:
        countries = (
            fact["Country"].dropna().unique().tolist()
            if "Country" in fact.columns
            else []
        )
    print("Países seleccionados (global):", countries)

    figs = []
    figs.append(
        gallery.compare_revenue_by_country(countries=countries)
    )  # Comparación (barh log + etiquetas)
    figs.append(
        gallery.trend_monthly_by_country(countries=countries)
    )  # Tendencia (líneas)
    figs.append(
        gallery.distribution_revenue_per_order(countries=countries, clean=False)
    )  # Distribución
    figs.append(
        gallery.part_to_whole_country_share_by_year(countries=countries)
    )  # Parte-todo (100%)
    figs.append(
        gallery.relation_price_vs_units(countries=countries)
    )  # Relación (scatter)
    figs.append(
        gallery.ranking_top_products(top_n=10, countries=countries)
    )  # Ranking (barh)
    figs.append(
        gallery.location_proxy_by_state(countries=countries)
    )  # Ubicación (proxy)
    figs.append(
        gallery.heatmap_country_by_month(countries=countries)
    )  # Heatmap País×Mes

    print("Figuras globales generadas:", [f"fig_{name}.png" for name in figs if name])

    # --------------------------------
    # 4b) GALERÍA POR PAÍS (FOCUS)
    # --------------------------------
    header("4b) GALERÍA POR PAÍS (FOCUS)")
    # Si no se pasó un país de enfoque, toma el primero disponible
    if not country_focus:
        country_focus = countries[0] if countries else "USA"
    print("País en enfoque:", country_focus)

    country_gallery = CountryVisualGallery(fact, country=country_focus, out_dir=out_dir)
    cn_figs = []
    cn_figs.append(
        country_gallery.compare_category_revenue()
    )  # Comparación (cat/fabricante, barh log + etiquetas)
    cn_figs.append(country_gallery.trend_monthly_total())  # Tendencia
    cn_figs.append(
        country_gallery.distribution_order_revenue(clean=False)
    )  # Distribución (opción clean=True para IQR)
    cn_figs.append(
        country_gallery.part_to_whole_category_share_by_year()
    )  # Parte-todo (100%)
    cn_figs.append(country_gallery.relation_price_vs_units())  # Relación
    cn_figs.append(
        country_gallery.ranking_top_products(top_n=12)
    )  # Ranking (barh + etiquetas)
    cn_figs.append(country_gallery.location_proxy())  # Ubicación (proxy)
    cn_figs.append(country_gallery.heatmap_month_by_category())  # Heatmap Mes×Cat

    print(
        "Figuras país (focus) generadas:",
        [f"fig_{name}.png" for name in cn_figs if name],
    )

    # --------------------------------
    # 4c) GALERÍA CUSTOM (COMPARACIONES ANUALES)
    # --------------------------------
    header("4c) GALERÍA CUSTOM (comparaciones anuales)")
    custom_gallery = VisualGalleryCustom(fact, out_dir=out_dir)
    custom_gallery.compare_revenue_by_country(countries=countries)  # genera:
    #   - fig_comparacion_ingresos_pais.png      (global barh, log, sin 10^x, con etiquetas)
    #   - fig_comparacion_ingresos_pais_<pais>.png (por país, bar, log-Y, sin 10^x, con etiquetas)
    #   - fig_comparacion_ingresos_anual_<año>.png (por año, barh, log-X, sin 10^x, con etiquetas)

    print(f"\nListo. Revisa la carpeta de salida: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Taller práctico: Elección de gráficos según el objetivo"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Carpeta con CSV de entrada"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Carpeta de salida para figuras y pivotes",
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="",  # si queda vacío, se usarán todos los países detectados
        help='Lista separada por comas con países a incluir (ej: "USA,Canada,Mexico")',
    )
    parser.add_argument(
        "--country",
        type=str,
        default="",  # si queda vacío, se toma el primero de --countries o "USA"
        help='País de enfoque para la galería por país (ej: "USA")',
    )
    args = parser.parse_args()

    countries_arg = (
        [c.strip() for c in args.countries.split(",") if c.strip()]
        if args.countries
        else None
    )
    country_focus_arg = args.country.strip() if args.country else None

    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        countries=countries_arg,
        country_focus=country_focus_arg,
    )
