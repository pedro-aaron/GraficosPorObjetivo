# -*- coding: utf-8 -*-
"""data_loader.py
============================================================
Utilidades de carga y limpieza de datos para el taller.

- Lee ventas por país (archivos separados) y los concatena en una tabla única.
- Lee catálogos: products, manufacturers, geo.
- Estandariza tipos (Zip como texto), fechas y columnas derivadas.

Autor: Pedro Aarón Hernández
"""
from __future__ import annotations
import os
import re
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

COUNTRY_FILES = [
    "Australia.csv",
    "Canada.csv",
    "Germany.csv",
    "Japan.csv",
    "Mexico.csv",
    "Nigeria.csv",
    "USA.csv",
]


class DataLoader:
    """Clase responsable de cargar, validar y preparar los datos.

    Parámetros
    ----------
    data_dir : str
        Carpeta donde residen los CSV de entrada.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self._validate_paths()

    # -------------------------------
    # Validaciones y lectura de CSVs
    # -------------------------------
    def _validate_paths(self) -> None:
        missing = [
            fn
            for fn in COUNTRY_FILES + ["products.csv", "manufacturers.csv", "geo.csv"]
            if not os.path.exists(os.path.join(self.data_dir, fn))
        ]
        if missing:
            raise ValueError(f"Faltan archivos en {self.data_dir}: {missing}")

    # --------------------------------
    # Lectura y limpieza de datos
    # --------------------------------
    @staticmethod
    def _smart_read_csv(path, **kwargs):
        encodings = [None, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_err = None
        for enc in encodings:
            try:
                if enc is None:
                    return pd.read_csv(path, **kwargs)
                else:
                    return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError as e:
                last_err = e
                continue
        raise last_err

    def _read_country(self, filename: str) -> pd.DataFrame:
        """Lee un archivo de ventas de un país y añade/ajusta columnas estándar.

        - Asegura dtypes: Zip como string, Units como int, Revenue como float.
        - Parsea fecha y agrega columnas Year, Month, Quarter, YearMonth.
        - Deriva UnitPrice = Revenue / Units.
        - Estandariza Country a partir del archivo.
        """
        path = os.path.join(self.data_dir, filename)
        df = self._smart_read_csv(path, dtype={"Zip": "string"})
        # Limpia espacios en nombres pero preserva capitalización original
        df.columns = [c.strip() for c in df.columns]

        # Tipos
        if "Units" in df.columns:
            df["Units"] = pd.to_numeric(df["Units"], errors="coerce").astype("Int64")
        if "Revenue" in df.columns:
            df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

        # Fechas
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month
            df["Quarter"] = df["Date"].dt.quarter
            df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

        # Country a partir del nombre del archivo (por si hay inconsistencias en el CSV)
        country = os.path.splitext(os.path.basename(filename))[0]
        df["Country"] = country

        # Métricas derivadas
        if "Revenue" in df.columns and "Units" in df.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                df["UnitPrice"] = df["Revenue"].astype(float) / df["Units"].astype(
                    float
                )

        return df

    def load_sales(self) -> pd.DataFrame:
        frames = [self._read_country(fn) for fn in COUNTRY_FILES]
        sales = pd.concat(frames, ignore_index=True)
        # Ordena columnas para facilitar lectura
        preferred = [
            "ProductID",
            "Date",
            "Year",
            "Quarter",
            "Month",
            "YearMonth",
            "Zip",
            "Units",
            "Revenue",
            "UnitPrice",
            "Country",
        ]
        cols = [c for c in preferred if c in sales.columns] + [
            c for c in sales.columns if c not in preferred
        ]
        sales = sales[cols]
        return sales

    def load_products(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "products.csv")
        prod = self._smart_read_csv(path)
        prod.columns = [c.strip() for c in prod.columns]

        # Ordenar pot ProductID ASC
        prod.sort_values("ProductID", inplace=True)

        # Product está compuesto por dos campos: "<Producto name>|<Segment>"
        #  por ejemplo: "Abbas MA-01|All Season"
        # Creamos dos columnas separadas: ProductName y Segment
        if "Product" in prod.columns:
            prod[["ProductName", "Segment"]] = prod["Product"].str.split(
                r"\s*\|\s*", n=1, expand=True
            )
            prod.drop(columns=["Product"], inplace=True)
            prod = prod.rename(columns={"ProductName": "Product"})

        # Hacer un FillDown de Category
        if "Category" in prod.columns:
            prod["Category"] = prod["Category"].ffill()
            prod["Category"] = prod["Category"].str.strip()

        # Price: 'USD 412.13' -> 412.13
        def _usd_to_float(x):
            import re, math, pandas as pd, numpy as np

            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)):
                return float(x)
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(x))
            return float(m.group(1)) if m else np.nan

        if "Price" in prod.columns:
            prod["ProductPriceUsd"] = prod["Price"].apply(_usd_to_float)
            prod.drop(columns=["Price"], inplace=True)
        return prod

    def load_manufacturers(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "manufacturers.csv")
        man = self._smart_read_csv(path)
        man.columns = [c.strip() for c in man.columns]
        return man

    def load_geo(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "geo.csv")
        geo = self._smart_read_csv(path, dtype={"Zip": "string"})
        geo.columns = [c.strip() for c in geo.columns]
        # 🔧 Evitar colisión con ventas
        if "Country" in geo.columns:
            geo = geo.rename(columns={"Country": "GeoCountry"})
        return geo

    def build_fact_table(self) -> pd.DataFrame:
        sales = self.load_sales()
        prod = self.load_products()
        man = self.load_manufacturers()
        geo = self.load_geo()

        fact = (
            sales.merge(prod, how="left", on="ProductID")
            .merge(man, how="left", on="ManufacturerID")
            .merge(geo, how="left", on="Zip")
        )

        # 🔧 Garantiza columna 'Country'
        if "Country" not in fact.columns:
            for cand in ["Country_x", "Country_y", "Country_geo", "GeoCountry"]:
                if cand in fact.columns:
                    fact["Country"] = fact[cand]
                    break

        # 🔧 Limpia duplicados/sobrantes
        for col in ["Country_x", "Country_y", "Country_geo"]:
            if col in fact.columns:
                fact.drop(columns=[col], inplace=True)

        preferred = [
            "Date",
            "Year",
            "Quarter",
            "Month",
            "YearMonth",
            "Country",
            "Zip",
            "City",
            "State",
            "ProductID",
            "Product",
            "Category",
            "Segment",
            "ManufacturerID",
            "Manufacturer",
            "ProductPriceUsd",
            "Units",
            "UnitPrice",
            "Revenue",
        ]
        cols = [c for c in preferred if c in fact.columns] + [
            c for c in fact.columns if c not in preferred
        ]
        fact = fact[cols]
        return fact
