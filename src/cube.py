# -*- coding: utf-8 -*-
"""cube.py
============================================================
Definición de un "cubo" analítico basado en pandas (pivotes MultiIndex)
para responder preguntas de agregación.
"""
from __future__ import annotations
import os
from typing import Iterable, Optional
import pandas as pd


class SalesCube:
    """Construye y expone pivotes a partir de la tabla de hechos.

    Parámetros
    ----------
    fact : pandas.DataFrame
        Tabla de hechos proveniente de DataLoader.build_fact_table().
    out_dir : str, default None
        Carpeta donde se exportarán CSV de pivotes (opcional).
    """

    def __init__(self, fact: pd.DataFrame, out_dir: Optional[str] = None) -> None:
        self.fact = fact.copy()
        self.out_dir = out_dir

    # -----------------
    # Pivotes básicos
    # -----------------
    def revenue_by(
        self, index: Iterable[str], columns: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """Ingreso total por combinaciones de dimensiones."""
        tbl = pd.pivot_table(
            self.fact,
            values="Revenue",
            index=list(index),
            columns=list(columns) if columns else None,
            aggfunc="sum",
            fill_value=0.0,
            margins=False,
        )
        return tbl

    def units_by(
        self, index: Iterable[str], columns: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """Unidades totales por combinaciones de dimensiones."""
        tbl = pd.pivot_table(
            self.fact,
            values="Units",
            index=list(index),
            columns=list(columns) if columns else None,
            aggfunc="sum",
            fill_value=0,
            margins=False,
        )
        return tbl

    def top_n_products_by_revenue(self, n: int = 10) -> pd.DataFrame:
        """Top-N productos por ingreso."""
        g = (
            self.fact.groupby(["ProductID", "Product"], dropna=False)["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(n)
        )
        return g.reset_index(name="Revenue")

    def export(self, df: pd.DataFrame, filename: str) -> None:
        """Exporta un DataFrame a CSV en out_dir si está configurado."""
        if not self.out_dir:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        df.to_csv(os.path.join(self.out_dir, filename), index=True)
