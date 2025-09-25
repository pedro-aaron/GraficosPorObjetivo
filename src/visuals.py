# -*- coding: utf-8 -*-
"""
visuals.py
============================================================
Galería de visualizaciones con matplotlib / seaborn guiada por OBJETIVO:
- Comparación (barras)
    - País (revenue)
    - Categoría (revenue)
    - Fabricante (revenue)
    - Segmento (revenue)  [si existiese]
    - Producto (revenue)
- Tendencia (tiempo) (líneas)
    - País (revenue)
    - Categoría (revenue)
    - Fabricante (revenue)
    - Segmento (revenue)
    - Producto (revenue)
- Distribución (histograma, boxplot)
    ANUAL:
    - País (revenue/order)
    - Categoría (revenue/order)
    - Fabricante (revenue/order)
    - Segmento (revenue/order)
    - Producto (revenue/order)
- Parte-todo (100% apilado)
    ANUAL:
    - País (revenue)
    - Categoría (revenue)
    - Fabricante (revenue)
    - Segmento (revenue)
    - Producto (revenue)
- Relación (correlación) (scatter)
- Ranking (top-N) (barras horizontales)
    ANUAL:
    - Fabricante (revenue/order)
    - Segmento (revenue/order)
    - Producto (revenue/order)
"""
from __future__ import annotations
import os
from typing import Optional, Sequence, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick


# =========================
# Helpers de formato/escala
# =========================
def fmt_human(v: float) -> str:
    """Formatea ticks sin notación científica, con separador de miles."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if v == 0:
        return "0"
    if v >= 1:
        return f"{v:,.0f}"
    return f"{v:,.3f}".rstrip("0").rstrip(".")


def apply_log_no_sci_barh(ax: plt.Axes) -> None:
    """Eje X logarítmico para barras horizontales, SIN notación científica."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mtick.LogLocator(base=10))
    ax.xaxis.set_minor_locator(mtick.LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, pos: fmt_human(v)))


def apply_log_no_sci_bar(ax: plt.Axes) -> None:
    """Eje Y logarítmico para barras verticales, SIN notación científica."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mtick.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mtick.LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, pos: fmt_human(v)))


def iqr_clean(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Filtra outliers con criterio IQR (Tukey)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return s
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return s[(s >= lower) & (s <= upper)]


def annotate_bars(ax, fmt="{:,.0f}", inside=True):
    """
    Coloca etiquetas de valor en cada barra (bar y barh).
    - fmt: formato numérico (default: miles sin decimales).
    - inside: True → etiqueta dentro de la barra; False → fuera (al final).
    Notas:
      * Funciona con ejes en log (posiciona en coordenadas de datos).
      * Omite barras con valor <= 0 (no tiene sentido en log).
    """
    for p in ax.patches:
        if not isinstance(p, plt.Rectangle):
            continue
        w, h = p.get_width(), p.get_height()
        # inferir orientación
        is_barh = w >= h
        if is_barh:
            if w <= 0:
                continue
            y = p.get_y() + h / 2
            x = p.get_x() + (0.98 * w if inside else 1.02 * w)
            ax.text(
                x,
                y,
                fmt.format(w),
                va="center",
                ha="right" if inside else "left",
                fontsize=8,
                color="white" if inside else "black",
                clip_on=True,
            )
        else:
            if h <= 0:
                continue
            x = p.get_x() + w / 2
            y = p.get_y() + (0.98 * h if inside else 1.02 * h)
            ax.text(
                x,
                y,
                fmt.format(h),
                va="top" if inside else "bottom",
                ha="center",
                fontsize=8,
                color="white" if inside else "black",
                clip_on=True,
            )


# ======================
# Galería global (mundo)
# ======================
class VisualGallery:
    def __init__(self, fact: pd.DataFrame, out_dir: str) -> None:
        self.fact = fact.copy()
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # Herramientas internas -----------------------------------------------
    def _savefig(self, fig: plt.Figure, name: str) -> None:
        path = os.path.join(self.out_dir, f"fig_{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 1) Comparación: barras por país (global, escala log y sin notación científica)
    def compare_revenue_by_country(self, countries: Optional[List[str]] = None) -> str:
        df = self.fact.copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]
        g = (
            df.groupby("Country")["Revenue"]
            .sum()
            .sort_values(ascending=True)
            .astype(float)
            .replace(0, np.nan)
            .dropna()
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        g.plot(kind="barh", ax=ax)
        apply_log_no_sci_barh(ax)
        annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS
        ax.set_title("Comparación de ingresos por país (barras, escala log)")
        ax.set_xlabel("Ingresos (USD, log)")
        outfile = "comparacion_ingresos_pais"
        self._savefig(fig, outfile)
        return outfile

    # 2) Tendencia: líneas por país (mensual)
    def trend_monthly_by_country(self, countries: Optional[List[str]] = None) -> str:
        df = self.fact.dropna(subset=["Date"]).copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]
        monthly = (
            df.set_index("Date")
            .groupby("Country")["Revenue"]
            .resample("MS")
            .sum()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        for country, sub in monthly.groupby("Country"):
            ax.plot(sub["Date"], sub["Revenue"], label=country, linewidth=1.5)
        ax.set_title("Tendencia mensual de ingresos por país (líneas)")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ingresos (USD)")
        ax.legend(ncol=3, fontsize=8)
        outfile = "tendencia_mensual_pais"
        self._savefig(fig, outfile)
        return outfile

    # 3) Distribución: hist y boxplot por país (con opción de limpiar outliers)
    def distribution_revenue_per_order(
        self,
        countries: Optional[List[str]] = None,
        clean: bool = False,
        clean_method: str = "iqr",
    ) -> str:
        df = self.fact.copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]

        # Histograma
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for country, sub in df.groupby("Country"):
            s = sub["Revenue"].dropna()
            if clean and clean_method == "iqr":
                s = iqr_clean(s)
            ax.hist(s, bins=30, alpha=0.5, label=country)
        ax.set_title("Distribución del ingreso por orden (histogramas)")
        ax.set_xlabel("Ingreso por orden (USD)")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=8, ncol=3)
        outfile = "distribucion_hist_ingreso_orden"
        self._savefig(fig, outfile)

        # Boxplot (matplotlib)
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        data = []
        labels = []
        for country, sub in df.groupby("Country"):
            s = sub["Revenue"].dropna()
            if clean and clean_method == "iqr":
                s = iqr_clean(s)
            if not s.empty:
                data.append(s.values)
                labels.append(country)
        if data:
            ax2.boxplot(data, vert=True, labels=labels)
        ax2.set_title("Distribución del ingreso por orden (boxplot)")
        ax2.set_xlabel("País")
        ax2.set_ylabel("Ingreso por orden (USD)")
        fig2.suptitle("")
        outfile2 = "distribucion_box_ingreso_orden"
        self._savefig(fig2, outfile2)

        # Boxplot (seaborn + strip para densidad)
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        plot_df = []
        for country, sub in df.groupby("Country"):
            s = sub["Revenue"].dropna()
            if clean and clean_method == "iqr":
                s = iqr_clean(s)
            if not s.empty:
                plot_df.append(pd.DataFrame({"Country": country, "Revenue": s}))
        plot_df = pd.concat(plot_df, ignore_index=True) if plot_df else pd.DataFrame()
        if not plot_df.empty:
            sns.boxplot(x="Country", y="Revenue", data=plot_df, ax=ax3)
            sns.stripplot(
                x="Country",
                y="Revenue",
                data=plot_df,
                ax=ax3,
                color="black",
                size=3,
                alpha=0.25,
                jitter=True,
            )
        ax3.set_title("Distribución del ingreso por orden (boxplot)")
        ax3.set_xlabel("País")
        ax3.set_ylabel("Ingreso por orden (USD)")
        fig3.suptitle("")
        outfile3 = "distribucion_box_ingreso_orden_seaborn"
        self._savefig(fig3, outfile3)

        return outfile2

    # 4) Parte-todo: barras 100% apiladas por año (share por país)
    def part_to_whole_country_share_by_year(
        self, countries: Optional[List[str]] = None
    ) -> str:
        df = self.fact.copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]
        if "Year" not in df.columns:
            return ""
        pt = df.pivot_table(
            values="Revenue",
            index="Year",
            columns="Country",
            aggfunc="sum",
            fill_value=0.0,
        )
        pt_pct = pt.div(pt.sum(axis=1), axis=0) * 100.0
        if pt_pct.empty:
            return ""
        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(pt_pct))
        x = np.arange(len(pt_pct.index))
        for col in pt_pct.columns:
            ax.bar(x, pt_pct[col].values, bottom=bottom, label=col)
            bottom += pt_pct[col].values
        ax.set_xticks(x, pt_pct.index.astype(int).astype(str), rotation=0)
        ax.set_title("Participación por país dentro del total anual (100% apilado)")
        ax.set_ylabel("Porcentaje del total anual (%)")
        ax.legend(ncol=3, fontsize=8)
        outfile = "parte_todo_share_pais_anual"
        self._savefig(fig, outfile)
        return outfile

    # 5) Relación: scatter precio vs unidades
    def relation_price_vs_units(self, countries: Optional[List[str]] = None) -> str:
        df = self.fact.copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)].dropna(subset=["UnitPrice", "Units"])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.scatter(df["UnitPrice"], df["Units"], alpha=0.6, s=20)
        ax.set_title("Relación precio unitario vs unidades (scatter)")
        ax.set_xlabel("Precio unitario (USD)")
        ax.set_ylabel("Unidades por orden")
        outfile = "relacion_precio_vs_unidades"
        self._savefig(fig, outfile)
        return outfile

    # 6) Ranking: top productos por ingreso (global)
    def ranking_top_products(
        self, top_n: int = 10, countries: Optional[List[str]] = None
    ) -> str:
        df = self.fact.copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]
        g = (
            df.groupby(["ProductID", "Product"], dropna=False)["Revenue"]
            .sum()
            .sort_values(ascending=True)
            .tail(top_n)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        g.plot(kind="barh", ax=ax)
        annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS

        ax.set_title(f"Top {top_n} productos por ingreso (ranking)")
        ax.set_xlabel("Ingresos (USD)")
        outfile = "ranking_top_productos"
        self._savefig(fig, outfile)
        return outfile

    # 7) Ubicación (proxy): barras por Estado (si disponible)
    def location_proxy_by_state(self, countries: Optional[List[str]] = None) -> str:
        df = self.fact.copy()
        if "State" not in df.columns:
            return ""
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]
        g = df.groupby("State")["Revenue"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        g.plot(kind="barh", ax=ax)
        annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS

        ax.set_title("Ingresos por Estado (proxy de ubicación)")
        ax.set_xlabel("Ingresos (USD)")
        outfile = "ubicacion_proxy_estado"
        self._savefig(fig, outfile)
        return outfile

    # 8) Heatmap País × Mes (estacionalidad)
    def heatmap_country_by_month(self, countries: Optional[List[str]] = None) -> str:
        df = self.fact.copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]
        if "Month" not in df.columns:
            return ""
        pt = df.pivot_table(
            values="Revenue",
            index="Country",
            columns="Month",
            aggfunc="sum",
            fill_value=0.0,
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(pt.values, aspect="auto")
        ax.set_yticks(np.arange(len(pt.index)), pt.index)
        ax.set_xticks(np.arange(12), [str(i) for i in range(1, 13)])
        ax.set_title("Heatmap de ingresos: País × Mes")
        ax.set_xlabel("Mes")
        ax.set_ylabel("País")
        fig.colorbar(im, ax=ax, shrink=0.7)
        outfile = "heatmap_pais_mes"
        self._savefig(fig, outfile)
        return outfile


# ===================================
# Galería enfocada en un solo país
# ===================================
class CountryVisualGallery:
    """
    Galería de visualizaciones para un país específico.
    Objetivos de diseño:
    - Comparación: ¿Qué categorías o fabricantes venden más en el país?
    - Tendencia (tiempo): ¿Cómo evoluciona el ingreso mensual en el país?
    - Distribución: ¿Cómo se distribuye el ingreso por orden en el país?
    - Parte-todo: ¿Cómo se reparte el total anual del país entre categorías?
    - Relación (correlación): ¿Existe relación entre precio unitario y unidades?
    - Ranking: Top productos por ingreso en el país.
    - Ubicación (proxy): Ingresos por Región/Estado dentro del país (si aplica).
    """

    def __init__(self, fact: pd.DataFrame, country: str, out_dir: str) -> None:
        self.country = country
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.fact = fact.query("Country == @country").copy()
        if "Date" in self.fact.columns:
            self.fact["Date"] = pd.to_datetime(self.fact["Date"], errors="coerce")

    def _savefig(self, fig: plt.Figure, name: str) -> str:
        path = os.path.join(self.out_dir, f"fig_{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return name

    # 1) Comparación: categoría vs ingresos (o fabricante si no hay categoría)
    def compare_category_revenue(self) -> str:
        if "Category" in self.fact.columns:
            group = self.fact.groupby("Category", dropna=False)["Revenue"].sum()
            title = f"Comparación de ingresos por categoría — {self.country}"
            fname = f"{self.country.lower()}_comparacion_categoria"
        else:
            group = self.fact.groupby("Manufacturer", dropna=False)["Revenue"].sum()
            title = f"Comparación de ingresos por fabricante — {self.country}"
            fname = f"{self.country.lower()}_comparacion_fabricante"

        g = group.sort_values(ascending=True).astype(float).replace(0, np.nan).dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        g.plot(kind="barh", ax=ax)
        apply_log_no_sci_barh(ax)
        ax.set_title(title)
        annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS

        ax.set_xlabel("Ingresos (USD, log)")
        return self._savefig(fig, fname)

    # 2) Tendencia: ingreso mensual (línea)
    def trend_monthly_total(self) -> str:
        df = self.fact.dropna(subset=["Date"]).copy()
        if df.empty:
            return ""
        monthly = df.set_index("Date")["Revenue"].resample("MS").sum().reset_index()
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(monthly["Date"], monthly["Revenue"], linewidth=1.8)
        ax.set_title(f"Tendencia mensual de ingresos — {self.country}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ingresos (USD)")
        return self._savefig(fig, f"{self.country.lower()}_tendencia_mensual")

    # 3) Distribución: ingreso por orden (hist + box con opción de limpieza IQR)
    def distribution_order_revenue(self, clean: bool = False) -> str:
        if "Revenue" not in self.fact.columns or self.fact["Revenue"].dropna().empty:
            return ""
        s = self.fact["Revenue"].dropna()
        if clean:
            s = iqr_clean(s)

        # Histograma
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(s, bins=30, alpha=0.8)
        ax.set_title(f"Distribución de ingreso por orden (hist) — {self.country}")
        ax.set_xlabel("Ingreso por orden (USD)")
        ax.set_ylabel("Frecuencia")
        name_hist = self._savefig(fig, f"{self.country.lower()}_dist_hist_revenue")

        # Boxplot
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.boxplot(s, vert=True, labels=[self.country])
        ax2.set_title(f"Distribución de ingreso por orden (box) — {self.country}")
        ax2.set_ylabel("Ingreso por orden (USD)")
        name_box = self._savefig(fig2, f"{self.country.lower()}_dist_box_revenue")

        # Seaborn (opcional)
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            x=pd.Series([self.country] * len(s), name="Country"),
            y=s.rename("Revenue"),
            ax=ax3,
        )
        sns.stripplot(
            x=pd.Series([self.country] * len(s), name="Country"),
            y=s.rename("Revenue"),
            color="black",
            alpha=0.25,
            size=3,
            ax=ax3,
            jitter=True,
        )
        ax3.set_title("Distribución de ingreso por orden (boxplot)")
        ax3.set_xlabel("País")
        ax3.set_ylabel("Ingreso por orden (USD)")
        fig3.suptitle("")
        _ = self._savefig(fig3, f"{self.country.lower()}_dist_box_revenue_seaborn")

        return f"{name_hist};{name_box}"

    # 4) Parte-todo: % del total anual por categoría (o fabricante)
    def part_to_whole_category_share_by_year(self) -> str:
        if "Year" not in self.fact.columns or "Revenue" not in self.fact.columns:
            return ""
        if "Category" in self.fact.columns:
            pt = self.fact.pivot_table(
                values="Revenue",
                index="Year",
                columns="Category",
                aggfunc="sum",
                fill_value=0.0,
            )
            label = "Categoría"
            base = "categoria"
        else:
            pt = self.fact.pivot_table(
                values="Revenue",
                index="Year",
                columns="Manufacturer",
                aggfunc="sum",
                fill_value=0.0,
            )
            label = "Fabricante"
            base = "fabricante"
        if pt.empty:
            return ""
        pt_pct = pt.div(pt.sum(axis=1), axis=0) * 100.0

        fig, ax = plt.subplots(figsize=(9, 5))
        bottom = np.zeros(len(pt_pct))
        x = np.arange(len(pt_pct.index))
        for col in pt_pct.columns:
            ax.bar(x, pt_pct[col].values, bottom=bottom, label=str(col))
            bottom += pt_pct[col].values
        ax.set_xticks(x, pt_pct.index.astype(int).astype(str))
        ax.set_title(f"Participación (% del total anual) por {label} — {self.country}")
        ax.set_ylabel("Porcentaje del total anual (%)")
        ax.legend(ncol=3, fontsize=8)
        return self._savefig(
            fig, f"{self.country.lower()}_parte_todo_share_{base}_anual"
        )

    # 5) Relación: scatter precio unitario vs unidades
    def relation_price_vs_units(self) -> str:
        needed = {"UnitPrice", "Units"}
        if not needed.issubset(self.fact.columns):
            return ""
        df = self.fact.dropna(subset=["UnitPrice", "Units"])
        if df.empty:
            return ""
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.scatter(df["UnitPrice"], df["Units"], alpha=0.6, s=20)
        ax.set_title(f"Relación precio unitario vs unidades — {self.country}")
        ax.set_xlabel("Precio unitario (USD)")
        ax.set_ylabel("Unidades por orden")
        return self._savefig(fig, f"{self.country.lower()}_relacion_precio_unidades")

    # 6) Ranking: Top-N productos por ingreso (barras horizontales)
    def ranking_top_products(self, top_n: int = 10) -> str:
        if "Product" not in self.fact.columns or "Revenue" not in self.fact.columns:
            return ""
        g = (
            self.fact.groupby(["ProductID", "Product"], dropna=False)["Revenue"]
            .sum()
            .sort_values(ascending=True)
            .tail(top_n)
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        g.plot(kind="barh", ax=ax)
        annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS

        ax.set_title(f"Top {top_n} productos por ingreso — {self.country}")
        ax.set_xlabel("Ingresos (USD)")
        return self._savefig(fig, f"{self.country.lower()}_ranking_top_productos")

    # 7) Ubicación (proxy): Región/Estado/City (en ese orden de preferencia)
    def location_proxy(self) -> str:
        level = next(
            (c for c in ["Region", "State", "City"] if c in self.fact.columns), None
        )
        if level is None:
            return ""
        g = self.fact.groupby(level)["Revenue"].sum().sort_values(ascending=True)
        if g.empty:
            return ""
        fig, ax = plt.subplots(figsize=(8, 5))
        g.plot(kind="barh", ax=ax)
        annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS

        ax.set_title(f"Ingresos por {level} — {self.country}")
        ax.set_xlabel("Ingresos (USD)")
        return self._savefig(
            fig, f"{self.country.lower()}_ubicacion_por_{level.lower()}"
        )

    # 8) Heatmap Mes × Categoría (estacionalidad)
    def heatmap_month_by_category(self) -> str:
        if (
            "Month" not in self.fact.columns
            or "Revenue" not in self.fact.columns
            or "Category" not in self.fact.columns
        ):
            return ""
        pt = self.fact.pivot_table(
            values="Revenue",
            index="Category",
            columns="Month",
            aggfunc="sum",
            fill_value=0.0,
        )
        if pt.empty:
            return ""
        fig, ax = plt.subplots(figsize=(8, 4.8))
        im = ax.imshow(pt.values, aspect="auto")
        ax.set_yticks(np.arange(len(pt.index)), pt.index)
        ax.set_xticks(np.arange(12), [str(i) for i in range(1, 13)])
        ax.set_title(f"Heatmap de ingresos (Mes × Categoría) — {self.country}")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Categoría")
        fig.colorbar(im, ax=ax, shrink=0.7)
        return self._savefig(fig, f"{self.country.lower()}_heatmap_mes_categoria")


# ===========================================================
# Galería “custom” con comparaciones anuales por país/año
# (ajustada a escala log y sin notación científica en ejes)
# ===========================================================
class VisualGalleryCustom:
    """
    Galería de visualizaciones con matplotlib / seaborn guiada por OBJETIVO:
    - Comparación (barras)
        - País (revenue): global + por año
    - Tendencia (tiempo) (líneas)
        - País (revenue)
    - Distribución (histograma, boxplot)
    - Parte-todo (100% apilado)
    - Relación (correlación) (scatter)
    - Ranking (top-N) (barras horizontales)
    """

    def __init__(self, fact: pd.DataFrame, out_dir: str) -> None:
        self.fact = fact.copy()
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    # Herramientas internas -----------------------------------------------
    def _savefig(self, fig: plt.Figure, name: str) -> str:
        path = os.path.join(self.out_dir, f"fig_{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # 1) Comparación: barras por país (global + por país + por año), TODOS en log sin 10^x
    def compare_revenue_by_country(
        self, countries: Optional[List[str]] = None, years: Optional[List[int]] = None
    ) -> str:
        import numpy as np

        # Base
        df = self.fact.copy()
        if countries is None:
            countries = df["Country"].dropna().unique().tolist()
        df = df[df["Country"].isin(countries)]

        if years is not None and "Year" in df.columns:
            df = df[df["Year"].isin(years)]
        else:
            years = (
                sorted(df["Year"].dropna().unique().tolist())
                if "Year" in df.columns
                else []
            )

        # --- Global por país (barh, log-X) ---
        g = (
            df.groupby("Country")["Revenue"]
            .sum()
            .sort_values(ascending=True)
            .astype(float)
            .replace(0, np.nan)
            .dropna()
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        g.plot(kind="barh", ax=ax)
        apply_log_no_sci_barh(ax)
        annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS

        ax.set_title("Comparación de ingresos por país (barras, escala log)")
        ax.set_xlabel("Ingresos (USD, log)")
        outfile = "comparacion_ingresos_pais"
        self._savefig(fig, outfile)

        # --- Por país: barras por AÑO (bar, log-Y) ---
        if "Year" in df.columns and years:
            for country in countries:
                dfc = df[df["Country"] == country]
                if dfc.empty:
                    continue
                g2 = (
                    dfc.groupby("Year")["Revenue"]
                    .sum()
                    .sort_values(ascending=True)
                    .astype(float)
                    .replace(0, np.nan)
                    .dropna()
                )
                if g2.empty:
                    continue
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                g2.plot(kind="bar", ax=ax2)
                apply_log_no_sci_bar(ax2)
                ax2.set_title(f"Ingresos anuales en {country} (barras, escala log)")
                ax2.set_xlabel("Año")
                ax2.set_ylabel("Ingresos (USD, log)")
                outfile2 = (
                    f"comparacion_ingresos_pais_{country.lower().replace(' ','_')}"
                )
                self._savefig(fig2, outfile2)

        # --- Por año: barras por PAÍS (barh, log-X) ---
        if "Year" in df.columns and years:
            for year in years:
                dfy = df[df["Year"] == year]
                if dfy.empty:
                    continue
                g3 = (
                    dfy.groupby("Country")["Revenue"]
                    .sum()
                    .sort_values(ascending=True)
                    .astype(float)
                    .replace(0, np.nan)
                    .dropna()
                )
                if g3.empty:
                    continue
                fig3, ax3 = plt.subplots(figsize=(8, 4.5))
                g3.plot(kind="barh", ax=ax3)
                apply_log_no_sci_barh(ax3)
                annotate_bars(ax, fmt="{:,.0f}", inside=True)  # <<--- ETIQUETAS

                ax3.set_title(f"Ingresos por país en {year} (barras, escala log)")
                ax3.set_xlabel("Ingresos (USD, log)")
                outfile3 = f"comparacion_ingresos_anual_{year}"
                self._savefig(fig3, outfile3)

        return outfile
