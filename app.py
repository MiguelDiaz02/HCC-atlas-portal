"""
HCC Single-Cell Atlas — Interactive Portal
===========================================
Dash application for exploring the integrated HCC / Healthy liver
single-cell RNA-seq atlas described in:

  Díaz-Campos & Hernández-Lemus (2025).
  "Single-Cell Transcriptomic Profiling Reveals Immunometabolic
   Reprogramming and Cell-Cell Communication in the Tumor
   Microenvironment of Human Hepatocellular Carcinoma."
  International Journal of Molecular Sciences (IJMS).

Data sources (public GEO/CellxGene datasets):
  GSE242889 · GSE228195 · GSE151530 · GSE189903 · CZ CellxGene

Run locally:
    python app.py
Deploy (Render.com):
    gunicorn app:server --bind 0.0.0.0:$PORT
"""

import json
import os

from flask import Flask
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, ctx
from dash.exceptions import PreventUpdate

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

CELL_TYPE_COLORS = {
    # Matched to manuscript Fig3C palette
    "Hepatocytes":       "#E41A1C",   # bright red
    "TAMs":              "#984EA3",   # purple
    "T cells":           "#1B7837",   # dark green
    "Malignant_cells":   "#00BCD4",   # cyan
    "TECs":              "#B8E186",   # lime green
    "B cells":           "#E91E8C",   # magenta
    "Plasma cells":      "#F9A8D4",   # light pink
    "NK cells":          "#FF7F00",   # orange
    "Neutrophils":       "#A65628",   # brown
    "Cholangiocytes":    "#8B8000",   # dark yellow/olive
    "Endothelial cells": "#3182BD",   # medium blue
    "Macrophages":       "#6BAED6",   # light blue
    "NK-TR-CD160":       "#41B6C4",   # teal
    "Monocytes":         "#D6B656",   # tan/gold
    "cDCs":              "#08519C",   # dark blue
    "CAF":               "#9E9AC8",   # lavender
    "Fibroblasts":       "#54278F",   # dark purple
    "Basophils":         "#7B2D00",   # dark maroon
    "pDCs":              "#BDBDBD",   # light gray
}

CONDITION_COLORS = {
    "Healthy Donors": "#2196F3",
    "HCC Diseased":   "#F44336",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (once at startup)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading portal data…")
metadata     = pd.read_parquet(os.path.join(DATA_DIR, "metadata.parquet"))
gene_summary = pd.read_parquet(os.path.join(DATA_DIR, "gene_summary.parquet"))
marker_expr  = pd.read_parquet(os.path.join(DATA_DIR, "marker_expr.parquet"))

with open(os.path.join(DATA_DIR, "top10_markers.json")) as fh:
    top10_markers = json.load(fh)

ALL_CELL_TYPES  = sorted(metadata["CellType_harmonized"].unique())
ALL_CONDITIONS  = sorted(metadata["Condition"].unique())
ALL_GENES       = sorted(gene_summary["gene"].unique())
MARKER_GENES    = sorted(marker_expr.columns.tolist())
TOTAL_CELLS     = len(metadata)

print(f"  {TOTAL_CELLS:,} cells · {len(ALL_GENES):,} genes · "
      f"{len(MARKER_GENES)} marker genes preloaded")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def subset_metadata(cell_types, conditions):
    df = metadata.copy()
    if cell_types:
        df = df[df["CellType_harmonized"].isin(cell_types)]
    if conditions:
        df = df[df["Condition"].isin(conditions)]
    return df


def build_umap_figure(df, color_by="Cell Type", gene=None):
    """Return a Plotly figure for the UMAP scatter."""
    if df.empty:
        return go.Figure().update_layout(
            template="plotly_white",
            annotations=[{"text": "No cells match the current filter.",
                          "showarrow": False, "font": {"size": 16}}],
        )

    hover = (
        "<b>%{customdata[0]}</b><br>"
        "Condition: %{customdata[1]}<br>"
        "Sample: %{customdata[2]}<br>"
        "n_genes: %{customdata[3]:,}<extra></extra>"
    )
    custom = df[["CellType_harmonized", "Condition",
                 "Patient", "n_genes"]].values

    if color_by == "Gene Expression" and gene and gene in MARKER_GENES:
        expr = marker_expr.loc[df.index, gene].values
        fig = go.Figure(go.Scatter(
            x=df["UMAP_1"], y=df["UMAP_2"],
            mode="markers",
            marker=dict(
                size=3, opacity=0.75,
                color=expr,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=gene, thickness=14, len=0.6),
            ),
            customdata=custom,
            hovertemplate=hover,
        ))
    elif color_by == "Condition":
        fig = px.scatter(
            df, x="UMAP_1", y="UMAP_2",
            color="Condition",
            color_discrete_map=CONDITION_COLORS,
            custom_data=["CellType_harmonized", "Condition", "Patient", "n_genes"],
            opacity=0.7,
        )
        fig.update_traces(marker_size=3, hovertemplate=hover)
    else:  # Cell Type (default)
        palette = {ct: CELL_TYPE_COLORS.get(ct, "#999999") for ct in df["CellType_harmonized"].unique()}
        fig = px.scatter(
            df, x="UMAP_1", y="UMAP_2",
            color="CellType_harmonized",
            color_discrete_map=palette,
            custom_data=["CellType_harmonized", "Condition", "Patient", "n_genes"],
            opacity=0.7,
        )
        fig.update_traces(marker_size=3, hovertemplate=hover)

    fig.update_layout(
        template="plotly_white",
        legend=dict(title="", itemsizing="constant",
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False),
        uirevision="keep",
    )
    return fig


def build_violin_figure(gene):
    """Violin plot: expression distribution per cell type for a given gene."""
    df = gene_summary[gene_summary["gene"] == gene].copy()
    if df.empty:
        return go.Figure().update_layout(template="plotly_white",
            annotations=[{"text": f"'{gene}' not found.", "showarrow": False}])

    # For violin we need the per-cell values from marker_expr if available
    if gene in MARKER_GENES:
        expr_df = marker_expr[[gene]].copy()
        expr_df = expr_df.join(metadata["CellType_harmonized"])
        expr_df.columns = ["expr", "cell_type"]
        fig = px.violin(
            expr_df[expr_df["expr"] > 0],
            x="cell_type", y="expr",
            color="cell_type",
            color_discrete_map={ct: CELL_TYPE_COLORS.get(ct, "#999") for ct in expr_df["cell_type"].unique()},
            box=True, points=False,
            labels={"expr": "log1p Expression", "cell_type": ""},
        )
    else:
        # Fallback: bar chart of mean expression
        fig = px.bar(
            df.sort_values("mean_expr", ascending=False),
            x="cell_type", y="mean_expr",
            color="cell_type",
            color_discrete_map={ct: CELL_TYPE_COLORS.get(ct, "#999") for ct in df["cell_type"].unique()},
            labels={"mean_expr": "Mean log1p Expr", "cell_type": ""},
        )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=80),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        title=dict(text=f"<b>{gene}</b> expression by cell type", font=dict(size=13)),
    )
    return fig


def build_dotplot_figure(gene):
    """Dot plot: % expressing vs mean expr per cell type."""
    df = gene_summary[gene_summary["gene"] == gene].copy()
    if df.empty:
        return go.Figure()
    df = df.sort_values("mean_expr", ascending=True)
    fig = px.scatter(
        df, x="cell_type", y="mean_expr",
        size="pct_expr", color="pct_expr",
        color_continuous_scale="Blues",
        size_max=22,
        labels={"mean_expr": "Mean log1p Expr", "cell_type": "",
                "pct_expr": "% Expressing"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=80),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        coloraxis_colorbar=dict(title="% Expr", thickness=12, len=0.5),
        title=dict(text=f"<b>{gene}</b> dot plot", font=dict(size=13)),
    )
    return fig


def marker_sidebar_table(cell_types_sel):
    """Build the top-10 markers table for selected cell type(s)."""
    if not cell_types_sel or len(cell_types_sel) != 1:
        return html.P("Select a single cell type to view top-10 markers.",
                      className="text-muted small")
    ct = cell_types_sel[0]
    markers = top10_markers.get(ct, [])
    rows = []
    for m in markers:
        rows.append(html.Tr([
            html.Td(html.B(m["gene"]), className="gene-name"),
            html.Td(f"{m['logFC']:.2f}", className="text-center text-muted small"),
            html.Td(f"{m['score']:.1f}", className="text-center text-muted small"),
        ]))
    return html.Div([
        html.P(f"Top 10 markers — {ct}", className="small fw-bold mb-1"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Gene"), html.Th("log2FC", className="text-center"),
                html.Th("Score", className="text-center"),
            ]), className="table-sm"),
            html.Tbody(rows),
        ], className="table table-sm table-hover mb-0"),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
dash_app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="HCC scRNA-seq Atlas",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app = dash_app.server  # Flask WSGI app for gunicorn/Render

# Health-check endpoint — required by Render.com to confirm the service is up
@app.route("/health")
def health():
    return {"status": "ok", "cells": TOTAL_CELLS}, 200

# ── Sidebar ──────────────────────────────────────────────────────────────────
sidebar = dbc.Card([
    dbc.CardHeader(html.H6("Filters & Options", className="mb-0 fw-bold")),
    dbc.CardBody([

        # Cell type
        html.Label("Cell Type", className="small fw-semibold mt-1"),
        dcc.Dropdown(
            id="ct-filter",
            options=[{"label": ct, "value": ct} for ct in ALL_CELL_TYPES],
            value=[],
            multi=True,
            placeholder="All cell types…",
            clearable=True,
            className="mb-3",
        ),
        dcc.Interval(id="load-interval", interval=100, max_intervals=1),  # Trigger on load

        # Condition
        html.Label("Condition", className="small fw-semibold"),
        dcc.RadioItems(
            id="condition-filter",
            options=[{"label": " All", "value": "all"}]
                   + [{"label": f" {c}", "value": c} for c in ALL_CONDITIONS],
            value="all",
            inputClassName="me-1",
            className="mb-3 small",
        ),

        # Color by
        html.Label("Color by", className="small fw-semibold"),
        dcc.RadioItems(
            id="color-by",
            options=[
                {"label": " Cell Type",        "value": "Cell Type"},
                {"label": " Condition",         "value": "Condition"},
                {"label": " Gene Expression",   "value": "Gene Expression"},
            ],
            value="Cell Type",
            inputClassName="me-1",
            className="mb-3 small",
        ),

        # Gene search
        html.Label("Gene Search", className="small fw-semibold"),
        dcc.Dropdown(
            id="gene-search",
            options=[{"label": g, "value": g} for g in ALL_GENES],
            value=None,
            placeholder="e.g. TREM2, FTH1…",
            clearable=True,
            searchable=True,
            className="mb-3",
        ),

        html.Hr(className="my-2"),

        # Live metadata display
        html.Div(id="metadata-display"),

        html.Hr(className="my-2"),

        # Top-10 markers
        html.Div(id="markers-display"),

    ], style={"overflowY": "auto", "maxHeight": "calc(100vh - 120px)"}),
], className="h-100 shadow-sm border-0")

# ── Main content ──────────────────────────────────────────────────────────────
main_content = dbc.Card([
    dbc.CardBody([
        dcc.Loading(
            dcc.Graph(
                id="umap-plot",
                config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]},
                style={"height": "55vh"},
            ),
            type="circle", color="#2C3E50",
        ),

        html.Hr(className="my-2"),

        dbc.Tabs([
            dbc.Tab(label="Gene Expression", tab_id="tab-gene", children=[
                dcc.Loading(
                    html.Div(id="gene-expr-display", className="mt-2"),
                    type="dot", color="#2C3E50",
                ),
            ]),
            dbc.Tab(label="Cell Type Summary", tab_id="tab-summary", children=[
                html.Div(id="celltype-summary", className="mt-2"),
            ]),
            dbc.Tab(label="Download", tab_id="tab-download", children=[
                html.Div([
                    html.P("Download the currently filtered cell metadata as CSV.",
                           className="mt-3 text-muted small"),
                    dbc.Button("Download metadata (CSV)", id="btn-download",
                               color="primary", size="sm", className="me-2"),
                    dcc.Download(id="download-metadata"),
                    html.Hr(),
                    html.P([
                        "Full processed dataset (h5ad) is available on ",
                        html.A("Zenodo", href="https://doi.org/10.5281/zenodo.XXXXXXX",
                               target="_blank"),
                        " under CC BY 4.0. Raw data from public repositories: ",
                        html.A("GSE242889", href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242889", target="_blank"),
                        ", ",
                        html.A("GSE228195", href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE228195", target="_blank"),
                        ", ",
                        html.A("GSE151530", href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151530", target="_blank"),
                        ", ",
                        html.A("GSE189903", href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189903", target="_blank"),
                        ", CZ CellxGene.",
                    ], className="small text-muted"),
                ], className="px-1"),
            ]),
        ], id="tabs", active_tab="tab-gene"),
    ])
], className="shadow-sm border-0")

# ── App layout ────────────────────────────────────────────────────────────────
dash_app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H4("HCC Single-Cell Transcriptomic Atlas",
                    className="mb-0 text-white fw-bold"),
            html.P(
                "Díaz-Campos & Hernández-Lemus · INMEGEN, México",
                className="mb-0 text-white-50 small",
            ),
        ], width=9),
        dbc.Col([
            html.Div([
                dbc.Badge(f"{TOTAL_CELLS:,} cells", color="light", text_color="dark",
                          className="me-1"),
                dbc.Badge(f"{len(ALL_GENES):,} genes", color="light", text_color="dark"),
            ], className="text-end mt-1"),
        ], width=3),
    ], className="py-2 px-3 mb-3",
       style={"background": "linear-gradient(90deg,#2C3E50,#4CA1AF)", "borderRadius": "8px"}),

    # Body
    dbc.Row([
        dbc.Col(sidebar,       width=3, className="pe-1"),
        dbc.Col(main_content,  width=9, className="ps-1"),
    ], className="g-2"),

    # Footer
    dbc.Row([
        dbc.Col(html.P(
            ["Data from public GEO/CellxGene repositories · "
             "Code: ",
             html.A("github.com/MiguelDiaz02/scRNAseq_a_pipeline_for_HCC",
                    href="https://github.com/MiguelDiaz02/scRNAseq_a_pipeline_for_HCC",
                    target="_blank"),
             " · Contact: ehernandez@inmegen.gob.mx / mdiazc161@unam.edu.mx"],
            className="text-center text-muted small mt-2 mb-1",
        )),
    ]),

    # Client-side stores
    dcc.Store(id="filtered-index"),   # list of cell_ids after filter
], fluid=True, className="px-3 py-2")

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. UMAP + stored filter index ─────────────────────────────────────────────
@dash_app.callback(
    Output("umap-plot",      "figure"),
    Output("filtered-index", "data"),
    Output("ct-filter", "value"),
    Input("ct-filter",        "value"),
    Input("condition-filter", "value"),
    Input("color-by",         "value"),
    Input("gene-search",      "value"),
    Input("load-interval",    "n_intervals"),
)
def update_umap(cell_types, condition, color_by, gene, n_intervals):
    # On initial load (n_intervals=1), set cell_types to all if empty
    if n_intervals == 1 and not cell_types:
        cell_types = list(ALL_CELL_TYPES)
    elif not cell_types:
        cell_types = list(ALL_CELL_TYPES)

    conditions = None if condition == "all" else [condition]
    df = subset_metadata(cell_types, conditions)
    fig = build_umap_figure(df, color_by=color_by, gene=gene)
    return fig, df.index.tolist(), cell_types


# ── 2. Metadata panel ─────────────────────────────────────────────────────────
@dash_app.callback(
    Output("metadata-display", "children"),
    Input("filtered-index", "data"),
)
def update_metadata(cell_ids):
    if not cell_ids:
        return html.P("No cells selected.", className="text-muted small")
    df = metadata.loc[cell_ids]
    n  = len(df)
    cond_counts = df["Condition"].value_counts()
    pct_healthy = cond_counts.get("Healthy Donors", 0) / n * 100
    pct_hcc     = cond_counts.get("HCC Diseased",   0) / n * 100
    ct_counts   = df["CellType_harmonized"].value_counts()
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Span(f"{n:,}", className="fs-5 fw-bold"),
                html.Br(),
                html.Span("cells", className="text-muted small"),
            ], className="text-center"), width=6),
            dbc.Col(html.Div([
                html.Span(f"{len(ct_counts)}", className="fs-5 fw-bold"),
                html.Br(),
                html.Span("cell types", className="text-muted small"),
            ], className="text-center"), width=6),
        ], className="mb-2"),
        html.Div([
            dbc.Progress(value=pct_healthy, color="primary",
                         label=f"Healthy {pct_healthy:.0f}%",
                         className="mb-1", style={"height": "14px", "fontSize": "10px"}),
            dbc.Progress(value=pct_hcc, color="danger",
                         label=f"HCC {pct_hcc:.0f}%",
                         style={"height": "14px", "fontSize": "10px"}),
        ]),
    ])


# ── 3. Top-10 markers sidebar ─────────────────────────────────────────────────
@dash_app.callback(
    Output("markers-display", "children"),
    Input("ct-filter", "value"),
)
def update_markers(cell_types):
    return marker_sidebar_table(cell_types)


# ── 4. Gene Expression tab ───────────────────────────────────────────────────
@dash_app.callback(
    Output("gene-expr-display", "children"),
    Input("gene-search",    "value"),
    Input("umap-plot",      "clickData"),
    Input("filtered-index", "data"),
)
def update_gene_expr(gene, click_data, cell_ids):
    triggered = ctx.triggered_id

    # Click on UMAP cell → show top-20 expressed genes for that cell
    if triggered == "umap-plot" and click_data:
        pt   = click_data["points"][0]
        umap1, umap2 = pt["x"], pt["y"]
        # Find closest cell
        dists = ((metadata["UMAP_1"] - umap1) ** 2 +
                 (metadata["UMAP_2"] - umap2) ** 2)
        cell_id = dists.idxmin()
        if cell_id in marker_expr.index:
            expr_row = marker_expr.loc[cell_id].sort_values(ascending=False).head(20)
            ct  = metadata.loc[cell_id, "CellType_harmonized"]
            cond = metadata.loc[cell_id, "Condition"]
            rows = [html.Tr([html.Td(html.B(g)), html.Td(f"{v:.3f}", className="text-end")])
                    for g, v in expr_row.items() if v > 0]
            return html.Div([
                html.P([html.B("Selected cell: "), f"{ct} · {cond}"],
                       className="small mb-1"),
                html.P("Top expressed marker genes:", className="small text-muted mb-1"),
                html.Table([
                    html.Thead(html.Tr([html.Th("Gene"), html.Th("log1p Expr", className="text-end")])),
                    html.Tbody(rows),
                ], className="table table-sm table-hover"),
            ])
        return html.P("Click a cell on the UMAP to see its top expressed genes.",
                      className="text-muted small mt-2")

    # Gene search → violin + dot
    if gene:
        in_markers = gene in MARKER_GENES
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=build_violin_figure(gene),
                                  config={"displayModeBar": False},
                                  style={"height": "260px"}), width=7),
                dbc.Col(dcc.Graph(figure=build_dotplot_figure(gene),
                                  config={"displayModeBar": False},
                                  style={"height": "260px"}), width=5),
            ]),
            html.P(
                "UMAP coloring by gene expression is available for marker genes. "
                "Select 'Gene Expression' in Color by to visualize."
                if in_markers else
                f"Note: '{gene}' is not in the preloaded marker set. "
                "Summary statistics are shown; UMAP coloring is unavailable for this gene.",
                className="text-muted small mt-1",
            ),
        ])

    return html.P("Search a gene above or click a cell on the UMAP.",
                  className="text-muted small mt-2")


# ── 5. Cell Type Summary tab ──────────────────────────────────────────────────
@dash_app.callback(
    Output("celltype-summary", "children"),
    Input("filtered-index",    "data"),
)
def update_ct_summary(cell_ids):
    if not cell_ids:
        return html.P("No cells.", className="text-muted small")
    df = metadata.loc[cell_ids]
    n  = len(df)

    summary = (
        df.groupby("CellType_harmonized")
        .agg(n_cells=("n_genes", "count"),
             mean_genes=("n_genes", "mean"))
        .reset_index()
    )
    summary["pct_total"] = (summary["n_cells"] / n * 100).round(1)
    summary = summary.sort_values("n_cells", ascending=False)

    # Top marker per cell type
    summary["top_markers"] = summary["CellType_harmonized"].apply(
        lambda ct: ", ".join([m["gene"] for m in top10_markers.get(ct, [])[:3]])
    )

    table_rows = [
        html.Tr([
            html.Td(html.Span("●", style={"color": CELL_TYPE_COLORS.get(row["CellType_harmonized"], "#999")})),
            html.Td(row["CellType_harmonized"]),
            html.Td(f"{row['n_cells']:,}", className="text-end"),
            html.Td(f"{row['pct_total']:.1f}%", className="text-end"),
            html.Td(row["top_markers"], className="text-muted small"),
        ])
        for _, row in summary.iterrows()
    ]

    # Composition bar chart
    comp_df = df.groupby(["Condition", "CellType_harmonized"]).size().reset_index(name="n")
    fig_comp = px.bar(
        comp_df, x="Condition", y="n", color="CellType_harmonized",
        color_discrete_map={ct: CELL_TYPE_COLORS.get(ct, "#999") for ct in comp_df["CellType_harmonized"].unique()},
        labels={"n": "# Cells", "CellType_harmonized": ""},
        barmode="stack",
    )
    fig_comp.update_layout(
        template="plotly_white",
        legend=dict(font=dict(size=9), title=""),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )

    return html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th(""), html.Th("Cell Type"),
                html.Th("n", className="text-end"),
                html.Th("%", className="text-end"),
                html.Th("Top 3 Markers"),
            ])),
            html.Tbody(table_rows),
        ], className="table table-sm table-hover mb-2"),
        dcc.Graph(figure=fig_comp, config={"displayModeBar": False},
                  style={"height": "220px"}),
    ])


# ── 6. Download filtered metadata ────────────────────────────────────────────
@dash_app.callback(
    Output("download-metadata", "data"),
    Input("btn-download", "n_clicks"),
    State("filtered-index", "data"),
    prevent_initial_call=True,
)
def download_metadata(n, cell_ids):
    if not n or not cell_ids:
        raise PreventUpdate
    df = metadata.loc[cell_ids].reset_index()
    return dcc.send_data_frame(df.to_csv, "hcc_atlas_cells.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"
    port  = int(os.environ.get("PORT", 8050))
    dash_app.run(debug=debug, host="0.0.0.0", port=port)
