"""
prepare_data.py
===============
Run ONCE locally (requires full h5ad + scanpy environment) to export
the minimal data files needed by the Dash portal at runtime.

Usage:
    conda activate sc_kernel   # or whichever env has scanpy + anndata
    python prepare_data.py

Outputs (all written to ./data/):
    metadata.parquet          – UMAP coords + cell metadata (~5–10 MB)
    top10_markers.json        – top 10 marker genes per cell type
    gene_summary.parquet      – mean log1p-expr + pct-expressing per gene × celltype
    marker_expr.parquet       – per-cell expression for marker genes (UMAP coloring)
"""

import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

H5AD_PATH = "/home/mdiaz/HCC_project/MERGED_adata/scvi_integrated.h5ad"
OUT_DIR   = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & FILTER
# ─────────────────────────────────────────────
print("Loading h5ad (backed mode)…")
adata = ad.read_h5ad(H5AD_PATH, backed="r")

# Keep only QC-passing, non-excluded cells
mask = (adata.obs["QC_pass"] == True) & (adata.obs["Condition"] != "Excluded")
print(f"  Total cells: {adata.n_obs:,}  →  after filter: {mask.sum():,}")

# Load fully into memory only the filtered subset
adata = adata[mask].to_memory()
print(f"  Genes: {adata.n_vars:,}")

# ─────────────────────────────────────────────
# 2. METADATA + UMAP  →  metadata.parquet
# ─────────────────────────────────────────────
print("Exporting metadata.parquet…")

umap = pd.DataFrame(
    adata.obsm["X_umap"],
    index=adata.obs_names,
    columns=["UMAP_1", "UMAP_2"],
)
obs_cols = ["CellType_harmonized", "Condition", "DX", "Patient",
            "n_genes", "pct_counts_mt", "total_counts", "dataset"]
obs_cols = [c for c in obs_cols if c in adata.obs.columns]

metadata = pd.concat([umap, adata.obs[obs_cols]], axis=1)
metadata.index.name = "cell_id"
metadata.to_parquet(os.path.join(OUT_DIR, "metadata.parquet"), index=True)
print(f"  metadata.parquet: {metadata.shape} rows×cols")

# ─────────────────────────────────────────────
# 3. TOP-10 MARKERS  →  top10_markers.json
# ─────────────────────────────────────────────
print("Computing top-10 markers per cell type (Wilcoxon, one-vs-rest)…")
sc.tl.rank_genes_groups(
    adata,
    groupby="CellType_harmonized",
    method="wilcoxon",
    n_genes=10,
    key_added="rank_genes_ct",
)

cell_types = list(adata.obs["CellType_harmonized"].cat.categories
                  if hasattr(adata.obs["CellType_harmonized"], "cat")
                  else adata.obs["CellType_harmonized"].unique())

top10 = {}
rgg = adata.uns["rank_genes_ct"]
for ct in cell_types:
    genes  = list(rgg["names"][ct])
    scores = [float(s) for s in rgg["scores"][ct]]
    logfcs = [float(f) for f in rgg["logfoldchanges"][ct]]
    pvals  = [float(p) for p in rgg["pvals_adj"][ct]]
    top10[ct] = [
        {"gene": g, "score": round(s, 2), "logFC": round(f, 2), "pval_adj": round(p, 4)}
        for g, s, f, p in zip(genes, scores, logfcs, pvals)
    ]

with open(os.path.join(OUT_DIR, "top10_markers.json"), "w") as fh:
    json.dump(top10, fh, indent=2)
print(f"  top10_markers.json: {len(top10)} cell types")

# ─────────────────────────────────────────────
# 4. GENE SUMMARY  →  gene_summary.parquet
#    mean log1p-expr + % expressing per gene × cell type
# ─────────────────────────────────────────────
print("Computing gene summary (mean expr + pct expressing per cell type)…")
import scipy.sparse as sp

X = adata.X
if not sp.issparse(X):
    X = sp.csr_matrix(X)

cell_types_arr = adata.obs["CellType_harmonized"].values
gene_names     = adata.var_names.tolist()

rows = []
for ct in cell_types:
    ct_mask  = cell_types_arr == ct
    X_ct     = X[ct_mask]
    n_ct     = ct_mask.sum()
    mean_expr = np.asarray(X_ct.mean(axis=0)).flatten()         # mean log1p
    pct_expr  = np.asarray((X_ct > 0).mean(axis=0)).flatten()   # fraction expressing
    for gi, gene in enumerate(gene_names):
        rows.append({
            "gene":      gene,
            "cell_type": ct,
            "mean_expr": round(float(mean_expr[gi]), 4),
            "pct_expr":  round(float(pct_expr[gi]),  4),
        })

gene_summary = pd.DataFrame(rows)
gene_summary.to_parquet(os.path.join(OUT_DIR, "gene_summary.parquet"), index=False)
print(f"  gene_summary.parquet: {gene_summary.shape}")

# ─────────────────────────────────────────────
# 5. MARKER EXPRESSION  →  marker_expr.parquet
#    Per-cell expression for top-N marker genes (for UMAP coloring)
# ─────────────────────────────────────────────
print("Exporting per-cell marker expression for UMAP coloring…")

# Collect all unique marker genes across cell types
marker_genes = sorted({
    entry["gene"]
    for ct_markers in top10.values()
    for entry in ct_markers
})
# Also add canonical HCC markers worth exploring interactively
extra_markers = [
    "FTH1", "FGB", "FGA", "FGG", "TREM2", "CD44", "EPCAM", "ALDH1A1",
    "CD24", "VEGFA", "KDR", "MYC", "CCND1", "AXIN2", "GLI1", "PTCH1",
    "GZMK", "PDCD1", "NCAM1", "CD3E", "HLA-A", "MARCO", "LILRA5",
    "CD36", "SPP1", "CXCL8", "CXCL2", "S100A4", "ADM", "COL4A1",
    "APOE", "APOB", "GPC3", "AFP", "ALB", "CYP3A4", "ACTA2", "EPCAM",
]
all_markers = sorted(set(marker_genes + extra_markers))

# Filter to genes actually present
present = [g for g in all_markers if g in adata.var_names]
print(f"  Marker genes for UMAP coloring: {len(present)}")

gene_idx = [adata.var_names.get_loc(g) for g in present]
X_markers = X[:, gene_idx]
if sp.issparse(X_markers):
    X_markers = X_markers.toarray()

marker_df = pd.DataFrame(
    X_markers,
    index=adata.obs_names,
    columns=present,
)
marker_df.index.name = "cell_id"
marker_df.to_parquet(os.path.join(OUT_DIR, "marker_expr.parquet"), index=True)
print(f"  marker_expr.parquet: {marker_df.shape}")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
import os as _os
for fname in ["metadata.parquet", "top10_markers.json",
              "gene_summary.parquet", "marker_expr.parquet"]:
    fpath = _os.path.join(OUT_DIR, fname)
    size_mb = _os.path.getsize(fpath) / 1e6
    print(f"  {fname}: {size_mb:.1f} MB")

print("\n✅  All data files ready in ./data/")
print("    Next: python app.py   or   gunicorn app:server")
