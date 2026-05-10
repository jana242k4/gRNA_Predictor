"""
Central configuration for the OmicsCRISPR data pipeline.

All paths, data source URLs, and cell type definitions live here.
Scripts import from this module rather than hardcoding values.
"""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
BACKEND_DIR   = Path(__file__).parent.parent
OMICS_DIR     = BACKEND_DIR / "data" / "omics"

DEPMAP_DIR    = OMICS_DIR / "depmap"
ENCODE_DIR    = OMICS_DIR / "encode"
SPLICE_DIR    = OMICS_DIR / "splice"
FEATURES_DIR  = OMICS_DIR / "features"

for _d in (DEPMAP_DIR, ENCODE_DIR, SPLICE_DIR, FEATURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── DepMap release ────────────────────────────────────────────────────────────
# DepMap 22Q2 — figshare article 19700056 (verified: contains guide-level data)
# https://figshare.com/articles/dataset/DepMap_Public_22Q2/19700056
DEPMAP_RELEASE       = "22Q2"
DEPMAP_FIGSHARE_ID   = 19700056
DEPMAP_FIGSHARE_API  = f"https://api.figshare.com/v2/articles/{DEPMAP_FIGSHARE_ID}/files"

# Files to download from 22Q2 (exact names as they appear in figshare)
DEPMAP_FILES = {
    "guide_map":      "Achilles_guide_map.csv",       #   3 MB — sequences + genomic coords
    "guide_efficacy": "Achilles_guide_efficacy.csv",  #   2 MB — aggregated mean LFC per guide
    "cell_info":      "sample_info.csv",              #  <1 MB — cell line metadata
    "gene_effect":    "Achilles_gene_effect.csv",     # 336 MB — CERES gene effect per cell line
}

# Processed cell-type-specific gene effect output
CELL_TYPE_GENE_EFFECT_CSV = DEPMAP_DIR / "cell_type_gene_effect.csv"

# ── ENCODE portal ─────────────────────────────────────────────────────────────
ENCODE_BASE    = "https://www.encodeproject.org"
ENCODE_SEARCH  = ENCODE_BASE + "/search/"

# Cell types: internal_name -> ENCODE biosample ontology term
# Focus: primary immune cells (CAR-T clinical relevance) + cell line controls
# Terms verified against ENCODE portal biosample search
TARGET_CELL_TYPES: dict[str, str] = {
    "T_cell_CD4":  "CD4-positive, alpha-beta T cell",
    "T_cell_CD8":  "CD8-positive, alpha-beta T cell",
    "NK_cell":     "natural killer cell",
    "B_cell":      "B cell",
    "monocyte":    "CD14-positive monocyte",
    "K562":        "K562",
    "HEK293":      "HEK293",
}

# ENCODE search params for RNA-seq gene quantification files
ENCODE_RNA_PARAMS = {
    "type":                     "File",
    "output_type":              "gene quantifications",
    "file_format":              "tsv",
    "assay_title":              "polyA plus RNA-seq",
    "status":                   "released",
    "assembly":                 "GRCh38",
    "format":                   "json",
    "limit":                    "10",
}

# ENCODE search params for ATAC-seq narrowPeak BED files
# file_format_type=narrowPeak covers replicated peaks, IDR thresholded peaks, etc.
ENCODE_ATAC_PARAMS = {
    "type":                     "File",
    "file_format":              "bed",
    "file_format_type":         "narrowPeak",
    "assay_title":              "ATAC-seq",
    "status":                   "released",
    "assembly":                 "GRCh38",
    "format":                   "json",
    "limit":                    "5",
}

# ── GENCODE splice annotations ────────────────────────────────────────────────
GENCODE_VERSION = 44
GENCODE_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
    f"release_{GENCODE_VERSION}/gencode.v{GENCODE_VERSION}.annotation.gtf.gz"
)
GENCODE_GTF_GZ  = SPLICE_DIR / f"gencode.v{GENCODE_VERSION}.annotation.gtf.gz"
SPLICE_BED_OUT  = SPLICE_DIR / "splice_sites_grch38.bed"
SPLICE_PKL_OUT  = SPLICE_DIR / "splice_sites_grch38.pkl"

# ── Output feature matrix ─────────────────────────────────────────────────────
OMICS_FEATURES_CSV = FEATURES_DIR / "omics_features.csv"
