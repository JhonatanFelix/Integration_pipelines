import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import gzip
import os 

from scipy.stats import median_abs_deviation ## use in function is_outlier
from scipy.io import mmread                  ## use in function get_matrix


def get_matrix(matrix_file):
    with gzip.open(matrix_file, 'rb') as f:
        sparse_matrix = mmread(f)
    dense_matrix = sparse_matrix.todense()  # Dimensions: cells x genes
    return dense_matrix


barcodes_all = pd.DataFrame()
features_all = pd.DataFrame()
matrix_all = None  


meta_path = './input_data/GSE227828_raw_clinical_metadata.csv'

meta_df = pd.read_csv(meta_path)
samples = meta_df[['SAMPLE_ID','series_id']]


	
for series in samples['series_id'].unique():
    path = f'./input_data/{series}_RAW_extracted'
    files = os.listdir(path)

    features_path = [file for file in files if file.endswith('features.tsv.gz')][0]
    features = pd.read_csv(os.path.join(path, features_path), compression='gzip', header=None, sep='\t')
    features = features.rename(columns = {0: 'gene_ids', 1: 'code', 2: 'feature_types'})
    for sample in samples['SAMPLE_ID']:
        list_samples = [file for file in files if file.startswith(sample)]

        barcodes_path = [file for file in list_samples if file.endswith('barcodes.tsv.gz')][0]
        barcodes = pd.read_csv(os.path.join(path, barcodes_path), compression='gzip', header=None, sep='\t')[:1000] #############################
        barcodes['sample_id'] = sample
        barcodes['dataset']   = series

        matrix_path = [file for file in list_samples if file.endswith('matrix.mtx.gz')][0] 
        matrix = np.array(get_matrix(os.path.join(path, matrix_path))[:,:1000].T) ################################# 

        barcodes_all = pd.concat([barcodes_all, barcodes], ignore_index=True)
        if matrix_all is None:
            matrix_all = matrix
        else:
            matrix_all = np.vstack([matrix_all, matrix])

print(barcodes_all)
print('################################ \n')
print(features)
print('################################ \n')
print(matrix_all)
print(matrix.shape)
print('################################ \n')
print(barcodes_all.iloc[:,[0]])





#### Pipeline



############################ Pre-processing ####################################
# Part 0
################################################################################
# organizing the obs dataset to follow with the basic analysis

def organizing_and_QC(matrix_all, barcodes_all, features):
  adata = ad.AnnData(X= matrix_all, var=features ,obs=barcodes_all.iloc[:,[0]])

  adata.obs = adata.obs.rename(columns = {0: 'barcodes'})
  adata.obs['sample_id'] = barcodes_all['sample_id'].astype(str).tolist()
  adata.obs['cell_id']   = adata.obs.sample_id + '_'+ adata.obs.barcodes
  adata.obs = adata.obs.set_index('cell_id')
  adata.obs['dataset'] = barcodes_all['dataset'].astype(str).tolist()
  adata.obs = adata.obs.drop(columns=  'barcodes')

  #. Organizing the var dataset also
  #adata.var = adata.var.rename(columns = {1: '',2: 'feature_types'})
  adata.var = adata.var.set_index('code')
  adata.var_names_make_unique()

  # Separating in different genes:
  # mitochondrial genes
  adata.var["mt"] = adata.var_names.str.startswith("MT-")
  # ribosomal genes
  adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
  # hemoglobin genes.
  adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))

  #### Calculating quality control metrics
  sc.pp.calculate_qc_metrics(
      adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20,30,50],
      log1p=True
  )
  return adata 

# Helper function to verify outlier points
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        M > np.median(M) + nmads * median_abs_deviation(M)
    )
    return outlier


############################ Processing #######################################
# Part 1 #######################################################################
# Filters to cells and genes in low quality
def filters_cell_genes(adata, min_cells = 100, min_genes = 200):
  adata.obs["outlier"] = (
      is_outlier(adata, "log1p_total_counts", 5)
      | is_outlier(adata, "log1p_n_genes_by_counts", 5)
      | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
  )

  adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
      adata.obs["pct_counts_mt"] > 8
  )

  sc.pp.filter_cells(adata, min_genes= min_genes)
  sc.pp.filter_genes(adata, min_cells= min_cells)

  return adata


# Part 2 #######################################################################
# Initial normalization
def normalization(adata, target_sum = 1e4):
  adata.layers["raw_counts"] = adata.X.copy()
  sc.pp.normalize_total(adata, target_sum= target_sum)
  adata.layers['norm_counts'] = adata.X.copy()

  return adata

# Selection of highly variable genes
def highly_genes_and_log(adata, n_top_genes = 2000, subset = False):
  sc.pp.highly_variable_genes(
      adata, flavor="cell_ranger", n_top_genes= n_top_genes, subset= subset
  )  ### It could be setted subset= True to really filter this dataset

  # Logaritmization
  sc.pp.log1p(adata)
  adata.layers["log1p_norm"] = adata.X.copy()

  return adata

# Part 3 #######################################################################
# dimentional reduction
def dimentional_reduction(adata, n_comps = 50):
  sc.tl.pca(adata, n_comps= n_comps)

  return adata

# Part 4 #######################################################################
# Clustering and Visualization

def clustering_and_visualization(adata, n_neighbors = 15, n_pcs = 30, 
                                 early_exaggeration = 12, learning_rate = 1000, 
                                 metric = 'euclidean', n_jobs = 1, perplexity = 30, 
                                 use_rep = 'X_pca', resolution = 1.0, 
                                 key_added = 'leiden_res_1'):
    
  sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
  sc.tl.tsne(adata, n_pcs=50, early_exaggeration = 12, learning_rate = 1000,
            metric = 'euclidean', n_jobs = 1, perplexity = 30, use_rep = 'X_pca')
  sc.tl.umap(adata)                                              
  sc.tl.leiden(adata, resolution=1.0, key_added = 'leiden_res_1')
  return adata



adata = organizing_and_QC(matrix_all, barcodes_all, features)
adata = filters_cell_genes(adata)
adata = normalization(adata)
adata = highly_genes_and_log(adata)
adata = dimentional_reduction(adata)
adata = clustering_and_visualization(adata)



adata.write('output_pipeline.h5ad')

