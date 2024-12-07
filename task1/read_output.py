import scanpy as sc
import anndata as ad

adata = sc.read_h5ad('output_pipeline.h5ad')

print(adata.var)
print(adata.obs)
