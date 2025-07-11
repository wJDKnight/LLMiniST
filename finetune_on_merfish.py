import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from sklearn.model_selection import train_test_split
from src.utils import *
import src.prompt as prompt
import subprocess
import openai
import json

def setup_config():
    """Load and setup configuration"""
    config = load_config("model_config/config_finetunePro_merfish.yaml")
    config.data_name = "MERFISH_25"
    config.folder_path = f"./batch_json/{config.data_name}_{config.model_type}"
    config.output_path = f"./batch_results/{config.data_name}_{config.model_type}"
    return config

def load_and_process_data(config):
    """Load and process MERFISH data"""
    data_path = f"/Users/hw568/storage/collections_spatial_datasets/MERFISH/{config.data_name}.h5ad"
    adata = sc.read_h5ad(data_path)
    adata.obs.rename(columns={'cell_class': 'cell_type'}, inplace=True)
    
    # Clean cell IDs and remove NaN values
    adata.obs_names = list(range(len(adata)))
    adata.obs_names = adata.obs_names.astype(str)
    adata = adata[~adata.obs[config.name_truth].isna()].copy()
    
    # Process spatial data
    pos_data = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
    adata.obs = adata.obs.join(pos_data)
    
    # Rename cell types
    celltype_rename = {
        'Astrocyte': 'Astrocyte',
        'Endothelial 1': 'Endothelial',
        'OD Mature 2': 'Mature oligodendrocytes',
        'Inhibitory': 'Inhibitory',
        'OD Immature 1': 'Immature oligodendrocytes',
        'Excitatory': 'Excitatory',
        'Endothelial 3': 'Endothelial',
        'Microglia': 'Microglia',
        'OD Mature 1': 'Mature oligodendrocytes',
        'Pericytes': 'Pericytes',
        'OD Mature 4': 'Mature oligodendrocytes',
        'Endothelial 2': 'Endothelial',
        'OD Mature 3': 'Mature oligodendrocytes',
        'OD Immature 2': 'Immature oligodendrocytes',
        'Ependymal': 'Ependymal'
    }
    adata.obs['cell_type'] = adata.obs['cell_type'].map(celltype_rename)
    
    # Preprocess expression data
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    
    return adata

def prepare_neighbor_data(adata, config):
    """Prepare neighbor matrices for cell types and genes"""
    pos_data = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
    
    # Compute adjacency matrix
    adj_matrix, _ = sparse_adjacency(pos_data, threshold=config.r)
    adj_matrix = adj_matrix + scipy.sparse.diags(np.ones(adj_matrix.shape[0]))
    
    # Calculate cell type neighbors
    one_hot_df = pd.get_dummies(adata.obs[['cell_type']], prefix='').astype(int)
    one_hot_matrix = csr_matrix(one_hot_df.values)
    neighbor_count = adj_matrix.dot(one_hot_matrix)
    n_neighbors = adj_matrix.sum(axis=1).A1
    n_neighbors_col = n_neighbors.reshape(-1, 1)
    
    # Normalize matrices
    neighbor_matrix_normalized = neighbor_count / n_neighbors_col
    neighbor_normalized_df = pd.DataFrame(neighbor_matrix_normalized.toarray(),
                                        index=adata.obs_names,
                                        columns=one_hot_df.columns.str.lstrip('_'))
    
    # Calculate gene neighbors
    neighbor_genes = adj_matrix.dot(adata[:, config.gene_list].X)
    neighbor_matrix_normalized_genes = neighbor_genes / n_neighbors_col
    neighbor_normalized_df_genes = pd.DataFrame(neighbor_matrix_normalized_genes,
                                              index=adata.obs_names,
                                              columns=config.gene_list)
    
    return neighbor_normalized_df, neighbor_normalized_df_genes

def generate_training_data(neighbor_normalized_df, neighbor_normalized_df_genes, adata, config):
    """Split data and generate prototype"""
    train_neighbor_normalized_df, val_neighbor_normalized_df = train_test_split(
        neighbor_normalized_df,
        test_size=1-config.prototype_p,
        random_state=42,
        stratify=adata.obs[config.name_truth]
    )
    
    train_neighbor_normalized_df_genes, val_neighbor_normalized_df_genes = train_test_split(
        neighbor_normalized_df_genes,
        test_size=1-config.prototype_p,
        random_state=42,
        stratify=adata.obs[config.name_truth]
    )
    
    # Calculate prototype
    train_neighbor_df = train_neighbor_normalized_df.join(train_neighbor_normalized_df_genes)
    one_shot_df = pd.concat([adata.obs[config.name_truth].loc[train_neighbor_df.index], 
                            train_neighbor_df], axis=1).groupby(config.name_truth, observed=False).mean()
    
    return (train_neighbor_normalized_df, train_neighbor_normalized_df_genes,
            val_neighbor_normalized_df, val_neighbor_normalized_df_genes,
            one_shot_df)

def main():
    # Initialize
    config = setup_config()
    client = openai.OpenAI()
    config.gene_list = ['Fn1', 'Slco1a4', 'Rgs5', 'Myh11', 'Klf4', 'Sst', 'Scgn', 'Cyp19a1', 
                       'Crh', 'Vgf', 'Aldh1l1', 'Aqp4', 'Cxcl14', 'Pou3f2', 'Mlc1', 'Oxt', 
                       'Ebf3', 'Trh', 'Cbln1', 'Adcyap1', 'Gal', 'Slc18a2', 'Nts', 'Calcr', 
                       'Scg2', 'Ermn', 'Mbp', 'Sgk1', 'Opalin', 'Gjc3', 'Pdgfra', 'Traf4', 
                       'Sox8', 'Sox6', 'Necab1', 'Ntng1', 'Slc17a6', 'Ramp3', 'Sp9', 'Selplg', 
                       'Man1a', 'Rgs2', 'Slc15a3', 'Nnat', 'Cd24a', 'Cckbr', 'Cyr61']
    
    # Load and process data
    adata = load_and_process_data(config)
    neighbor_normalized_df, neighbor_normalized_df_genes = prepare_neighbor_data(adata, config)
    
    # Generate training data
    (train_neighbor_normalized_df, train_neighbor_normalized_df_genes,
     val_neighbor_normalized_df, val_neighbor_normalized_df_genes,
     one_shot_df) = generate_training_data(neighbor_normalized_df, neighbor_normalized_df_genes, adata, config)
    
    # Setup domain mappings
    config.domain_mapping = {i: layer for i, layer in enumerate(adata.obs[config.name_truth].unique())}
    config.cell_names_mapping = {ct: ct for ct in adata.obs['cell_type'].unique()}
    
    # Generate system prompt
    config.system_prompt = prompt.CP_celltype_geneorder(one_shot_df, config)
    
    # Generate and save training file
    output_folder = f"finetune_json/{config.data_name}_{config.model_type}/"
    os.makedirs(output_folder, exist_ok=True)
    train_output_file = f"{output_folder}{config.data_name}_train.json"
    
    with open(train_output_file, 'w') as f:
        for i in range(train_neighbor_normalized_df.shape[0]):
            system_p = config.system_prompt
            user_p = prompt.finetune_user_celltype_geneorder(
                train_neighbor_normalized_df, train_neighbor_normalized_df_genes, i, config
            )
            assistant_p = prompt.finetune_assistant(
                train_neighbor_normalized_df, i, adata.obs[config.name_truth]
            )
            row_data = {
                "messages": [
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p},
                    {"role": "assistant", "content": assistant_p}
                ]
            }
            f.write(json.dumps(row_data) + '\n')
    
    # Submit finetune job
    train_file = client.files.create(
        file=open(train_output_file, "rb"),
        purpose="fine-tune"
    )
    
    finetune_job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        model="gpt-4o-mini-2024-07-18",
        suffix=f"{config.model_type}_{config.r}"
    )
    
    # Generate end2end evaluation data
    generate_json_end2end(
        val_neighbor_normalized_df,
        config,
        prompt_func=prompt.finetune_user_celltype_geneorder,
        df_extra=val_neighbor_normalized_df_genes,
        batch_size=3000
    )
    
    # Run submit_end2end.py
    cmd = f"nohup python -u submit_end2end.py model_config/config_finetunePro_libd.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_finetunePro{config.replicate}.out 2>&1 &"
    subprocess.run(cmd, check=True, text=True, shell=True)

if __name__ == "__main__":
    main()



