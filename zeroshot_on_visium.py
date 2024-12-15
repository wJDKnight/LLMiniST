import os
import numpy as np
import pandas as pd
import scanpy as sc
from utils import *
import prompt
import subprocess
import openai

def setup_config():
    """Load and setup configuration"""
    config = load_config("model_config/config_finetunePro_libd.yaml")
    config.data_name = "151673"
    config.folder_path = f"./batch_json/{config.data_name}_{config.model_type}"
    config.output_path = f"./batch_results/{config.data_name}_{config.model_type}"
    return config

def load_and_process_data(config):
    """Load and process Visium data"""
    print(f"Processing data: {config.data_name}")
    data_path = f"/Users/hw568/storage/collections_spatial_datasets/spatialLIBD/{config.data_name}/"
    adata = sc.read_visium(data_path)
    adata.var_names_make_unique()

    # Load cell proportions
    cell_proportion_data = pd.read_csv(os.path.join("./deconv_result", f"celltype_proportions_{config.data_name}.csv"), index_col=0)
    adata.obs = adata.obs.join(cell_proportion_data)

    # Process metadata
    meta_data = pd.read_csv(os.path.join(data_path, "metadata.tsv"), sep="\t")
    adata.obs = adata.obs.merge(meta_data, left_index=True, right_index=True, how="left")
    adata = adata[~adata.obs[config.name_truth].isna()].copy()
    
    # Clean cell IDs
    adata.obs_names = [f'spot_{i}' for i in range(len(adata.obs_names))]
    
    return adata, cell_proportion_data

def prepare_neighbor_data(adata, cell_proportion_data, config):
    """Prepare neighbor matrices"""
    pos_data = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
    
    # Compute adjacency matrix
    adj_matrix, _ = sparse_adjacency(pos_data, threshold=config.r, add_diagonal=True)
    n_neighbors = adj_matrix.sum(axis=1).A1
    n_neighbors_col = n_neighbors.reshape(-1, 1)
    
    # Calculate neighbor counts
    neighbor_count = adj_matrix.dot(cell_proportion_data)
    neighbor_matrix_normalized = neighbor_count / n_neighbors_col
    neighbor_normalized_df = pd.DataFrame(neighbor_matrix_normalized, 
                                        index=adata.obs_names,
                                        columns=cell_proportion_data.columns)

    # Calculate neighbor genes
    neighbor_genes = adj_matrix.dot(adata.X)
    neighbor_matrix_normalized_genes = neighbor_genes / n_neighbors_col
    neighbor_normalized_df_genes = pd.DataFrame(neighbor_matrix_normalized_genes, 
                                              index=adata.obs_names, 
                                              columns=adata.var_names)
    
    return neighbor_normalized_df, neighbor_normalized_df_genes

def setup_domain_mappings(adata, config):
    """Setup domain and cell name mappings"""
    unique_layers = adata.obs[config.name_truth].unique()
    config.domain_mapping = {i: layer for i, layer in enumerate(unique_layers)}
    
    config.cell_names_mapping = {
        'Astro': 'Astrocyte',
        'EndoMural': 'Endothelial and mural cells',
        'Excit_L2_3': 'Excitatory neuron layer 2/3',
        'Excit_L3': 'Excitatory neuron layer 3',
        'Excit_L3_4_5': 'Excitatory neuron layer 3/4/5',
        'Excit_L4': 'Excitatory neuron layer 4',
        'Excit_L5': 'Excitatory neuron layer 5',
        'Excit_L5_6': 'Excitatory neuron layer 5/6',
        'Excit_L6': 'Excitatory neuron layer 6',
        'Inhib': 'Inhibitory neuron',
        'Micro': 'Microglia',
        'OPC': 'Oligodendrocyte precursor cell',
        'Oligo': 'Oligodendrocyte'
    }

def main():
    # Initialize
    config = setup_config()
    client = openai.OpenAI()
    
    # Load and process data
    adata, cell_proportion_data = load_and_process_data(config)
    
    # Prepare neighbor data
    neighbor_normalized_df, neighbor_normalized_df_genes = prepare_neighbor_data(
        adata, cell_proportion_data, config
    )
    
    # Setup domain mappings
    setup_domain_mappings(adata, config)
    
    # Generate system prompt for zero-shot
    config.system_prompt = prompt.CP_celltype_zeroshot(config)
    
    # Generate end2end evaluation data
    generate_json_end2end(
        neighbor_normalized_df, 
        config, 
        prompt_func=prompt.zeroshot_user_celltype,
        n_rows=1,
        batch_size=3000,
        df_extra=neighbor_normalized_df_genes
    )
    
    # Run submit_end2end.py
    cmd = f"nohup python -u submit_end2end.py model_config/config_finetunePro_libd.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_finetunePro{config.replicate}.out 2>&1 &"
    subprocess.run(cmd, check=True, text=True, shell=True)

if __name__ == "__main__":
    main()



