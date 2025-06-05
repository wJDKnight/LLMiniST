import os
import numpy as np
import pandas as pd
import scanpy as sc
import json
from sklearn.model_selection import train_test_split
from src.utils import *
import src.prompt as prompt
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

def generate_training_data(neighbor_normalized_df, neighbor_normalized_df_genes, adata, config):
    """Generate training data and prototype"""
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
    
    return (train_neighbor_normalized_df, train_neighbor_normalized_df_genes, 
            val_neighbor_normalized_df, val_neighbor_normalized_df_genes)

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
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Setup configuration
    config = setup_config()
    
    # Load and process data
    adata, cell_proportion_data = load_and_process_data(config)
    
    # Prepare neighbor data
    neighbor_normalized_df, neighbor_normalized_df_genes = prepare_neighbor_data(
        adata, cell_proportion_data, config
    )
    
    # Generate training data
    (train_neighbor_normalized_df, train_neighbor_normalized_df_genes,
     val_neighbor_normalized_df, val_neighbor_normalized_df_genes) = generate_training_data(
        neighbor_normalized_df, neighbor_normalized_df_genes, adata, config
    )
    
    # Setup domain mappings
    setup_domain_mappings(adata, config)
    
    # Generate and save training files
    output_folder = f"finetune_json/{config.data_name}_{config.model_type}/"
    os.makedirs(output_folder, exist_ok=True)
    
    train_output_file = f"{output_folder}{config.data_name}_train_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.json"
    
    # Generate training data JSON
    print(f"Generating json for training into {train_output_file}")
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
            json_str = json.dumps(row_data)
            f.write(json_str + '\n')
    
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
    
    # Generate end2end JSON
    generate_json_end2end(
        val_neighbor_normalized_df, 
        config, 
        prompt_func=prompt.finetune_user_celltype_geneorder,
        n_rows=1,
        batch_size=3000,
        df_extra=val_neighbor_normalized_df_genes
    )
    
    # Run submit_end2end.py
    cmd = f"nohup python -u submit_end2end.py model_config/config_finetunePro_libd.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_finetunePro{config.replicate}.out 2>&1 &"
    subprocess.run(cmd, check=True, text=True, shell=True)

if __name__ == "__main__":
    main()


