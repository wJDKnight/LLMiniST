# %%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.utils import *
import src.prompt as prompt
from src.data_loader import load_spatial_data_csv
import subprocess
import json


import google.generativeai as genai
import pickle
# %%
# optional
representetive_gene_list = ["Flt1", "Mgp", "Bgn", "Mylk", "Aqp4", "Rorb", "Cxcl14", "Pcdhgc3", "Ctgf", "Mog", "Enpp2", "Tpbg", "Tcerg1l",
"Reln", "Pnoc", "Gad1", "Gad2", "Ndnf", "Vip", "Synpr", "Cux2", "Nos1", "Npy", "Lhx6", "Rbp4", "Pcdhgc4",
"Sema3e", "Sema3c", "Sst", "Syt6", "Sla", "Pcp4", "Foxp2"]  # cell type marker genes

# %%
config = load_config("model_config/config_zeroshot_example.yaml")
# in case you want to change the data name, replicate, folder path, output path
config.data_name = "BZ5"
config.replicate = "_testgene"
config.folder_path = f"./batch_json/{config.data_name}_{config.model_type}"
config.output_path = f"./batch_results/{config.data_name}_{config.model_type}"

# for data with both cell type and representetive_gene_list
prompt_func = prompt.zeroshot_celltype_geneorder

# for data with only cell type
# prompt_func = prompt.zeroshot_celltype

# %%
# potential niches exist in the data
domain_mapping = {1 : "Layer 1", 2 : "Layer 2/3", 3 : "Layer 5", 4 : "Layer 6"}
config.domain_mapping = domain_mapping

# optional, if you want to map the cell type to a more specific name
cell_names_mapping = {'Astro': 'Astrocytes',
 'Endo': 'Endothelial cells',
 'L5-1': 'Layer 5 pyramidal neuron subtype 1',
 'Lhx6': 'Lhx6-expressing interneurons',
 'NPY': 'Neuropeptide Y-expressing interneurons',
 'Oligo': 'Oligodendrocytes',
 'Reln': 'Reelin-expressing cells',
 'SST': 'Somatostatin-expressing interneurons',
 'Smc': 'Smooth muscle cells',
 'VIP': 'Vasoactive intestinal peptide-expressing interneurons',
 'eL2/3': 'Excitatory neuron layer 2/3',
 'eL5-2': 'Excitatory neuron layer 5 subtype 2',
 'eL5-3': 'Excitatory neuron layer 5 subtype 3',
 'eL6-1': 'Excitatory neuron layer 6 subtype 1',
 'eL6-2': 'Excitatory neuron layer 6 subtype 2'}
config.cell_names_mapping= cell_names_mapping
# %%
# --- Load data ---
data_path = f"example_data/{config.data_name}/"


# Load data using the new function
adata = load_spatial_data_csv(
    data_path=data_path,
    main_data_file="data.csv",
    celltype_file="celltype.csv", 
    pos_file="pos.csv",
    domain_file="domain.csv",
    config=config,
    truth_column_name=config.name_truth,
    index_col=0,
    first_column_names=True
)

# clean the cell ID to save token
# Rename the obs_names of adata
adata.obs_names = [f'cell_{i}' for i in range(len(adata.obs_names))]

# %%
# Prepare neighbor data 
neighbor_normalized_df, neighbor_normalized_df_genes, adj_matrix = prepare_neighbor_data(
    adata, config, representetive_gene_list
)


# %%
# shorten the prompt
unique_indices, idx_mapping, inverse_mapping = get_unique_prompts(
    neighbor_normalized_df, config, prompt_func, df_extra=neighbor_normalized_df_genes)

# %%
# batch run with openai api
generate_json_end2end(neighbor_normalized_df.iloc[unique_indices], 
                      config, 
                      prompt_func, 
                      df_extra=neighbor_normalized_df_genes.iloc[unique_indices],
                      batch_size = 5000,
                      max_completion_tokens = 512,  # key to control the cost, expecially for o3-mini
                      n_rows = 1)

cmd = f"nohup python -u src/submit_end2end.py model_config/config_zeroshot_example.yaml {config.data_name} {config.replicate} > outs/{config.data_name}_zeroshot{config.replicate}.out 2>&1 &"
subprocess.run(cmd, check=True, text=True, shell=True)



# %%
# retrieve the results from openai api
gpt_results_df = process_batch_results(config, n_batch=1)
# restore the full labels if use unique_df
gpt_results_df = gpt_results_df.iloc[unique_indices]  # make sure the order of the index is the same as the unique_df
# restore the full labels
gpt_results_df = restore_full_labels_df(gpt_results_df, inverse_mapping)



# %% 
# ## check results

gpt_results_df.index = gpt_results_df.index.astype(str)
gpt_results_df.index.difference(adata.obs.index)
# check the value counts of the results, if some results are
gpt_results_df.zeroshot_gpt4o_mini.value_counts()


gpt_results_df.columns = ['zeroshot_gpt4o_mini']



# %% 
# # plot

# %%

adata.obs = adata.obs.join(gpt_results_df)
adata.obs['zeroshot_gpt4o_mini'] = adata.obs['zeroshot_gpt4o_mini'].fillna("unknown")
sc.pl.scatter(adata, x="x", y="y", color="zeroshot_gpt4o_mini", title =  f"zeroshot_gpt4o_mini")
print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['zeroshot_gpt4o_mini']))


# %% 
# ## refine the results using the neighbor matrix, this is optional
adata.obs['zeroshot_gpt4o_mini_refined'] = relabel_cells(adj_matrix.toarray(), adata.obs['zeroshot_gpt4o_mini'])
print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['zeroshot_gpt4o_mini_refined']))
#  save
adata.obs.to_csv(f"results/{config.data_name}_with_refined_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}{config.replicate}.csv")




# %% 
# # use Gemini

genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-pro")
gen_config=genai.types.GenerationConfig(temperature=1.0, max_output_tokens=2000)

# %%
gemini_results_df, store_responses = run_gemini(model, gen_config, 
                                                neighbor_normalized_df, config, 
                                                prompt_func, n_rows=1, 
                                                df_extra=neighbor_normalized_df_genes,
                                                column_name="zeroshot_gemini")

# %%
with open(f'./gemini_results/{config.data_name}_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}_{config.confident_output}{config.replicate}.pkl', 'wb') as file:
    pickle.dump(store_responses, file)

gemini_results_df.to_csv(f"./gemini_results/{config.data_name}_results_df_{config.model_type}_{config.use_full_name}_{config.with_self_type}_{config.with_region_name}_{config.Graph_type}_{config.with_negatives}_{config.with_CoT}_{config.with_numbers}_{config.with_domain_name}_{config.confident_output}{config.replicate}.csv")


# %%
gemini_results_df.index = gemini_results_df.index.astype(str)

gemini_results_df.index.difference(adata.obs.index)

gemini_results_df.zeroshot_gemini.value_counts()

adata.obs = adata.obs.join(gemini_results_df)
adata.obs['zeroshot_gemini'] = adata.obs['zeroshot_gemini'].fillna("unknown")
sc.pl.scatter(adata, x="x", y="y", color="zeroshot_gemini", title =  f"zeroshot_gemini")
print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['zeroshot_gemini']))

adata.obs['zeroshot_gemini_refined'] = relabel_cells(adj_matrix.toarray(), adata.obs['zeroshot_gemini'])
print(adjusted_rand_score(adata.obs[config.name_truth], adata.obs['zeroshot_gemini_refined']))