from utils import sig_figs, extract_output_microenvironments
import json
import os
import time
import pandas as pd
from utils import get_farthest_cell_types

def zeroshot_geneorder(df, rows, config):
    """
    Transfer multiple rows of a neighbor dataframe to a list of strings
    df: the dataframe of neighbor_matrix, with cells in rows and genes in columns
    rows: the rows id (int) of the dataframe to be transferred to strings
    """
    df = df.copy()
    strings_list = []
    for i in rows:
        row = df.iloc[i]
        
        # Sort cell types by count in descending order
        sorted_cells = row.sort_values(ascending=False)[:config.top_n]
        
        one_frequency = []
        for gene_name, expression in sorted_cells.items():
            if expression <= config.minimal_f:
                continue
            expression = sig_figs(expression, 3)
            if config.with_numbers:
                one_frequency.append(f"{gene_name}: {expression}")
            else:
                one_frequency.append(f"{gene_name}")
        
        one_frequency = ", ".join(one_frequency)
        strings_list.append(f"{{{one_frequency}}}")
    
    multi_rows_strings = ";\n ".join(strings_list)
    
    if config.output_type == "niche":
        result = f"""You will be provided with a list of marker genes expressed in the neighborhood of a cell.{config.region1} \
The genes are ordered from highest expression to lowest expression.{config.text_expression_numbers} \
Your task is to identify the microenvironments the cell belong to. Consider the possiblily existing cell types in the neighborhood based on the marker genes.  \
Each cell can only belong to one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}. \
Only output the most possible microenvironment in the plain text format like this: Outputs: {{'microenvironment 1'}} \
Below are the list of marker genes in the cell's neighborhood:\n{multi_rows_strings}\n"""
    elif config.output_type == "celltype":
        result = f"""In spatial transcriptomic data, one spot contains few cells. \
Your task is to infer the combinations of different cell types based on the marker genes and identify the types of cells existing in the neighborhood of a target spot. \
The cell types should be broken down into subtypes of the major cell types, such as {", ".join(config.cell_names_mapping.keys())}. \
The output should be a list of cell types in the order from highest potential to the least, following the plain text format: Outputs: {{'Cell ID': 'cell_type_1, cell_type_2, ...', 'Cell ID': 'cell_type_1, cell_type_2, ...', ...}}\nOnly provide the list of cell types for each spot ID in the specified output format. \
Below are the list of marker genes in spots' neighborhood.{config.text_expression_numbers} Each row is one spot:\n{multi_rows_strings}"""
    return result


def zeroshot_celltype(df, rows, config):
    """
    Transfer multiple rows of a dataframe to a list of strings
    df: the dataframe, with cells in rows and cell types in columns
    rows: the rows of the dataframe to be transferred to strings
    """
    df = df.copy()
    strings_list = []
    negatives = ""
    for i in rows:
        row = df.iloc[i]
        
        # Sort cell types by count in descending order
        sorted_cells = row.sort_values(ascending=False)[:config.top_n]
        
        one_frequency = []
        
        # Iterate over sorted cell types
        for cell_type, count in sorted_cells.items():
            if count <= config.minimal_f:
                continue
            count = sig_figs(count, 3)
            if config.use_full_name:
                cellname = config.cell_names_mapping[cell_type]
            else:
                cellname = cell_type
            if config.with_numbers:
                one_frequency.append(f"{cellname}: {count}")
            else:
                one_frequency.append(f"{cellname}")
        
        one_frequency = ", ".join(one_frequency)
        strings_list.append(f"{{{one_frequency}}}")

        if config.with_negatives and len(rows)==1:
            negatives = f" The target cell is far from {get_farthest_cell_types(config.pos_data, config.celltype_data, i, config, n=3)}, making it unlikely to belong to the same microenvironment as these cells."
        else:
            negatives = ""
    multi_rows_strings = ";\n ".join(strings_list)
    result = f"""You are tasked with identifying the type of microenvironment of a given cell using the provided cell types in the neighborhood of the given cell. \
Cell types are ordered from most frequent to least frequent.{config.text_frequencies}{config.region1} \
You should consider how different combinations of cell types form microenvironments. \
The target cell can only be one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}.{negatives} \
Below are the cell's neighbor cell-types.\n """
    result += f" {multi_rows_strings}\n  Only output the most possible microenvironment in the plain text like this: Outputs: {{'microenvironment 1'}} "

    return(result)


def zeroshot_celltype_geneorder(df_cell, df_gene, rows, config):
    df_cell = df_cell.copy()
    df_gene = df_gene.copy()
    strings_list = []
    for i in rows:
        cell_row = df_cell.iloc[i]
        gene_row = df_gene.iloc[i]
        sorted_cells = cell_row.sort_values(ascending=False)[:config.top_n]
        sorted_genes = gene_row.sort_values(ascending=False)[:config.top_n]

        cell_frequency = []
        # Iterate over sorted cell types
        for cell_type, count in sorted_cells.items():
            if count <= config.minimal_f:
                continue
            count = sig_figs(count, 3)
            if config.use_full_name:
                cell_name = config.cell_names_mapping[cell_type]
            else:
                cell_name = cell_type
            if config.with_numbers:
                cell_frequency.append(f"{cell_name}: {count}")
            else:
                cell_frequency.append(f"{cell_name}")
        cell_frequency = ", ".join(cell_frequency)
        
        gene_frequency = []
        for gene, count in sorted_genes.items():
            if count <= config.minimal_gene_threshold:
                continue
            count = sig_figs(count, 3)
            if config.with_numbers:
                gene_frequency.append(f"{gene}: {count}")
            else:
                gene_frequency.append(f"{gene}")
        gene_frequency = ", ".join(gene_frequency)

        strings_list.append(f"Neighbor cell-types: {{{cell_frequency}}}, Marker genes: {{{gene_frequency}}}")

    multi_rows_strings = ";\n ".join(strings_list)

    result = f"""You will be provided with a list of cell types in the neighborhood of a cell and marker genes expressed in the neighborhood.{config.region1} \
The Neighbor cell-types are ordered from most frequent to least frequent.{config.text_frequencies} \
The marker genes are ordered from highest expression to lowest expression.{config.text_expression_numbers} \
Your task is to identify the microenvironments the cell belong to. \
The cell can only belong to one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}. \
Only output the most possible microenvironment in the plain text format like this: Outputs: {{'microenvironment 1'}} \
Below are the list of cell-types and marker genes in the cell's neighborhood:\n{multi_rows_strings}\n"""
    
    return result


def CP_geneorder(df, config):
    """
    Comparison-based Prompting
    Use training data to get "mean sample" to represent niches
    Need domain_mapping in the global env
    if the df is a dataframe of genes, then the cell_type is the gene name
    """
    df = df.copy()
    typical_frequencies = []
    # Select the sample row
    for niche_id, niche_name in config.domain_mapping.items():
        row = df.loc[niche_name]
        
        # Sort cell types by count in descending order
        sorted_cells = row.sort_values(ascending=False)[:config.top_n]
        
        one_frequency = []
        
        # Iterate over sorted cell types
        for gene, count in sorted_cells.items():
            if count <= config.minimal_f:
                continue
            count = sig_figs(count, 3)
            if config.with_numbers:
                one_frequency.append(f"{gene}: {count}")
            else:
                one_frequency.append(f"{gene}")
        
        one_frequency = ", ".join(one_frequency)
        typical_frequencies.append(f" Microenvironment: {{{niche_name}}}, Marker genes expressed in the neighborhood: {{{one_frequency}}}")
    
    one_shot = ";\n ".join(typical_frequencies)
    # Combine the results
    result = f"""You will be provided with a list of distinct microenvironments{config.region2}. \
The cell type marker genes expressed in the neighborhood of those microenvironments will be shown. The genes are ordered from highest expression to lowest expression.{config.text_expression_numbers} \
Below is the list:\n {one_shot}\n """
    result += f"""Comparing with the provided list, identify which microenvironment category the following cell belongs to. \
Consider the microenvironment-specific genes of the provided representative cells. \
The target cell can only be one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}. \
Only output the most possible microenvironment in the plain text like this: Outputs: {{'microenvironment 1'}}. \
Below are genes expressed in the neighborhood of the cell:\n"""
    
    return(result)



def oneshot_geneorder(df, rows, config):
    """
    Transfer multiple rows of a sparse neighbor_matrix to a list of strings
    df: the dataframe of neighbor_matrix, with cells in rows and genes in columns
    rows: the rows of the sparse matrix to be transferred to strings
    """
    df = df.copy()
    strings_list = []
    for i in rows:
        row = df.iloc[i]
        
        # Sort cell types by count in descending order
        sorted_cells = row.sort_values(ascending=False)[:config.top_n]
        
        one_frequency = []
        for gene_name, expression in sorted_cells.items():
            if expression <= config.minimal_f:
                continue
            expression = sig_figs(expression, 3)
            if config.with_numbers:
                one_frequency.append(f"{gene_name}: {expression}")
            else:
                one_frequency.append(f"{gene_name}")

        one_frequency = ", ".join(one_frequency)
        strings_list.append(f"{{{one_frequency}}}")
    
    multi_rows_strings = ";\n ".join(strings_list)
    
    if config.output_type == "niche":
        result = f" {multi_rows_strings}\n Please identify the microenvironment category for the cell. Only output the most possible microenvironment in the plain text like this: Outputs: {{'microenvironment 1'}} "
    elif config.output_type == "celltype":
        result = f" {multi_rows_strings}\n Please identify the cell types for each spot. Only output the cell types in the plain text like this: Outputs: {{'Cell ID': 'cell_type_1, cell_type_2, ...', 'Cell ID': 'cell_type_1, cell_type_2, ...', ...}} "
    
    return result




def CP_celltype(df, config):
    """
    Comparison-based Prompting
    Use training data to get "mean sample" to represent niches
    Need domain_mapping in the global env
    if the df is a dataframe of genes, then the cell_type is the gene name
    """
    df = df.copy()
    typical_frequencies = []
    # Select the sample row
    for niche_id, niche_name in config.domain_mapping.items():
        row = df.loc[niche_name]
        
        # Sort cell types by count in descending order
        sorted_cells = row.sort_values(ascending=False)[:config.top_n]
        
        one_frequency = []
        
        # Iterate over sorted cell types
        for cell_type, count in sorted_cells.items():
            if count <= config.minimal_f:
                continue
            if config.use_full_name:
                cell_name = config.cell_names_mapping[cell_type]
            else:
                cell_name = cell_type
            count = sig_figs(count, 3)
            if config.with_numbers:
                one_frequency.append(f"{cell_name}: {count}")
            else:
                one_frequency.append(f"{cell_name}")
        
        one_frequency = ", ".join(one_frequency)
        typical_frequencies.append(f" Microenvironment: {{{niche_name}}}, Neighbor cell-type: {{{one_frequency}}}")
    
    one_shot = ";\n ".join(typical_frequencies)
    # Combine the results
    result = f"""You will be provided with a list of distinct microenvironments{config.region2}. \
The cell types in the neighborhood of those microenvironments will be shown{config.text_frequencies}. \
The cell types are ordered from most frequent to least frequent.\n Below is the list:\n {one_shot}\n """
    result += f"Based on the provided list, identify which microenvironment category the following cell belongs to. \
Consider how the different combinations of cell types form microenvironments. \
The target cell can only be one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}. \
Only output the most possible microenvironment in the plain text like this: Outputs: {{'microenvironment 1'}}. \
Below are cell types in the neighborhood of the cell:\n"
    return(result)



def oneshot_celltype(df, rows, config):
    """
    Transfer multiple rows of a sparse neighbor_matrix to a list of strings
    df: the dataframe of neighbor_matrix, with cells in rows and genes in columns
    rows: the rows of the sparse matrix to be transferred to strings
    """
    df = df.copy()
    strings_list = []
    for i in rows:
        row = df.iloc[i]
        
        # Sort cell types by count in descending order
        sorted_cells = row.sort_values(ascending=False)[:config.top_n]
        
        one_frequency = []
        for cell_type, count in sorted_cells.items():
            if config.use_full_name:
                cell_name = config.cell_names_mapping[cell_type]
            else:
                cell_name = cell_type
            if count <= config.minimal_f:
                continue
            count = sig_figs(count, 3)
            if config.with_numbers:
                one_frequency.append(f"{cell_name}: {count}")
            else:
                one_frequency.append(f"{cell_name}")

        one_frequency = ", ".join(one_frequency)
        strings_list.append(f"{{{one_frequency}}}")
    
    multi_rows_strings = ";\n ".join(strings_list)
    

    result = f" {multi_rows_strings}\n Please identify the microenvironment category for each cell. Only output the most possible microenvironment in the plain text like this: Outputs: {{'microenvironment 1'}} "
    
    return result



def CP_celltype_geneorder(df, config):
    df_cell = df[config.cell_names].copy()
    df_gene = df[config.gene_names].copy()
    typical_frequencies = []
    for niche_id, niche_name in config.domain_mapping.items():
        row_cell = df_cell.loc[niche_name]
        row_gene = df_gene.loc[niche_name]

        # Sort cell types by count in descending order
        sorted_cells = row_cell.sort_values(ascending=False)[:config.top_n]
        sorted_genes = row_gene.sort_values(ascending=False)[:config.top_n]

        cell_frequency = []
        # Iterate over sorted cell types
        for cell_type, count in sorted_cells.items():
            if count <= config.minimal_f:
                continue
            count = sig_figs(count, 3)
            if config.use_full_name:
                cell_name = config.cell_names_mapping[cell_type]
            else:
                cell_name = cell_type
            if config.with_numbers:
                cell_frequency.append(f"{cell_name}: {count}")
            else:
                cell_frequency.append(f"{cell_name}")
        cell_frequency = ", ".join(cell_frequency)
        
        gene_frequency = []
        for gene, count in sorted_genes.items():
            if count <= config.minimal_gene_threshold:
                continue
            count = sig_figs(count, 3)
            if config.with_numbers:
                gene_frequency.append(f"{gene}: {count}")
            else:
                gene_frequency.append(f"{gene}")
        gene_frequency = ", ".join(gene_frequency)

        typical_frequencies.append(f"Microenvironment: {{{niche_name}}}, Neighbor cell-types: {{{cell_frequency}}}, marker genes: {{{gene_frequency}}}")

    one_shot = ";\n ".join(typical_frequencies)
    result = f"""You will be provided with a list of tissue microenvironments{config.region2}. \
The cell types and marker genes in the neighborhood of those microenvironments will be shown. \
The neighbor cell-types are ordered from most frequent to least frequent.{config.text_frequencies} \
The genes are ordered from highest expression to lowest expression.{config.text_expression_numbers} \
\n Below are typical examples of microenvironments:\n {one_shot} \n"""
    result += f"Based on the examples, identify which microenvironment categories the following cell belongs to. \
The target cell can only be one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}. \
Only output the most possible microenvironment in the plain text like this: Outputs: {{'microenvironment 1'}}. \
Below are cell types and marker genes in the neighborhood of the cell:\n"
    return(result)


def oneshot_celltype_geneorder(df_cell, df_gene, rows, config):
    df_cell = df_cell.copy()
    df_gene = df_gene.copy()
    strings_list = []
    for i in rows:
        row_cell = df_cell.iloc[i]
        row_gene = df_gene.iloc[i]

        sorted_cells = row_cell.sort_values(ascending=False)[:config.top_n]
        sorted_genes = row_gene.sort_values(ascending=False)[:config.top_n]

        cell_frequency = []
        # Iterate over sorted cell types
        for cell_type, count in sorted_cells.items():
            if count <= config.minimal_f:
                continue
            count = sig_figs(count, 3)
            if config.use_full_name:
                cell_name = config.cell_names_mapping[cell_type]
            else:
                cell_name = cell_type
            if config.with_numbers:
                cell_frequency.append(f"{cell_name}: {count}")
            else:
                cell_frequency.append(f"{cell_name}")
        cell_frequency = ", ".join(cell_frequency)
        
        gene_frequency = []
        for gene, count in sorted_genes.items():
            if count <= config.minimal_gene_threshold:
                continue
            count = sig_figs(count, 3)
            if config.with_numbers:
                gene_frequency.append(f"{gene}: {count}")
            else:
                gene_frequency.append(f"{gene}")
        gene_frequency = ", ".join(gene_frequency)

        strings_list.append(f"Neighbor cell-types: {{{cell_frequency}}}, Marker genes: {{{gene_frequency}}}")

    rows_strings = ";\n ".join(strings_list)

    return rows_strings






def finetune_system_deconv(config):
    return f"""You are tasked with identifying the type of microenvironment of a given cell using the provided cell types in the neighborhood of the given cell. \
Cell types are ordered from most frequent to least frequent.{config.text_frequencies}{config.region1} \
You should consider how different combinations of cell types form microenvironments. \
The target cell can only be one of the {len(config.domain_mapping)} microenvironments: {', '.join(config.domain_mapping.values())}. \
Only output the most possible microenvironment in the plain text like this: Outputs: {{'microenvironment'}} \
Below are the cell's neighbor cell-types:\n 
"""


def finetune_user_deconv(df, i, config):
    """
    df: the dataframe of neighbor_matrix, with cells in rows and genes in columns
    row: the row of the sparse matrix to be transferred to strings
    """
    df = df.copy()
    if isinstance(i, list):
        i = i[0]
    row = df.iloc[i]
    sorted_cells = row.sort_values(ascending=False)[:config.top_n]
    one_frequency = []
    for cell_type, count in sorted_cells.items():
        if count <= config.minimal_f:
            continue
        count = sig_figs(count, 3)
        if config.use_full_name:
            cell_name = config.cell_names_mapping[cell_type]
        else:
            cell_name = cell_type
        if config.with_numbers:
            one_frequency.append(f"{cell_name}: {count}")
        else:
            one_frequency.append(f"{cell_name}")

    one_frequency = ", ".join(one_frequency)

    return f"""{{{one_frequency}}}"""


def finetune_assistant(df, i, niche_df):
    if isinstance(i, list):
        i = i[0]
    id = df.index[i]
    niche = niche_df.loc[id]
    return f"Outputs: {{'{niche}'}}"

def finetune_user_celltype_geneorder(df_cell, df_gene, i, config):
    """
    df_cell: the dataframe of neighbor_matrix, with cells in rows and genes in columns
    df_gene: the dataframe of neighbor_matrix, with cells in rows and genes in columns
    row: the row of the sparse matrix to be transferred to strings
    """
    df_cell = df_cell.copy()
    df_gene = df_gene.copy()
    if isinstance(i, list):
        i = i[0]

    row_cell = df_cell.iloc[i]  
    row_gene = df_gene.iloc[i]
    sorted_cells = row_cell.sort_values(ascending=False)[:config.top_n]
    sorted_genes = row_gene.sort_values(ascending=False)[:config.top_n]

    cell_frequency = []
        # Iterate over sorted cell types
    for cell_type, count in sorted_cells.items():
        if count <= config.minimal_f:
            continue
        count = sig_figs(count, 3)
        if config.use_full_name:
            cell_name = config.cell_names_mapping[cell_type]
        else:
            cell_name = cell_type
        if config.with_numbers:
            cell_frequency.append(f"{cell_name}: {count}")
        else:
            cell_frequency.append(f"{cell_name}")
    cell_frequency = ", ".join(cell_frequency)
    
    gene_frequency = []
    for gene, count in sorted_genes.items():
        if count <= config.minimal_gene_threshold:
            continue
        count = sig_figs(count, 3)
        if config.with_numbers:
            gene_frequency.append(f"{gene}: {count}")
        else:
            gene_frequency.append(f"{gene}")
    gene_frequency = ", ".join(gene_frequency)

    negative_text = ""
    if config.with_negatives:
        if config.neg_top_3_genes is not None:
            # get the least expressed genes 
            least_expressed_genes = row_gene.sort_values(ascending=True)[:config.top_n].index.tolist()
            negative_niche_names = []
            for gene in least_expressed_genes:
                # Check each row in config.neg_top_3_genes
                for idx, gene_list in config.neg_top_3_genes.items():
                    if gene in gene_list:  # Check if gene is in this list
                        negative_niche_names.append(idx)

            # Remove duplicates if any
            negative_niche_names = list(set(negative_niche_names))
            negative_text = f". Because other genes are not expressed, the cell is unlikely to belong to these microenvironments: {negative_niche_names}"
        else:
            raise ValueError("neg_top_3_genes is not defined")
    else:
        # if neg_top_3_genes is in the attributes of config, raise error
        if hasattr(config, 'neg_top_3_genes'):
            raise ValueError("neg_top_3_genes is defined but with_negatives is False")

    return(f"Neighbor cell-types: {{{cell_frequency}}}, Marker genes: {{{gene_frequency}}}{negative_text}")






