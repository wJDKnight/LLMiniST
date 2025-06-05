import os
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, List, Dict, Union


def load_spatial_data_csv(
    data_path: str,
    main_data_file: str = "data.csv",
    celltype_file: str = "celltype.csv",
    pos_file: str = "pos.csv",
    domain_file: str = "domain.csv",
    config: Optional[object] = None,
    index_col: int = 0,
    first_column_names: bool = True,
    optional_files: Optional[List[str]] = None,
    truth_column_name: Optional[str] = None
    
) -> ad.AnnData:
    """
    Load spatial transcriptomics data from a folder containing CSV files.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing CSV files
    main_data_file : str, default "data.csv"
        Name of the main expression data file
    celltype_file : str, default "celltype.csv"
        Name of the cell type annotation file
    pos_file : str, default "pos.csv"
        Name of the position/coordinates file
    domain_file : str, default "domain.csv"
        Name of the domain/truth labels file (optional)
    index_col : int, default 0
        Column to use as row index for metadata files
    first_column_names : bool, default True
        Whether first column contains gene names in main data file
    optional_files : List[str], optional
        List of additional CSV files to load and join
    truth_column_name : str, optional
        Name to assign to the truth/domain column
    config : Optional[object], optional
        Additional configuration parameters
        
    Returns:
    --------
    adata : anndata.AnnData
        AnnData object with expression data and metadata
        
    Examples:
    ---------
    >>> # Basic usage
    >>> adata = load_spatial_data_csv("/path/to/data/")
    
    >>> # With custom file names and truth column
    >>> adata = load_spatial_data_csv(
    ...     "/path/to/data/",
    ...     main_data_file="expression.csv",
    ...     truth_column_name="niche_truth"
    ... )
    
    >>> # With additional metadata files
    >>> adata = load_spatial_data_csv(
    ...     "/path/to/data/",
    ...     optional_files=["batch_info.csv", "quality_metrics.csv"]
    ... )
    """
    
    # Validate data path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Load main expression data
    main_file_path = os.path.join(data_path, main_data_file)
    if not os.path.exists(main_file_path):
        raise FileNotFoundError(f"Main data file not found: {main_file_path}")
    
    print(f"Loading main data from: {main_file_path}")
    adata = sc.read_csv(main_file_path, first_column_names=first_column_names)
    
    # Initialize list to store metadata DataFrames
    metadata_dfs = []
    
    # Load cell type data
    celltype_path = os.path.join(data_path, celltype_file)
    if os.path.exists(celltype_path):
        print(f"Loading cell type data from: {celltype_path}")
        celltype_data = pd.read_csv(celltype_path, index_col=index_col)
        
        # If celltype_data has only one column, create a one-hot encoding dataframe
        if celltype_data.shape[1] == 1:
            column_name = config.celltype_name
            celltype_data.columns = [column_name]
            print(f"Converting single celltype column '{column_name}' to one-hot encoding")
            cellprop_data = pd.get_dummies(celltype_data[column_name])
        else:
            # If multiple columns, find cell type with largest value for each row
            print(f"Multi-column celltype data detected. Finding dominant cell type for each cell.")
            cellprop_data = celltype_data
            # Get the column name with the maximum value for each row
            dominant_celltype = celltype_data.idxmax(axis=1)
            
            # Create new single-column dataframe
            column_name = config.celltype_name
            celltype_data = pd.DataFrame({column_name: dominant_celltype}, index=celltype_data.index)
            print(f"Created single celltype column '{column_name}' with dominant cell types")
        
        metadata_dfs.append(celltype_data)
        metadata_dfs.append(cellprop_data)
    else:
        print(f"Warning: Cell type file not found: {celltype_path}")
    
    # Load position data
    pos_path = os.path.join(data_path, pos_file)
    if os.path.exists(pos_path):
        print(f"Loading position data from: {pos_path}")
        pos_data = pd.read_csv(pos_path, index_col=index_col)
        metadata_dfs.append(pos_data)
    else:
        print(f"Warning: Position file not found: {pos_path}")
    
    # Load domain/truth data (optional)
    domain_path = os.path.join(data_path, domain_file)
    if os.path.exists(domain_path):
        print(f"Loading domain data from: {domain_path}")
        domain_data = pd.read_csv(domain_path, index_col=index_col)
        if truth_column_name:
            domain_data.columns = [truth_column_name]
        metadata_dfs.append(domain_data)
    else:
        print(f"Info: Domain file not found (optional): {domain_path}")
    
    # Load additional optional files
    if optional_files:
        for optional_file in optional_files:
            optional_path = os.path.join(data_path, optional_file)
            if os.path.exists(optional_path):
                print(f"Loading optional data from: {optional_path}")
                optional_data = pd.read_csv(optional_path, index_col=index_col)
                metadata_dfs.append(optional_data)
            else:
                print(f"Warning: Optional file not found: {optional_path}")
    
    # Join all metadata to adata.obs
    if metadata_dfs:
        # Remove None values and join
        valid_metadata_dfs = [df for df in metadata_dfs if df is not None]
        if valid_metadata_dfs:
            adata.obs = adata.obs.join(valid_metadata_dfs)
    
    # Clean cell IDs to save tokens (convert to simple integer strings)
    adata.obs_names = list(range(len(adata)))
    adata.obs_names = adata.obs_names.astype(str)
    
    print(f"Data loading complete. Shape: {adata.shape}")
    print(f"Observations (cells): {adata.n_obs}")
    print(f"Variables (genes): {adata.n_vars}")
    print(f"Metadata columns: {list(adata.obs.columns)}")
    
    return adata


def load_spatial_data_flexible(
    data_path: str,
    file_mapping: Dict[str, str],
    index_col: int = 0,
    main_data_first_column_names: bool = True
) -> ad.AnnData:
    """
    More flexible data loading function that accepts a mapping of data types to file names.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing CSV files
    file_mapping : Dict[str, str]
        Dictionary mapping data types to file names
        Required key: 'main_data'
        Optional keys: 'celltype', 'position', 'domain', or any custom names
    index_col : int, default 0
        Column to use as row index for metadata files
    main_data_first_column_names : bool, default True
        Whether first column contains gene names in main data file
        
    Returns:
    --------
    adata : anndata.AnnData
        AnnData object with expression data and metadata
        
    Examples:
    ---------
    >>> file_mapping = {
    ...     'main_data': 'expression_matrix.csv',
    ...     'celltype': 'cell_annotations.csv',
    ...     'position': 'spatial_coordinates.csv',
    ...     'batch': 'batch_info.csv'
    ... }
    >>> adata = load_spatial_data_flexible("/path/to/data/", file_mapping)
    """
    
    if 'main_data' not in file_mapping:
        raise ValueError("file_mapping must contain 'main_data' key")
    
    # Load main data
    main_file_path = os.path.join(data_path, file_mapping['main_data'])
    adata = sc.read_csv(main_file_path, first_column_names=main_data_first_column_names)
    
    # Load metadata files
    metadata_dfs = []
    for data_type, filename in file_mapping.items():
        if data_type == 'main_data':
            continue
            
        file_path = os.path.join(data_path, filename)
        if os.path.exists(file_path):
            print(f"Loading {data_type} data from: {file_path}")
            metadata_df = pd.read_csv(file_path, index_col=index_col)
            metadata_dfs.append(metadata_df)
        else:
            print(f"Warning: {data_type} file not found: {file_path}")
    
    # Join metadata
    if metadata_dfs:
        adata.obs = adata.obs.join(metadata_dfs)
    
    # Clean cell IDs
    adata.obs_names = list(range(len(adata)))
    adata.obs_names = adata.obs_names.astype(str)
    
    return adata


def get_data_info(data_path: str) -> Dict[str, bool]:
    """
    Check what CSV files are available in the data directory.
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory
        
    Returns:
    --------
    file_info : Dict[str, bool]
        Dictionary showing which standard files are present
    """
    
    standard_files = {
        'data.csv': 'main_data',
        'celltype.csv': 'celltype',
        'pos.csv': 'position', 
        'domain.csv': 'domain'
    }
    
    file_info = {}
    
    print(f"Checking files in: {data_path}")
    print("-" * 50)
    
    for filename, description in standard_files.items():
        file_path = os.path.join(data_path, filename)
        exists = os.path.exists(file_path)
        file_info[description] = exists
        status = "✓" if exists else "✗"
        print(f"{status} {filename} ({description})")
    
    # List any other CSV files
    try:
        all_files = os.listdir(data_path)
        csv_files = [f for f in all_files if f.endswith('.csv')]
        other_csvs = [f for f in csv_files if f not in standard_files.keys()]
        
        if other_csvs:
            print("\nOther CSV files found:")
            for f in other_csvs:
                print(f"  • {f}")
    except OSError:
        print(f"Error: Cannot access directory {data_path}")
    
    return file_info 