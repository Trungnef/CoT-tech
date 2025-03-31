"""
File handling utilities.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union, Optional

# Set up logging
logger = logging.getLogger(__name__)

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path: Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def create_timestamp_directory(base_dir: Union[str, Path]) -> Path:
    """
    Create a directory with a timestamp in the name.
    
    Args:
        base_dir: Base directory
    
    Returns:
        Path: Path object for the created directory
    """
    # Create base directory if it doesn't exist
    base_path = ensure_directory(base_dir)
    
    # Create a timestamped directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = base_path / timestamp
    
    # Create the directory
    ensure_directory(result_dir)
    
    return result_dir

def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> str:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save to
        indent: Indentation level for JSON
    
    Returns:
        str: Path to the saved file
    """
    # Ensure the parent directory exists
    ensure_directory(Path(filepath).parent)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Saved JSON data to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        raise

def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to load from
    
    Returns:
        Any: Loaded data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {str(e)}")
        raise

def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], format: str = "csv") -> str:
    """
    Save a pandas DataFrame to a file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save to
        format: File format (csv, excel, pickle)
    
    Returns:
        str: Path to the saved file
    """
    # Ensure the parent directory exists
    ensure_directory(Path(filepath).parent)
    
    try:
        if format.lower() == "csv":
            df.to_csv(filepath, index=False, encoding='utf-8')
        elif format.lower() == "excel":
            df.to_excel(filepath, index=False)
        elif format.lower() == "pickle":
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved DataFrame to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving DataFrame to {filepath}: {str(e)}")
        raise

def load_dataframe(filepath: Union[str, Path], format: Optional[str] = None) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a file.
    
    Args:
        filepath: Path to load from
        format: File format (csv, excel, pickle) - if None, inferred from extension
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    # Determine format from extension if not provided
    if format is None:
        ext = Path(filepath).suffix.lower()
        if ext == '.csv':
            format = 'csv'
        elif ext in ['.xlsx', '.xls']:
            format = 'excel'
        elif ext == '.pkl':
            format = 'pickle'
        else:
            raise ValueError(f"Cannot determine format from extension: {ext}")
    
    try:
        if format.lower() == "csv":
            df = pd.read_csv(filepath, encoding='utf-8')
        elif format.lower() == "excel":
            df = pd.read_excel(filepath)
        elif format.lower() == "pickle":
            df = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded DataFrame from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading DataFrame from {filepath}: {str(e)}")
        raise

def save_config(config: Dict[str, Any], result_dir: Union[str, Path]) -> str:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        result_dir: Directory to save the config file
    
    Returns:
        str: Path to the saved config file
    """
    config_path = Path(result_dir) / "config.json"
    return save_json(config, config_path) 