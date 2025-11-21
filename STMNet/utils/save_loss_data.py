import json
from pathlib import Path
from typing import Dict, Union, List, Any
import numpy as np


def save_loss_data(
        loss_dict: Dict[str, Union[List[float], np.ndarray]],
        file_path: Union[str, Path],
        metadata: Dict[str, Any] = None
) -> None:
    """
    Save training loss data to JSON file with proper serialization

    Args:
        loss_dict: Dictionary containing loss data with keys as phase names and values as loss arrays
            Example: {'train': [0.5, 0.4, ...], 'valid': [0.6, 0.55, ...]}
        file_path: Path where the JSON file will be saved (parent directories will be created automatically)
        metadata: Optional dictionary containing additional metadata like hyperparameters, timestamps, etc.

    Raises:
        ValueError: If loss_dict is empty or contains invalid data types
        IOError: If there are issues creating directories or writing the file
    """
    # Validate input data
    if not loss_dict:
        raise ValueError("Loss dictionary cannot be empty")

    # Convert numpy arrays to Python native types for JSON serialization
    safe_data = {}
    for key, values in loss_dict.items():
        try:
            if isinstance(values, np.ndarray):
                safe_data[key] = values.tolist()
            elif isinstance(values, list):
                safe_data[key] = [float(x) for x in values]
            else:
                raise TypeError(f"Unsupported data type for key '{key}': {type(values)}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to process loss data for key '{key}': {str(e)}")

    # Add metadata with proper serialization
    if metadata:
        safe_data['metadata'] = {}
        for key, value in metadata.items():
            try:
                # Convert various types to string for safe serialization
                if isinstance(value, (int, float, str, bool, type(None))):
                    safe_data['metadata'][key] = value
                else:
                    safe_data['metadata'][key] = str(value)
            except Exception as e:
                print(f"Warning: Could not serialize metadata key '{key}': {str(e)}")
                continue

    # Add timestamp if not already present in metadata
    if metadata is None or 'timestamp' not in metadata:
        from datetime import datetime
        safe_data.setdefault('metadata', {})['timestamp'] = datetime.now().isoformat()

    # Create parent directories if they don't exist
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Failed to create directory {path.parent}: {str(e)}")

    # Write data to JSON file with proper error handling
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(safe_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Loss data successfully saved to: {path}")
    except IOError as e:
        raise IOError(f"Failed to write to file {path}: {str(e)}")
    except TypeError as e:
        raise ValueError(f"Data serialization error: {str(e)}")


def load_loss_data(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load loss data from JSON file

    Args:
        file_path: Path to the JSON file containing loss data

    Returns:
        Dictionary containing the loaded loss data and metadata

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        JSONDecodeError: If the file contains invalid JSON
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Loss data file not found: {path}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loss data successfully loaded from: {path}")
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in {path}: {str(e)}")
    except IOError as e:
        raise IOError(f"Failed to read file {path}: {str(e)}")


def compare_loss_curves(file_paths: List[Union[str, Path]],
                        keys: List[str] = None) -> Dict[str, Any]:
    """
    Compare loss curves from multiple experiment files

    Args:
        file_paths: List of paths to loss data files
        keys: Specific loss keys to compare (if None, compares all common keys)

    Returns:
        Dictionary containing comparative analysis of loss curves
    """
    comparison_data = {}

    for file_path in file_paths:
        try:
            data = load_loss_data(file_path)
            experiment_name = Path(file_path).stem

            # Extract loss data for comparison
            comparison_data[experiment_name] = {
                'losses': {k: v for k, v in data.items() if k != 'metadata'},
                'metadata': data.get('metadata', {})
            }
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {str(e)}")
            continue

    return comparison_data