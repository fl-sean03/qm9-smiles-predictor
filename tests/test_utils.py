import pytest
import numpy as np

# Assuming standard pytest discovery, 'src' should be in PYTHONPATH or discoverable
from src.hybrid.utils import compute_ecfp4_array, compute_ecfp4_list

def test_compute_ecfp4_array():
    """
    Tests the compute_ecfp4_array function with a single SMILES string.
    """
    smiles = 'CCO'  # Ethanol
    nBits = 1024    # Match default or common usage

    result = compute_ecfp4_array(smiles, nBits=nBits)

    # Check type
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"

    # Check shape
    assert result.shape == (nBits,), f"Result shape should be ({nBits},), but got {result.shape}"

    # Check dtype - ECFP4 typically results in integer counts or boolean presence
    # Using issubdtype is flexible for different integer types (int32, int64) or bool
    assert np.issubdtype(result.dtype, np.integer) or np.issubdtype(result.dtype, np.bool_), \
           f"Result dtype should be integer or boolean, but got {result.dtype}"

    # Optional: Check if *any* bits are set (sanity check for a known molecule)
    # This depends on the specific implementation details of ECFP4 generation
    # assert np.any(result), "Expected some bits to be set for Ethanol ECFP4"


def test_compute_ecfp4_list():
    """
    Tests the compute_ecfp4_list function with a list of SMILES strings.
    """
    smiles_list = ['CCO', 'C']  # Ethanol and Methane
    nBits = 1024

    result = compute_ecfp4_list(smiles_list, nBits=nBits)

    # Check type
    assert isinstance(result, np.ndarray), "Result should be a NumPy array"

    # Check shape
    expected_shape = (len(smiles_list), nBits)
    assert result.shape == expected_shape, f"Result shape should be {expected_shape}, but got {result.shape}"

    # Check dtype
    assert np.issubdtype(result.dtype, np.integer) or np.issubdtype(result.dtype, np.bool_), \
           f"Result dtype should be integer or boolean, but got {result.dtype}"

    # Optional: Check dimensions individually
    assert result.shape[0] == len(smiles_list), "Number of rows should match number of SMILES strings"
    assert result.shape[1] == nBits, f"Number of columns should match nBits ({nBits})"