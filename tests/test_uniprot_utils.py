"""
Unit tests for the UniProtCache utility in uniprot_utils.py.
"""

import pytest
from unittest.mock import MagicMock
from uniprot_utils import UniProtCache

# Sample successful API response from UniProt
mock_api_response = {
    "results": [
        {
            "primaryAccession": "P15056",
            "proteinDescription": {
                "recommendedName": {
                    "fullName": {
                        "value": "Proto-oncogene B-Raf"
                    }
                }
            },
            "comments": [
                {
                    "commentType": "FUNCTION",
                    "texts": [
                        {
                            "value": "Protein kinase involved in the ERK1/2 signaling pathway."
                        }
                    ]
                }
            ],
            "sequence": {
                "length": 766
            }
        }
    ]
}

@pytest.fixture
def mock_requests_get(mocker):
    """Fixture to mock requests.get."""
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_api_response
    
    # Patch requests.get to return our mock response
    return mocker.patch("requests.get", return_value=mock_response)

def test_fetch_protein_info_success(mock_requests_get):
    """
    Test that fetch_protein_info successfully parses a response from the API.
    """
    # Use a temporary cache file for isolation
    cache = UniProtCache(cache_dir="./tests/temp_cache")
    gene_name = "BRAF"

    # Fetch the info
    info = cache.fetch_protein_info(gene_name)

    # --- Assertions ---
    # 1. Check that the API was called correctly
    mock_requests_get.assert_called_once_with(
        f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_id:9606&format=json&size=1",
        timeout=10
    )

    # 2. Check that the returned info is correct
    assert info is not None
    assert info['gene'] == gene_name
    assert info['protein_name'] == "Proto-oncogene B-Raf"
    assert "Protein kinase involved" in info['function']
    assert info['accession'] == "P15056"

    # 3. Check that the info was saved to the cache
    assert gene_name in cache.cache

def test_fetch_protein_info_from_cache(mock_requests_get):
    """
    Test that fetch_protein_info returns cached data without making a new API call.
    """
    cache = UniProtCache(cache_dir="./tests/temp_cache")
    gene_name = "BRAF"

    # Pre-populate the cache
    cached_info = {"gene": "BRAF", "protein_name": "Cached B-Raf"}
    cache.cache[gene_name] = cached_info

    # Fetch the info
    info = cache.fetch_protein_info(gene_name)

    # --- Assertions ---
    # 1. Check that the API was NOT called
    mock_requests_get.assert_not_called()

    # 2. Check that the returned info is the cached data
    assert info is not None
    assert info['protein_name'] == "Cached B-Raf"
