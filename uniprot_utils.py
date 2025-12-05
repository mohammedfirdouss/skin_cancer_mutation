import json
import requests
from pathlib import Path
from typing import Dict, Optional

class UniProtCache:
    """Creating Cached access to UniProt protein database"""

    def __init__(self, cache_dir="./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "uniprot_cache.json"
        self.cache = self._load_cache()

        # Listing some common skin cancer proteins
        self.cancer_proteins = [
            'BRAF', 'TP53', 'NRAS', 'CDKN2A', 'PTEN',
            'KIT', 'NF1', 'MAP2K1', 'TERT', 'ARID2'
        ]

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)

    def fetch_protein_info(self, gene_name: str) -> Optional[Dict]:
        """Fetch protein info from UniProt (cached)"""
        if gene_name in self.cache:
            return self.cache[gene_name]

        try:
            # APi request to extract the protein features of skin cancer
            url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_id:9606&format=json&size=1"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    info = {
                        'gene': gene_name,
                        'protein_name': result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
                        'function': result.get('comments', [{}])[0].get('texts', [{}])[0].get('value', 'No function info'),
                        'accession': result.get('primaryAccession', ''),
                        'sequence_length': result.get('sequence', {}).get('length', 0)
                    }
                    self.cache[gene_name] = info
                    self._save_cache()
                    return info
        except Exception as e:
            print(f"Error fetching {gene_name}: {e}")

        return None

    def preload_cancer_proteins(self):
        """Preloading the common skin cancer proteins"""
        print("Preloading cancer protein database...")
        for protein in self.cancer_proteins:
            if protein not in self.cache:
                print(f"  Fetching {protein}...")
                self.fetch_protein_info(protein)
        print(f"Cached {len(self.cache)} proteins")
