import json
from pathlib import Path
import torch
from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

class MolInstructionsLoader:
    """Loads and filters data, returning LlamaIndex Documents."""
    
    def __init__(self, cache_dir="./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.filtered_file = self.cache_dir / "cancer_filtered.json"

    def load_documents(self, max_samples=2000):
        data = self._get_filtered_data(max_samples)
        documents = []
        
        print("Converting to LlamaIndex Documents...")
        for item in data:
            # Combine fields into a single text block for the LLM
            text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
            
            # Metadata for filtering and traceability
            metadata = {
                "source": "Mol-Instructions",
                "id": item['id'],
                "task": "mutation_analysis"
            }
            
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)
            
        print(f"Created {len(documents)} documents.")
        return documents

    def _get_filtered_data(self, max_samples):
        if self.filtered_file.exists():
            print("Loading cached data...")
            with open(self.filtered_file, 'r') as f:
                return json.load(f)
        
        print("Downloading and filtering dataset...")
        try:
            # Streaming to avoid downloading the massive full dataset
            dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        cancer_keywords = ['cancer', 'tumor', 'mutation', 'melanoma', 'braf', 'tp53', 'v600e', 'carcinoma', 'oncogene']
        filtered_data = []
        count = 0
        
        for example in dataset:
            if count >= max_samples: break
            
            # Check if keywords exist in instruction or output
            text = f"{example.get('instruction', '')} {example.get('output', '')}".lower()
            if any(k in text for k in cancer_keywords):
                filtered_data.append({
                    'instruction': example.get('instruction', ''),
                    'input': example.get('input', ''),
                    'output': example.get('output', ''),
                    'id': count
                })
                count += 1
                if count % 100 == 0:
                    print(f"Collected {count} samples...")
        
        with open(self.filtered_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        return filtered_data

def load_models():
    """Initializes the Embedding model and Quantized LLM."""
    print("Loading Embeddings...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Loading Quantized LLM...")
    # 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load Llama-3.2-1B-Instruct
    llm = HuggingFaceLLM(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        tokenizer_name="unsloth/Llama-3.2-1B-Instruct",
        context_window=2048,
        max_new_tokens=512,
        model_kwargs={"quantization_config": quantization_config},
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        device_map="cpu",
    )
    
    return embed_model, llm

def initialize_rag_pipeline(max_samples=500):
    """Sets up the full RAG pipeline and returns the query engine."""
    
    # 1. Load Models
    embed_model, llm = load_models()
    
    # 2. Configure Global Settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # 3. Load Data
    loader = MolInstructionsLoader()
    documents = loader.load_documents(max_samples=max_samples)
    
    # 4. Build Index
    print("Building Vector Index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 5. Create Query Engine
    return index.as_query_engine(similarity_top_k=3)
