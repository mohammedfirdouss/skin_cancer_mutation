import json
import torch
from pathlib import Path
from typing import Any, List, Dict

from llama_index.core import Document, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

from datasets import load_dataset
from transformers import BitsAndBytesConfig

from uniprot_utils import UniProtCache

# --- 1. Data Loading ---

class MolInstructionsLoader:
    """Loads and filters data from Mol-Instructions, returning LlamaIndex Documents."""

    def __init__(self, cache_dir="./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.filtered_file = self.cache_dir / "cancer_filtered.json"

    def load_documents(self, max_samples=2000):
        """Loads filtered data and converts them into LlamaIndex Document objects."""
        data = self._get_filtered_data(max_samples)
        documents = []
        print("Converting to LlamaIndex Documents...")
        for item in data:
            text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
            metadata = {
                "source": "Mol-Instructions",
                "id": item.get('id', 'N/A'),
                "task": "mutation_analysis"
            }
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)
        print(f"Created {len(documents)} documents.")
        return documents

    def _get_filtered_data(self, max_samples):
        """Downloads and filters the dataset, caching the result."""
        if self.filtered_file.exists():
            print("Loading cached filtered data...")
            with open(self.filtered_file, 'r') as f:
                return json.load(f)

        print("Downloading and filtering Mol-Instructions dataset...")
        try:
            dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", split="train", streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        cancer_keywords = ['cancer', 'tumor', 'mutation', 'melanoma', 'braf', 'tp53', 'v600e', 'carcinoma', 'oncogene']
        filtered_data = []
        count = 0
        for example in dataset:
            if count >= max_samples:
                break
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

# --- 2. Model and Settings Configuration ---

def _load_models():
    """Initializes the Embedding model and a 4-bit quantized LLM."""
    print("Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Loading quantized LLM...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFaceLLM(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        tokenizer_name="unsloth/Llama-3.2-1B-Instruct",
        context_window=2048,
        max_new_tokens=512,
        model_kwargs={"quantization_config": quantization_config},
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        device_map="cpu",  # Set to "auto" for GPU
    )
    return embed_model, llm

# --- 3. Custom RAG Query Engine with UniProt Enrichment ---

class UniProtEnrichedQueryEngine(BaseQueryEngine):
    """
    A custom query engine that enriches the context with UniProt data
    before generating an answer.
    """
    def __init__(self, retriever, llm, prompt_template):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.uniprot_cache = UniProtCache()
        super().__init__()

    def _extract_proteins(self, text: str) -> List[str]:
        """A simple method to extract potential protein names from text."""
        # This can be improved with more sophisticated entity recognition
        known_proteins = self.uniprot_cache.cancer_proteins
        text_upper = text.upper()
        return [p for p in known_proteins if p in text_upper]

    def _build_context(self, retrieved_nodes, protein_info) -> str:
        """Combines retrieved documents and UniProt info into a single context string."""
        context_parts = []
        if retrieved_nodes:
            context_parts.append("### Retrieved Scientific Information:")
            for i, node in enumerate(retrieved_nodes, 1):
                context_parts.append(f"{i}. {node.get_text()[:350]}...")
        
        if protein_info:
            context_parts.append("\n### Protein Database Information (from UniProt):")
            for protein in protein_info:
                context_parts.append(
                    f"- **{protein['gene']}**: {protein['protein_name']}\n"
                    f"  *Function*: {protein['function'][:200]}...")
        return "\n".join(context_parts)

    def _query(self, query_str: str) -> Any:
        """The main query logic."""
        print(f"Processing query: {query_str}")

        # 1. Retrieve relevant documents
        print("-> Retrieving documents...")
        retrieved_nodes = self.retriever.retrieve(query_str)
        
        # 2. Extract protein names and fetch data
        print("-> Fetching protein information from UniProt...")
        proteins_mentioned = self._extract_proteins(query_str)
        protein_info = [self.uniprot_cache.fetch_protein_info(p) for p in proteins_mentioned if p]

        # 3. Build the augmented context
        print("-> Building augmented context...")
        context_str = self._build_context(retrieved_nodes, protein_info)
        
        # 4. Format prompt and generate answer
        print("-> Generating answer with LLM...")
        formatted_prompt = self.prompt_template.format(
            context_str=context_str,
            query_str=query_str
        )
        
        response = self.llm.complete(formatted_prompt)
        
        # We can return a structured response if needed, but for now, just the text.
        return {"response": str(response), "source_nodes": retrieved_nodes}


# --- 4. Main Pipeline Initialization ---

def initialize_rag_pipeline(max_samples=500):
    """
    The main function to set up the entire RAG pipeline.
    Returns a custom query engine ready for use.
    """
    # 1. Configure global models
    print("="*50)
    print("INITIALIZING RAG PIPELINE")
    print("="*50)
    embed_model, llm = _load_models()
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 2. Load data and build the vector index
    loader = MolInstructionsLoader()
    documents = loader.load_documents(max_samples=max_samples)
    
    print("Building Vector Index...")
    index = VectorStoreIndex.from_documents(documents)
    
    # 3. Define the retriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

    # 4. Define the prompt template
    qa_prompt_tmpl = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "You are a scientific AI assistant specializing in cancer biology. "
        "Given the context information (which includes retrieved scientific documents and protein data from UniProt), "
        "and not prior knowledge, answer the query in a precise, factual, and concise manner.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    # 5. Create the custom query engine
    print("Creating UniProt-enriched query engine...")
    query_engine = UniProtEnrichedQueryEngine(
        retriever=retriever,
        llm=llm,
        prompt_template=qa_prompt_tmpl
    )
    
    # Preload protein data for faster queries
    query_engine.uniprot_cache.preload_cancer_proteins()
    
    print("\n✅ RAG Pipeline Initialized Successfully!")
    print("="*50)
    
    return query_engine