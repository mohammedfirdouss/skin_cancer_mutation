import yaml
import torch
import logging
from pathlib import Path

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate
)
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter

# LlamaIndex integrations
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore

# Third-party libraries
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import chromadb

# Local utilities
from uniprot_utils import UniProtCache

# --- Configuration Loading ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration globally
CONFIG = load_config()

# --- 1. Data Loading ---

def load_mol_instructions():
    """
    Loads, filters, and caches data from the Mol-Instructions dataset,
    returning a list of LlamaIndex Document objects.
    """
    data_config = CONFIG['data']
    max_samples = data_config['max_samples']
    cache_dir = Path(data_config['cache_dir'])
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    filtered_file = cache_dir / f"cancer_filtered_{max_samples}.json"

    if filtered_file.exists():
        logging.info("Loading cached filtered data...")
        with open(filtered_file, 'r') as f:
            data = yaml.safe_load(f)
    else:
        logging.info("Downloading and filtering Mol-Instructions dataset...")
        try:
            dataset = load_dataset("zjunlp/Mol-Instructions", "Molecule-oriented Instructions", split="train", streaming=True)
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return []

        cancer_keywords = data_config['keywords']
        data = []
        count = 0
        for example in dataset:
            if count >= max_samples:
                break
            text = f"{example.get('instruction', '')} {example.get('output', '')}".lower()
            if any(k in text for k in cancer_keywords):
                data.append({
                    'instruction': example.get('instruction', ''),
                    'input': example.get('input', ''),
                    'output': example.get('output', ''),
                    'id': count
                })
                count += 1
        
        with open(filtered_file, 'w') as f:
            yaml.dump(data, f, indent=2)
        logging.info(f"Cached {len(data)} filtered samples.")

    logging.info("Converting data to LlamaIndex Documents...")
    documents = [
        Document(
            text=f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}",
            metadata={"source": "Mol-Instructions", "id": item.get('id', 'N/A')}
        ) for item in data
    ]
    logging.info(f"Created {len(documents)} documents.")
    return documents

def configure_models():
    """Initializes and configures the global LLM and embedding models."""
    logging.info("Configuring models...")
    
    model_config = CONFIG['models']
    llm_gen_config = CONFIG['llm_generation']

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    Settings.llm = HuggingFaceLLM(
        model_name=model_config['llm'],
        tokenizer_name=model_config['llm'],
        context_window=llm_gen_config['context_window'],
        max_new_tokens=llm_gen_config['max_new_tokens'],
        model_kwargs={"quantization_config": quantization_config},
        generate_kwargs={
            "temperature": llm_gen_config['temperature'],
            "do_sample": llm_gen_config['do_sample']
        },
        device_map="auto",
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name=model_config['embedding'])
    logging.info("Models configured successfully.")

def get_or_build_index(documents):
    """
    Builds or loads a persistent ChromaDB vector index.
    """
    vs_config = CONFIG['vector_store']
    parser_config = CONFIG['node_parser']
    db_dir = Path(vs_config['db_directory'])
    db_dir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Setting up ChromaDB in {db_dir}...")
    db = chromadb.PersistentClient(path=str(db_dir))
    chroma_collection = db.get_or_create_collection(vs_config['collection_name'])
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    if chroma_collection.count() == 0:
        logging.info("Building new vector index...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[SentenceSplitter(chunk_size=parser_config['chunk_size'])]
        )
        logging.info("Vector index built and persisted.")
    else:
        logging.info("Loading existing vector index from ChromaDB.")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            transformations=[SentenceSplitter(chunk_size=parser_config['chunk_size'])]
        )
    return index

class UniProtEnrichedQueryEngine(BaseQueryEngine):
    """
    Custom query engine that enriches context with UniProt data.
    """
    def __init__(self, retriever, llm, prompt_template):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.uniprot_cache = UniProtCache()
        self.uniprot_cache.preload_cancer_proteins()
        super().__init__()

    def _extract_proteins(self, text):
        text_upper = text.upper()
        return [p for p in self.uniprot_cache.cancer_proteins if p in text_upper]

    def _build_context(self, retrieved_nodes, protein_info):
        context_parts = []
        if retrieved_nodes:
            context_parts.append("### Retrieved Scientific Information:")
            for i, node in enumerate(retrieved_nodes, 1):
                context_parts.append(f"{i}. {node.get_text()[:350]}...")
        
        if any(protein_info):
            context_parts.append("\n### Protein Database Information (from UniProt):")
            for protein in protein_info:
                if protein:
                    context_parts.append(
                        f"- **{protein['gene']}**: {protein['protein_name']}\n"
                        f"  *Function*: {protein['function'][:200]}...")
        return "\n".join(context_parts)

    def _query(self, query_str):
        logging.info("Retrieving documents...")
        retrieved_nodes = self.retriever.retrieve(query_str)
        
        logging.info("Fetching protein information from UniProt...")
        proteins_mentioned = self._extract_proteins(query_str)
        protein_info = [self.uniprot_cache.fetch_protein_info(p) for p in proteins_mentioned]

        logging.info("Building augmented context...")
        context_str = self._build_context(retrieved_nodes, protein_info)
        
        logging.info("Generating answer with LLM...")
        formatted_prompt = self.prompt_template.format(
            context_str=context_str,
            query_str=query_str
        )
        
        response = self.llm.complete(formatted_prompt)
        
        return {"response": str(response), "source_nodes": retrieved_nodes}


def create_rag_engine():
    """
    Main function to set up and return the complete RAG pipeline.
    """
    logging.info("="*50)
    logging.info("INITIALIZING RAG PIPELINE")
    
    configure_models()
    documents = load_mol_instructions()
    index = get_or_build_index(documents)
    
    retriever_config = CONFIG['retriever']
    retriever = VectorIndexRetriever(index=index, similarity_top_k=retriever_config['similarity_top_k'])

    prompt_template = PromptTemplate(CONFIG['prompt_template'])

    logging.info("Creating UniProt-enriched query engine...")
    query_engine = UniProtEnrichedQueryEngine(
        retriever=retriever,
        llm=Settings.llm,
        prompt_template=prompt_template
    )
    
    logging.info("✅ RAG Pipeline Initialized Successfully!")
    logging.info("="*50)
    
    return query_engine