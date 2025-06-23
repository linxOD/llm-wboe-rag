import os
from time import sleep
from utils.load_models import WboeLoadModels
from utils.load_vectorestore_documents import WboeLoadVectorstore
from pydantic import BaseModel
from typing import Literal


class WboeBaseRAG(BaseModel):
    # Common attributes and configuration
    context: dict[str, str] = {}
    backend: Literal["ollama", "hf_pipeline", "llama_cpp"] = "ollama"

    model_config: dict = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    def __init__(self, **args):
        super().__init__(**args)

        if not self.backend:
            raise ValueError("Backend must be specified.\
                Supported backends are: 'ollama', 'hf_pipeline', 'llama_cpp'.")

        if self.backend not in ["ollama", "hf_pipeline", "llama_cpp"]:
            raise ValueError(f"Unsupported backend: {self.backend}.\
                Supported backends are: 'ollama', 'hf_pipeline', 'llama_cpp'.")


class WboeRAGPipeline(WboeBaseRAG, WboeLoadVectorstore, WboeLoadModels):

    """WBOE RAG Pipeline for generating word embeddings using Ollama
    and Hugging Face models."""

    def __init__(self, **args):

        super().__init__(**args)

        if not self.hf_model or not self.ollama_model:
            raise ValueError("Both Hugging Face and Ollama models must be\
                specified.")

        if not self.ollama_model or not self.hf_model:
            raise ValueError("Both Ollama and Hugging Face models must be\
                specified.")

        if not self.jwt_token or not self.hf_token:
            raise ValueError("Both JWT token for Ollama and Hugging Face token\
                must be set.")

        if not self.hf_model_fn:
            raise ValueError("Hugging Face model file name must be specified.")

        if not self.collection_name:
            raise ValueError("Collection name for the vector store must be\
                specified.")

        if not self.vector_store_filepath_name:
            raise ValueError("Vector store file path name must be specified.")

        if not self.user_input:
            raise ValueError("User input files must be specified.")

    def get_documents(self) -> None:
        """Initializes the vector store."""

        self.context = self.yield_documents()
        if not self.context:
            raise ValueError("No documents found in the vector store.")

    def generate(self):
        """Generates word embeddings using the Ollama model."""

        for doc in self.context:
            print(f"Processing document: {doc['keyword']}")

            try:
                self.inputs = doc["context"]
                if len(self.inputs) > self.max_context_length:
                    print(f"User Input Context of: {len(self.inputs)} is\
                        larger than max length of:\
                            {self.max_context_length} truncating inputs...")
                    self.inputs = self.inputs[:self.max_context_length]
            except Exception as e:
                print(f"Error processing document {doc['keyword']}: {e}")
                continue

            match self.backend:
                case "ollama":
                    print("Using Ollama model for word embeddings.")
                    print(f"Using Ollama model: {self.ollama_model}")
                    self.ollama()

                case "hf_pipeline":
                    print("Using Hugging Face pipeline for word embeddings.")
                    print(f"Using Hugging Face model: {self.hf_model}")
                    self.huggingface()

                case "llama_cpp":
                    print("Using Llama CPP model for word embeddings.")
                    print(f"Using Hugging Face model: {self.hf_model}")
                    print(f"Using Llama CPP model file: {self.hf_model_fn}")
                    self.llama_cpp()

            print(f"Processed document: {doc['keyword']} successfully.")
            sleep(5)

    def main(self) -> None:
        """Main method to run the RAG pipeline."""

        print(f"Initializing WBOE RAG Pipeline with backend: {self.backend}")
        # Load documents from the vector store
        print("Loading documents from the vector store...")
        self.get_documents()
        print("Documents loaded successfully.")
        # Generate word embeddings using the specified model
        print("Generate LLM respones using the specified model...")
        self.generate()
        print("RAG pipeline completed successfully.")

        # Unload models and clear up memory
        self.unloading_models_and_clear_up_memory()


if __name__ == "__main__":
    wboe_embeddings = WboeRAGPipeline(
        backend="llama_cpp",
        hf_model="lmstudio-community/Llama-3.3-70B-Instruct-GGUF",
        hf_model_fn="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        ollama_model="llama3.2:3b",
        collection_name="wboe_word_embeddings",
        vector_store_filepath_name="chroma_langchain_db_wboe_embeddings",
        jwt_token=os.getenv("OLLAMA_API_KEY"),
        hf_token=os.getenv("HUGGINGFACE_API_KEY"),
        user_input=["prompt1.txt", "prompt2.txt", "prompt3.txt"],
        keyword="keusch",
        max_context_length=8192,
        output_dir="output",
    )
    wboe_embeddings.main()
