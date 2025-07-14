import os
import json
from time import sleep
from utils.load_models import WboeLoadModels
from utils.load_vectorestore_documents import WboeLoadVectorstore
from pydantic import BaseModel
from typing import Literal


class WboeBaseRAG(BaseModel):
    # Common attributes and configuration
    context: dict[str, str] = {}
    backend: Literal["ollama", "hf_pipeline", "llama_cpp"] = "ollama"
    embeddings: int = 0
    available_gpu_memory: int = 80  # in GB
    model_memory_usage: int = 44  # in GB, for the model itself

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

        if not self.hf_model and self.backend in ["hf_pipeline", "llama_cpp"]:
            raise ValueError("Hugging Face model must be specified.")

        if not self.hf_model_fn and self.backend == "llama_cpp":
            raise ValueError("Hugging Face model file name must be specified.")

        if not self.ollama_model and self.backend == "ollama":
            raise ValueError("Ollama model must be specified.")

        if not self.jwt_token and self.backend == "ollama":
            raise ValueError("JWT token for Ollama API must be set.")

        if not self.hf_token and self.backend in ["hf_pipeline", "llama_cpp"]:
            raise ValueError("JWT token for Hugging Face API must be set.")

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

        self.max_context_length = self.calc_max_context_length()
        print(f"1: Max. context length calculated: {self.max_context_length}")

        for doc in self.context:
            print(f"2: Processing document: {doc['keyword']}")
            self.keyword = doc["keyword"]

            if self.keyword != "43218__Gefrette_Simplex":
                print(f"Skipping document {self.keyword}\
                    as it is not '43218__Gefrette_Simplex'.")
                continue

            try:
                self.inputs = doc["context"]
                self.embeddings = len(doc["embeddings"])
                print(f"3: Document {self.keyword} has {self.embeddings}\
                    embeddings.")

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
                    self.load_huggingface_model()
                    print(f"Using Hugging Face model: {self.hf_model}")
                    self.huggingface()

                case "llama_cpp":
                    print("4: Using Llama CPP model for word embeddings.")
                    print(f"Using Hugging Face model: {self.hf_model}")
                    print(f"Using Llama CPP model file: {self.hf_model_fn}")
                    self.llama_cpp()

            print(f"Processed document: {doc['keyword']} successfully.")
            sleep(5)

    def save_chat_history(self) -> None:
        """Saves the chat history to a file."""

        if not self.conversation_messages:
            print("No conversation messages to save.")
            return

        output_file = os.path.join(
            self.output_dir,
            "conversation_history.json"
        )

        try:
            with open(output_file, "w") as file:
                json.dump(self.conversation_messages, file, indent=4)
        except IOError as e:
            print(f"Error saving chat history to {output_file}: {e}")
            return

        print(f"Conversation history saved to {output_file} successfully.")

    def calc_max_context_length(self) -> int:
        """Calculates the maximum context length based on the backend."""
        usage_for_max_length = 40
        avail_gpu_memory = self.available_gpu_memory - self.model_memory_usage
        token_per_gb = self.max_context_length / usage_for_max_length

        return int(token_per_gb * avail_gpu_memory)

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

        # save chat history to a file
        self.save_chat_history()

        # Unload models and clear up memory
        self.unloading_models_and_clear_up_memory()


if __name__ == "__main__":
    wboe_embeddings = WboeRAGPipeline(
        backend="llama_cpp",
        hf_model="lmstudio-community/Llama-3.3-70B-Instruct-GGUF",
        hf_model_fn="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        # hf_model="bartowski/Llama-3.2-3B-Instruct-GGUF",
        # hf_model_fn="Llama-3.2-3B-Instruct-Q4_0.gguf",
        ollama_model="deepseek-r1:32b",
        collection_name="wboe_word_embeddings",
        vector_store_filepath_name="chroma_langchain_db_wboe_embeddings",
        jwt_token=os.getenv("OLLAMA_API_KEY"),
        hf_token=os.getenv("HUGGINGFACE_API_KEY"),
        user_input=["prompt1.txt", "prompt2.txt", "prompt3.txt"],
        max_context_length=128000,
        available_gpu_memory=80,
        model_memory_usage=44,
        output_dir="output",
    )
    wboe_embeddings.main()
