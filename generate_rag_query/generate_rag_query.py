import os
import logfire
import json
from time import sleep, strftime
from utils.load_models import WboeLoadModels
from utils.load_vectorestore_documents import WboeLoadVectorstore
from pydantic import BaseModel
from typing import Literal, Dict, Any

# Configure Logfire for logging
if os.getenv("LOGFIRE_TOKEN") is None:
    raise ValueError("LOGFIRE_TOKEN environment variable is not set.")

logfire.configure()


class WboeBaseRAG(BaseModel):
    # Common attributes and configuration
    context: dict[str, str] = {}
    keywords_to_process: list[str] = []
    backend: Literal["ollama", "llama_cpp", "openAI"] = "ollama"
    embeddings: int = 0
    model_memory_usage: float = 44.0  # in GB, for the model itself
    conversations: list[dict[str, str]] = []

    model_config: dict = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    def __init__(self, **args):
        super().__init__(**args)

        if not self.backend:
            raise ValueError("Backend must be specified.\
                Supported backends are: 'ollama', 'llama_cpp', 'openAI'.")

        if self.backend not in ["ollama", "llama_cpp", "openAI"]:
            raise ValueError(f"Unsupported backend: {self.backend}.\
                Supported backends are: 'ollama', 'llama_cpp', 'openAI'.")


class WboeRAGPipeline(WboeBaseRAG, WboeLoadVectorstore, WboeLoadModels):

    """WBOE RAG Pipeline for generating word embeddings using Ollama
    and Hugging Face models."""

    def __init__(self, **args):

        super().__init__(**args)

        if not self.hf_model and self.backend == "llama_cpp":
            raise ValueError("Hugging Face model must be specified.")

        if not self.hf_model_fn and self.backend == "llama_cpp":
            raise ValueError("Hugging Face model file name must be specified.")

        if not self.ollama_model and self.backend == "ollama":
            raise ValueError("Ollama model must be specified.")

        if not self.openai_model and self.backend == "openAI":
            raise ValueError("OpenAI model must be specified.")

        if not self.jwt_token and self.backend == "ollama":
            raise ValueError("JWT token for Ollama API must be set.")

        if not self.hf_token and self.backend == "llama_cpp":
            raise ValueError("JWT token for Hugging Face API must be set.")

        if not self.openai_model and self.backend == "openAI":
            raise ValueError("OpenAI model must be specified.")

        if not self.collection_name:
            raise ValueError("Collection name for the vector store must be\
                specified.")

        if not self.vector_store_filepath_name:
            raise ValueError("Vector store file path name must be specified.")

        if not self.user_input:
            raise ValueError("User input files must be specified.")

    def create_keyword_conversation_history(self, keyword: str) -> None:
        """Creates a conversation history for a specific keyword."""
        messages: list[dict[str, str]] = self.conversation_messages

        if not messages:
            logfire.info("No conversation messages to save.")
            return

        self.conversations.append({
            "keyword": keyword,
            "messages": messages
        })

    def save_chat_history(self) -> None:
        """Saves the chat history to a file."""
        conversations = self.conversations
        if not conversations:
            logfire.info("No conversation to save.")
            return

        output_stats = {
            "pipeline_config": {
                "backend": self.backend,
                "ollama_model": self.ollama_model,
                "hf_model": self.hf_model,
                "hf_model_fn": self.hf_model_fn,
                "max_context_length": self.max_context_length,
                "model_memory_usage": self.model_memory_usage,
                "keywords_to_process": self.keywords_to_process,
                "vector_store_collection_name": self.collection_name,
                "vector_store_filepath_name": self.vector_store_filepath_name,
                "user_input_files": self.user_input,
                "enable_memory_monitoring": self.enable_memory_monitoring,
                "aggressive_cleanup": self.aggressive_cleanup,
                "retry_on_oom": self.retry_on_oom
            },
            "conversation": conversations,
            "statistics": {
                "total_documents_processed": len(conversations),
                "total_user_inputs": len(self.user_input),
                "total_inference_time_seconds": sum(
                    msg.get("elapsed_time_seconds", 0)
                    for conv in conversations
                    for msg in conv.get("messages", [])
                    if "generation completed." in msg.get("content")
                ),
                "average_inference_time_per_document_seconds": (
                    sum(
                        msg.get("elapsed_time_seconds", 0)
                        for conv in conversations
                        for msg in conv.get("messages", [])
                        if "generation completed." in msg.get("content")
                    ) / len(conversations)
                    if conversations else 0
                ),
            }
        }

        fn_time = strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(
            self.output_dir,
            f"conversation_history_{fn_time}.json"
        )

        try:
            with open(output_file, "w") as file:
                json.dump(output_stats, file, indent=4)
        except IOError as e:
            logfire.info(f"Error saving chat history to {output_file}: {e}")
            return

        logfire.info(f"Conversation history saved to {output_file} successfully.")

    def calc_max_context_length(self) -> int:
        """Calculates the maximum context length based on the backend."""
        total_available_gpu_memory: float = self.total_available_gpu_memory
        model_memory_usage: float = self.model_memory_usage
        max_context_length: int = self.max_context_length
        usage_for_max_length = 40  # default for Llama3.3 70B
        avail_gpu_memory = max(total_available_gpu_memory - model_memory_usage, 24)
        token_per_gb = max_context_length / usage_for_max_length

        return int(token_per_gb * avail_gpu_memory)

    def model_memory_handling(self) -> None:
        """Handles model loading and unloading based on memory requirements."""
        self.model_size_gb = self.model_memory_usage
        self.total_available_gpu_memory = self.check_free_gpu_memory()
        self.model_memory_usage_1k_token = (40 / 128000) * 1000  # GB per 1000 tokens
        logfire.info("Model memory handling:")
        logfire.info(f"Model size (GB): {self.model_size_gb}")
        logfire.info(f"Total available GPU memory (GB): {self.total_available_gpu_memory}")
        logfire.info(f"Model memory usage per 1000 tokens (GB): {self.model_memory_usage_1k_token}")

    def get_documents(self) -> None:
        """Initializes the vector store."""

        self.context = self.yield_documents()
        if not self.context:
            raise ValueError("No documents found in the vector store.")

    def generate(self):
        """Generates word embeddings using the Ollama model."""

        self.max_context_length = self.calc_max_context_length()
        logfire.info(f"1: Max. context length calculated: {self.max_context_length}")
        context: dict[str, str] = self.context
        backend: str = self.backend

        with logfire.span("iterate documents for llm context"):
            for doc in context:
                self.keyword = doc["keyword"]
                if len(self.keywords_to_process) > 0 and self.keyword not in self.keywords_to_process:
                    continue

                logfire.info(f"2: Processing document: {doc['keyword']}")
                try:
                    self.inputs = doc["context"]
                    # self.embeddings = len(doc["embeddings"])
                    # logfire.info(f"3: Document {self.keyword} has {self.embeddings}\
                    #     embeddings.")

                except Exception as e:
                    logfire.info(f"Error processing document {doc['keyword']}: {e}")
                    continue

                match backend:
                    case "openAI":
                        logfire.info("Using OpenAI model for word embeddings.")
                        logfire.info(f"Using OpenAI model: {self.openai_model}")
                        self.openai()

                    case "ollama":
                        logfire.info("Using Ollama model for word embeddings.")
                        logfire.info(f"Using Ollama model: {self.ollama_model}")
                        self.ollama()

                    case "llama_cpp":
                        logfire.info("4: Using Llama CPP model for word embeddings.")
                        logfire.info(f"Using Llama CPP model: {self.hf_model}")
                        logfire.info(f"Using Llama CPP model file: {self.hf_model_fn}")
                        self.llama_cpp()

                # save chat history to a file
                with logfire.span("save chat history"):
                    self.create_keyword_conversation_history(self.keyword)
                    logfire.info("Chat history updated successfully.")

                logfire.info(f"Processed document: {doc['keyword']} successfully.")
                sleep(5)

    def main(self) -> Dict[str, Any]:
        """Main method to run the RAG pipeline."""

        logfire.info("Starting WBOE RAG Pipeline")
        self.model_memory_handling()
        enable_memory_monitoring: bool = self.enable_memory_monitoring

        if enable_memory_monitoring:
            logfire.info("1. Initial Memory Status:")
            self.print_gpu_memory_status()
            logfire.info("3. Memory Information:")
            memory_info = self.get_gpu_memory_info()
            if "error" not in memory_info:
                for key, value in memory_info.items():
                    if isinstance(value, float):
                        logfire.info(f"{key}: {value:.2f}")
                    else:
                        logfire.info(f"{key}: {value}")

        logfire.info(f"Initializing WBOE RAG Pipeline with backend: {self.backend}")

        # Load documents from the vector store
        logfire.info("Loading documents from the vector store...")
        with logfire.span("load documents"):
            self.get_documents()
            logfire.info("Documents loaded successfully.")

        # Generate word embeddings using the specified model
        with logfire.span("generate llm responses"):
            logfire.info("Generate LLM responses using the specified model...")
            self.generate()
            logfire.info("RAG pipeline completed successfully.")

        with logfire.span("save chat history to a file"):
            self.save_chat_history()
            logfire.info("Chat history saved successfully.")

        # Unload vector store and clear up memory
        with logfire.span("unload vector store, model and clear up memory"):
            self.unloading_vector_store_and_clear_up_memory()
            self.unloading_models_and_clear_up_memory()
            logfire.info("Vector store unloaded and cleared up memory successfully.")

        # Return results summary
        return {
            "status": "completed",
            "conversations": self.conversations,
            "total_documents_processed": len(self.conversations),
            "backend_used": self.backend,
            "output_directory": self.output_dir
        }


if __name__ == "__main__":
    model_handler = WboeRAGPipeline(
        backend="llama_cpp",
        openai_model="gpt-4o",
        # hf_model="lmstudio-community/Llama-3.3-70B-Instruct-GGUF",
        # hf_model_fn="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        hf_model="bartowski/Llama-3.2-3B-Instruct-GGUF",
        hf_model_fn="Llama-3.2-3B-Instruct-Q4_0.gguf",
        ollama_model="llama3.2:3b",
        collection_name="wboe_word_embeddings",
        vector_store_filepath_name="chroma_langchain_db_wboe_embeddings",
        user_input=[
            "prompt1.txt",
            "prompt2.txt",
            "prompt3.txt",
            "prompt4.txt"
        ],
        keywords_to_process=[
            "44358__geilig_Simplex",
            "43221__gockert_Simplex",
        ],
        max_context_length=128000,
        model_memory_usage=4.0,
        output_dir="output",
        gpu_memory_threshold=0.9,  # Use max 90% of GPU memory
        enable_memory_monitoring=True,
        aggressive_cleanup=True,
        retry_on_oom=True
    )

    model_handler.main()
    # keywords_to_process=[
    #     "44358__geilig_Simplex",
    #     "43221__gockert_Simplex",
    #     "44375__Käue_Simplex",
    #     "44224__Köder_Simplex",
    #     "44394__Keife_Simplex",
    #     "44376__käuen_Simplex",
    #     "44399__kenten_Simplex",
    #     "44370__kardätschen_Simplex",
    #     "44223__ködern_Simplex"
    # ],
