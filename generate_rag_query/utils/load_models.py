import os
import json
import torch
import time
import gc
# import psutil
# import numpy as np
from time import sleep
# from langchain.chat_models import init_chat_model
# from langchain.document_loaders import DirectoryLoader
from pydantic import BaseModel
# from langchain_core.prompts import PromptTemplate
# from typing import Generator


class WboeLoadModels(BaseModel):

    hf_model: str = "lmstudio-community/Llama-3.3-70B-Instruct-GGUF"
    hf_model_fn: str = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    ollama_model: str = "llama3.3:latest"
    jwt_token: str = os.getenv("OLLAMA_API_KEY")
    hf_token: str = os.getenv("HUGGINGFACE_API_KEY")
    user_input: list[str] = ["prompt1.txt", "prompt2.txt", "prompt3.txt"]
    keyword: str = "keusch",
    max_context_length: int = 128000
    context_token_length: int = 2048
    reserved_prompt_length: int = 4096
    model_size_gb: float = 44.0
    model_memory_usage_1k_token: float = 0.3125  # GB per 1000 tokens
    total_available_gpu_memory: float = 79.0  # in GB
    output_dir: str = "output"
    model: object = None
    tokenizer: object = None
    pipe: object = None
    inputs: str = ""
    conversation_messages: list[dict[str, str]] = []

    # Memory management settings
    gpu_memory_threshold: float = 0.8  # Use max 80% of GPU memory
    enable_memory_monitoring: bool = True
    aggressive_cleanup: bool = True
    retry_on_oom: bool = True

    # Define the model configuration for Pydantic
    # This allows for arbitrary types and forbids extra fields
    # and allows mutation of fields
    # This is necessary for the OllamaEmbeddings and HuggingFacePipeline
    model_config: dict = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    def __init__(self, **data):

        super().__init__(**data)

        for file in self.user_input:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} does not exist.")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if not self.jwt_token and self.backend == "ollama":
            raise ValueError("JWT token for Ollama API must be set.")

        if not self.hf_token and self.backend in ["hf_pipeline", "llama_cpp"]:
            raise ValueError("JWT token for Hugging Face API must be set.")

    def load_ollama_embeddings_function(self):
        """Loads the Ollama embeddings function."""

        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=self.ollama_model,
            base_url="https://open-webui.acdh-dev.oeaw.ac.at/ollama",
            sync_client_kwargs={
                "headers": {"Authorization": f"Bearer {self.jwt_token}"},
            },
            # keep_alive=6000,
            num_ctx=self.context_token_length,
            # temperature=0.4,
        )
        return embeddings

    def load_ollama_model(self):
        """Loads the Ollama model."""

        from langchain_ollama import OllamaLLM

        if not self.jwt_token:
            raise ValueError("OLLAMA_API_KEY environment variable is not set.")

        if not self.ollama_model:
            raise ValueError("Ollama model is not specified.")

        llm = OllamaLLM(
            model=self.ollama_model,
            base_url="https://open-webui.acdh-dev.oeaw.ac.at/ollama",
            sync_client_kwargs={
                "headers": {"Authorization": f"Bearer {self.jwt_token}"},
            },
            num_ctx=self.context_token_length,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
        )

        return llm

    def load_llama_cpp_model(
        self,
        embeddings: bool = False
    ) -> object:
        """Loads the LlamaCpp model with memory management."""

        from llama_cpp import Llama

        if not self.hf_model:
            raise ValueError("LlamaCpp model is not specified.")

        # Check memory before loading
        self.print_gpu_memory_status("Before model loading - ")

        # Clear any existing models first
        self.clear_model_from_cache()

        try:
            model = Llama.from_pretrained(
                repo_id=self.hf_model,
                filename=self.hf_model_fn,
                n_gpu_layers=-1,
                seed=1337,
                embeddings=embeddings,
                n_ctx=self.context_token_length,
                n_batch=2048,
                n_ubatch=512,
                verbose=False,  # Reduce output spam
            )

            self.print_gpu_memory_status("After model loading - ")
            return model

        except Exception as e:
            print(f"Error loading model: {e}")
            # Try with fewer GPU layers if out of memory
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print("Attempting to load with fewer GPU layers...")
                self.clear_model_from_cache()
                if self.aggressive_cleanup:
                    self.force_memory_cleanup()

                if self.retry_on_oom:
                    try:
                        model = Llama.from_pretrained(
                            repo_id=self.hf_model,
                            filename=self.hf_model_fn,
                            n_gpu_layers=-1,  # Try to use all layers on GPU or reduce if needed
                            seed=1337,
                            embeddings=embeddings,
                            n_ctx=self.context_token_length,
                            n_batch=1024,  # Reduced batch size
                            n_ubatch=256,  # Reduced micro batch size
                            verbose=False,
                        )
                        print("Successfully loaded with reduced GPU layers")
                        return model
                    except Exception as e2:
                        print(f"Failed to load with reduced settings: {e2}")
                        raise e2
            raise e

    def load_huggingface_model(self):
        """Loads the Hugging Face model and tokenizer with memory management."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.hf_token:
            raise ValueError("HUGGINGFACE_API_KEY environment\
                variable is not set.")

        if not self.hf_model:
            raise ValueError("Hugging Face model is not specified.")

        # Check memory before loading
        self.print_gpu_memory_status("Before HF model loading - ")

        # Clear any existing models first
        self.clear_model_from_cache()

        try:
            # Load the model and tokenizer with optimized settings
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model,
                token=self.hf_token,
                device_map="auto",  # Use GPU if available
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,  # Optimize CPU memory usage
                trust_remote_code=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model,
                token=self.hf_token,
                trust_remote_code=True,
            )

            self.print_gpu_memory_status("After HF model loading - ")
            return model, tokenizer

        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            if "out of memory" in str(e).lower():
                print("Attempting to load with optimized memory settings...")
                self.clear_model_from_cache()
                if self.aggressive_cleanup:
                    self.force_memory_cleanup()

                if self.retry_on_oom:
                    try:
                        # Try with more aggressive memory optimization
                        model = AutoModelForCausalLM.from_pretrained(
                            self.hf_model,
                            token=self.hf_token,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            max_memory={0: "80%"},  # Limit to 80% of GPU 0
                            offload_folder="./offload",  # CPU offload folder
                        )

                        tokenizer = AutoTokenizer.from_pretrained(
                            self.hf_model,
                            token=self.hf_token,
                            trust_remote_code=True,
                        )

                        print("Successfully loaded with memory optimization")
                        return model, tokenizer

                    except Exception as e2:
                        print(f"Failed to load with optimized settings: {e2}")
                        raise e2
            raise e

    def load_huggingface_pipeline(self):
        """Loads the Hugging Face text generation pipeline."""

        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline

        if not self.hf_token:
            raise ValueError("HUGGINGFACE_API_KEY environment\
                variable is not set.")

        if not self.hf_model:
            raise ValueError("Hugging Face model is not specified.")

        # Create the text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # model_kwargs={
            #     "torch_dtype": torch.float16,
            # },
            # device_map="auto",  # Use GPU if available
            max_new_tokens=512,
            truncation=True,
            max_length=self.context_token_length,
            return_full_text=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            repetition_penalty=1.2,
        )

        pipe = HuggingFacePipeline(pipeline=pipe)

        return pipe

    def unloading_models_and_clear_up_memory(self) -> None:
        """Unloads the Hugging Face model and tokenizer.
        This is necessary to free up GPU memory."""

        # Clear all model references
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        # Clear other references to free memory
        self.user_input = None
        self.keyword = None
        self.output_dir = None

        # Force garbage collection
        gc.collect()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        print("Memory cleared successfully")

    def generate_ollama(self) -> str:
        """Generates a response using the Ollama LLM."""

        try:
            return self.model.invoke(self.conversation_messages)

        except (IndexError, AttributeError):
            print("Error accessing LLM response")
            return ""

    def generate_huggingface(self) -> str:
        """Generates a response using the Hugging Face LLM."""

        # llama_sepecial_tokens = {
        #     "eot_id": "<|eot_id|>",
        #     "start_header_id": "<|start_header_id|>",
        #     "end_header_id": "<|end_header_id|>",
        #     "begin_of_text": "<|begin_of_text|>"
        # }

        try:
            return self.pipe.invoke(self.conversation_messages)

        except (IndexError, AttributeError):
            print("Error accessing LLM response")
            return ""

    def generate_llama_cpp(self) -> str:
        """Generates a response using the Hugging Face LLM."""

        try:
            response = self.model.create_chat_completion(
                messages=self.conversation_messages,
                max_tokens=-1,
                temperature=0.7,
                seed=1337,
                # top_p=0.9,
                # top_k=50,
                # repeat_penalty=1.2,
                # response_format={
                #     "type": "json_object",
                # }
            )
            return response

        except (IndexError, AttributeError):
            print("Error accessing LLM response")
            return {"error": "Error accessing LLM response"}

    def create_conversation_messages(self) -> None:
        """Initializes the chat messages."""

        system_message_content = (
            "Du bist ein Assistent, der auf die Frage des Benutzers antwortet."
            "Du hast Zugriff auf den Kontext, der vom Benutzer bereitgestellt "
            "wird. Deine Aufgabe ist es, die Instruktionen des Benutzers "
            "zu befolgen, indem du die bereitgestellten Informationen nutzt. "
            "Du solltest keine Informationen hinzufügen, die nicht im Kontext "
            "enthalten sind, und du solltest keine Annahmen treffen. "
            "Gehe step-by-step vor und erkläre deine Schritte "
            "ausführlich bevor du eine Antwort gibst."
            "\n\n"
        )

        conversation_messages = [
            {"role": "system", "content": system_message_content},
        ]

        return conversation_messages

    def update_conversation_messages(
        self,
        new_message: str = None,
        role: str = "user"
    ) -> None:
        """Updates the conversation messages with the
        latest user input and model response."""

        if new_message:
            # Add the new message to the conversation
            self.conversation_messages.append(
                {"role": role, "content": (new_message)}
            )

    def verify_conversation_messages_length(self) -> bool:
        """Verifies if the conversation messages length is within the limit."""

        print("Verifying conversation messages length...")
        conversation_length: int = 0

        match self.backend:
            case "ollama":
                self.model = self.load_ollama_embeddings_function()
                conversation_length = sum(
                    len(self.model.embed_query(
                        msg["content"].encode("utf-8")
                    )) for msg in self.conversation_messages
                )
            case "hf_pipeline":
                # must be verified
                self.model, tokenizer = self.load_huggingface_model()
                conversation_length = sum(
                    len(tokenizer(
                        msg["content"].encode("utf-8"),
                        return_tensors="pt"
                    )["input_ids"]) for msg in self.conversation_messages
                )

            case "llama_cpp":
                self.context_token_length = 1024
                self.model = self.load_llama_cpp_model()
                conversation_length = sum(
                    len(self.model.tokenize(
                        msg["content"].encode("utf-8")
                    )) for msg in self.conversation_messages
                )

        if conversation_length > (
                self.max_context_length - self.reserved_prompt_length):
            print(f"Conversation messages length exceeds the limit:\
                {conversation_length}")
            return True, conversation_length

        print(f"Conversation messages length is within the limit:\
            {conversation_length}")
        return False, conversation_length

    def truncate_text(self, text: str, conversation_length: int) -> str:
        """Truncates the text to fit within the context token length."""

        match self.backend:
            case "hf_pipeline":
                # must be verified
                self.model, tokenizer = self.load_huggingface_model()
                tokens = tokenizer(
                    text.encode("utf-8"),
                    return_tensors="pt"
                )["input_ids"][0].tolist()

            case "llama_cpp":
                self.context_token_length = 1024
                self.model = self.load_llama_cpp_model()
                tokens = self.model.tokenize(text.encode("utf-8"))

        token_count = len(tokens)
        print(f"Token count: {token_count}")
        print(f"Max length: {self.max_context_length}")

        exceeded_length = conversation_length - (
            self.max_context_length - self.reserved_prompt_length)
        to_truncate = max(0, token_count - exceeded_length)

        truncated_tokens = tokens[:to_truncate]
        truncated_text = self.model.detokenize(truncated_tokens)
        print(f"Truncated text length: {len(truncated_text)}")

        return truncated_text.decode("utf-8")

    def clear_model_from_cache(self) -> None:
        """Clears the model cache to free up memory."""

        # Clear model references
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
            self.pipe = None

        # Force garbage collection
        gc.collect()

        # Clear GPU memory if available
        if torch.cuda.is_available():
            print("Clearing GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()

        # Wait for memory to be fully released
        sleep(2)

    def get_gpu_memory_info(self) -> dict:
        """Get current GPU memory usage information."""

        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")

        mem_i_all_devices = {
            "devices": [],
            "total_memory_gb": 0.0,
            "allocated_memory_gb": 0.0,
            "reserved_memory_gb": 0.0,
            "free_memory_gb": 0.0,
            "memory_usage_percent": 0.0
        }

        for i in range(device_count):
            device = torch.device(f'cuda:{i}')
            print(f"Device {i}: {torch.cuda.get_device_name(device)}")
            total_memory = torch.cuda.get_device_properties(device).total_memory
            print(f"  - Total Memory: {total_memory / (1024**3):.2f} GB")
            reserved_memory = torch.cuda.memory_reserved(device)
            allocated_memory = torch.cuda.memory_allocated(device)
            free_memory = total_memory - allocated_memory

            memory_info = {
                "device_index": i,
                "device_name": torch.cuda.get_device_name(device),
                "total_memory_gb": total_memory / (1024**3),
                "allocated_memory_gb": allocated_memory / (1024**3),
                "reserved_memory_gb": reserved_memory / (1024**3),
                "free_memory_gb": free_memory / (1024**3),
                "memory_usage_percent": (allocated_memory / total_memory) * 100
            }

            mem_i_all_devices["devices"].append(memory_info)
            mem_i_all_devices["total_memory_gb"] += memory_info["total_memory_gb"]
            mem_i_all_devices["allocated_memory_gb"] += memory_info["allocated_memory_gb"]
            mem_i_all_devices["reserved_memory_gb"] += memory_info["reserved_memory_gb"]
            mem_i_all_devices["free_memory_gb"] += memory_info["free_memory_gb"]
            mem_i_all_devices["memory_usage_percent"] += memory_info["memory_usage_percent"]

        return mem_i_all_devices

    def check_free_gpu_memory(self) -> float:
        """Get the amount of free GPU memory in GB."""

        memory_info = self.get_gpu_memory_info()
        if "error" in memory_info:
            print("Warning: Cannot check GPU memory, CUDA not available")
            return float('inf')
        return memory_info['free_memory_gb']

    def print_gpu_memory_status(self, prefix: str = "") -> None:
        """Print current GPU memory status."""

        memory_info = self.get_gpu_memory_info()
        if "error" in memory_info:
            print(f"{prefix}GPU Memory: {memory_info['error']}")
            return

        print(f"{prefix}GPU Memory Status:")
        print(f"  - Total: {memory_info['total_memory_gb']:.2f} GB")
        print(f"  - Allocated: {memory_info['allocated_memory_gb']:.2f} GB")
        print(f"  - Free: {memory_info['free_memory_gb']:.2f} GB")
        print(f"  - Usage: {memory_info['memory_usage_percent']:.1f}%")

    def check_available_memory(self, required_memory_gb: float = 8.0) -> bool:
        """Check if enough GPU memory is available for model loading."""

        memory_info = self.get_gpu_memory_info()
        if "error" in memory_info:
            print("Warning: Cannot check GPU memory, CUDA not available")
            return True  # Assume it's okay if we can't check

        available_memory = memory_info['free_memory_gb']
        if available_memory < required_memory_gb:
            print(f"Warning: Only {available_memory:.2f} GB available, "
                  f"but {required_memory_gb:.2f} GB required")
            return False

        print(f"Memory check passed: {available_memory:.2f} GB available")
        return True

    def force_memory_cleanup(self) -> None:
        """Force aggressive memory cleanup."""

        print("Performing aggressive memory cleanup...")

        # Clear all model references
        self.clear_model_from_cache()

        # Multiple garbage collection passes
        for i in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            sleep(1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("Aggressive cleanup completed")
        self.print_gpu_memory_status("After cleanup - ")

    def ollama(self) -> None:
        """Generates responses using the Ollama LLM."""

        self.conversation_messages = self.create_conversation_messages()
        self.update_conversation_messages(new_message=self.inputs, role="user")

        prompt_files = self.user_input
        for file in prompt_files:
            print(f"Processing file: {file}")

            with open(file, "r") as f:
                text = f.read().strip()

            self.update_conversation_messages(new_message=text, role="user")
            if self.ollama_model in ["gemma3:27b"]:
                self.context_token_length = 35000
            else:
                _, length = self.verify_conversation_messages_length()
                self.context_token_length = (
                    length + self.reserved_prompt_length)
                print(f"Context token length: {self.context_token_length}")

            self.model = self.load_ollama_model()
            response = self.generate_ollama()

            # print(response)
            fn_prompt = file.split("/")[-1].replace(".txt", "")
            fn_model = self.ollama_model.replace("/", "-")
            fn_out_dir = self.output_dir
            fn_keyword = self.keyword
            fn_name = f"{fn_keyword}_{fn_prompt}_{fn_model}.txt"
            fn = os.path.join(fn_out_dir, fn_name)

            try:
                with open(fn, "w") as file:
                    file.write(response)

            except Exception as e:
                print(f"Error writing response to file: {e}")
                print("Response:", response)

            self.update_conversation_messages(
                new_message=response,
                role="assistant"
            )

            sleep(5)

    def huggingface(self) -> None:
        """Generates responses using the Hugging Face LLM."""

        # Load the Hugging Face model and tokenizer sperately
        # to allow manually unloading
        self.model, self.tokenizer = self.load_huggingface_model()

        # Create the Hugging Face pipeline
        self.pipe = self.load_huggingface_pipeline()

        self.conversation_messages = self.create_conversation_messages()

        prompt_files = self.user_input

        for file in prompt_files:

            with open(file, "r") as f:
                text = f.read().strip()

            # concatenate the inputs (context from vectorstore)
            # and user prompt message
            self.user_input_text = text

            self.update_conversation_messages(new_message=text, role="user")
            # truncate the input text to fit within the context length
            exceeded, length = self.verify_conversation_messages_length()
            if exceeded:
                self.inputs = self.truncate_text(self.inputs, length)
                self.conversation_messages[1]["content"] = (self.inputs)

            response = self.generate_huggingface()

            # print(response)
            fn_prompt = file.split("/")[-1].replace(".txt", "")
            fn_model = self.hf_model.replace("/", "-")
            fn_out_dir = self.output_dir
            fn_keyword = self.keyword
            fn_name = f"{fn_keyword}_{fn_prompt}_{fn_model}.txt"
            fn = os.path.join(fn_out_dir, fn_name)

            try:
                with open(fn, "w") as file:
                    file.write(response)

            except Exception as e:
                print(f"Error writing response to file: {e}")
                print("Response:", response)

            self.update_conversation_messages(
                new_message=response,
                role="assistant"
            )

            # Clear GPU memory after each file processing
            if torch.cuda.is_available():
                print("Clearing GPU memory...")
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            sleep(5)

    def evaluate_gpu_memory_requirements(self) -> float:
        """Evaluates the GPU memory requirements based on model size
        and context length."""

        context_memory_gb = (
            (self.context_token_length / 1000) * self.model_memory_usage_1k_token
        )
        total_memory_gb = self.model_size_gb + context_memory_gb

        print(f"Estimated GPU memory requirement: {total_memory_gb:.2f} GB")
        return float(total_memory_gb)

    def llama_cpp(self) -> None:
        """Generates responses using the LlamaCpp LLM
        with improved memory management."""

        # count time
        start_time = time.time()
        print("5: Starting LlamaCpp generation...")

        if self.enable_memory_monitoring:
            self.print_gpu_memory_status("Initial GPU Memory - ")

        self.conversation_messages = self.create_conversation_messages()
        self.update_conversation_messages(
            new_message=self.inputs,
            role="user"
        )
        print("6: System Msg. and Inputs set.")

        for i, file in enumerate(self.user_input):
            print(f"\n--- Processing file {i + 1}/{len(self.user_input)}:\
                {file} ---")

            if self.enable_memory_monitoring and i > 0:
                self.print_gpu_memory_status(f"Before iteration {i + 1} - ")

            with open(file, "r") as f:
                text = f.read().strip()

            self.update_conversation_messages(
                new_message=text,
                role="user"
            )
            print("7: Adding User prompt to messsages.")

            # truncate the input text to fit within the context length
            exceeded, length = self.verify_conversation_messages_length()
            print(f"8: Exceeded: {exceeded}, Length: {length}")

            # Clear model before loading new one
            self.clear_model_from_cache()

            if exceeded:
                print("Context length exceeded, truncating...")
                self.inputs = self.truncate_text(self.inputs, length)
                self.conversation_messages[1]["content"] = (self.inputs)
                self.context_token_length = self.max_context_length
                # Additional cleanup after truncation
                self.clear_model_from_cache()
                if self.aggressive_cleanup:
                    self.force_memory_cleanup()
            else:
                self.context_token_length = (
                    length + self.reserved_prompt_length
                )

            if self.enable_memory_monitoring and i > 0:
                required_memory = self.evaluate_gpu_memory_requirements()
                print(f"Required memory for loading model:\
                    {required_memory:.2f} GB")

                # Check if we have enough memory before loading model
                if not self.check_available_memory(
                        required_memory_gb=required_memory):
                    print("Insufficient memory detected,\
                        performing cleanup...")

                    self.clear_model_from_cache()
                    if self.aggressive_cleanup:
                        self.force_memory_cleanup()

                    # Wait a bit more for memory to be released
                    sleep(3)

                    # Check again
                    if not self.check_available_memory(
                            required_memory_gb=required_memory):
                        print("Still insufficient memory,\
                            using smaller context...")
                        self.context_token_length = min(
                            self.context_token_length, 4096)

            try:
                self.model = self.load_llama_cpp_model()
                print(f"9: Model loaded: {self.hf_model_fn}")

                # generate response from the LlamaCpp model
                response = self.generate_llama_cpp()
                print("10: Response generated.")

            except Exception as e:
                print(f"Error during model loading or generation: {e}")
                if "out of memory" in str(e).lower():
                    print("OOM detected, performing aggressive cleanup and\
                        retrying...")
                    self.clear_model_from_cache()
                    if self.aggressive_cleanup:
                        self.force_memory_cleanup()
                    sleep(5)

                    if self.retry_on_oom:
                        # Reduce context length and try again
                        self.context_token_length = min(
                            self.context_token_length // 2, 2048)

                        try:
                            self.model = self.load_llama_cpp_model()
                            response = self.generate_llama_cpp()
                            print("10: Response generated after OOM recovery.")
                        except Exception as e2:
                            print(f"Failed to recover from OOM: {e2}")
                            continue  # Skip this file and move to next
                else:
                    print(f"Non-memory related error: {e}")
                    continue

            # save the response to a file
            fn_prompt = file.split("/")[-1].replace(".txt", "")
            fn_model = self.hf_model.replace("/", "-")
            fn_out_dir = self.output_dir
            fn_keyword = self.keyword
            fn_name = f"{fn_keyword}_{fn_prompt}_{fn_model}.json"
            fn = os.path.join(fn_out_dir, fn_name)

            # count time
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken to process {file}: {elapsed_time:.2f} seconds")
            response["elapsed_time"] = elapsed_time

            try:
                with open(fn, "w") as file:
                    json.dump(response, file, indent=4)
                print(f"Response saved to {fn}")

            except Exception as e:
                print(f"Error writing response to file: {e}")
                print("Response:", response)

            response = response.get(
                "choices", [{}])[0].get("message", {}).get("content", "")

            self.update_conversation_messages(
                new_message=response,
                role="assistant"
            )
            print("11: Updated conversation messages with response.")

            # Clean up after each iteration to prevent memory accumulation
            self.clear_model_from_cache()

            print(f"12: Iteration {i + 1} completed for file: {file}")

            # Memory status after iteration
            self.print_gpu_memory_status(f"After iteration {i + 1} - ")

            # Additional wait between iterations for memory stabilization
            if i < len(self.user_input) - 1:  # Don't wait after last iteration
                print("Waiting for memory stabilization...")
                sleep(3)

        print("\n=== LlamaCpp generation completed ===")
        self.print_gpu_memory_status("Final GPU Memory - ")
