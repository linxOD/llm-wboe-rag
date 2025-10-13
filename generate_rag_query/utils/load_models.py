import os
import json
import torch
import time
import gc
import logfire
# import psutil
# import numpy as np
from time import sleep
# from langchain.chat_models import init_chat_model
# from langchain.document_loaders import DirectoryLoader
from pydantic import BaseModel
# from langchain_core.prompts import PromptTemplate
# from typing import Generator
from typing import Union, Literal


class WboeLoadModels(BaseModel):

    backend: Literal["ollama", "llama_cpp", "openAI"] = "ollama"
    openai_model: str = "gpt-4"
    hf_model: str = "lmstudio-community/Llama-3.3-70B-Instruct-GGUF"
    hf_model_fn: str = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
    ollama_model: str = "llama3.3:latest"
    jwt_token: str = os.getenv("OLLAMA_API_KEY")
    hf_token: str = os.getenv("HUGGINGFACE_API_KEY")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
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
    inputs: str = ""
    conversation_messages: list[dict[str, str]] = []
    user_context_length: int = 0
    context_tokens: int = 0

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

        if not self.hf_token and self.backend == "llama_cpp":
            raise ValueError("JWT token for Hugging Face API must be set.")

        if not self.openai_api_key and self.backend == "openAI":
            raise ValueError("OpenAI API key must be set.")

    # def load_openai_tokenizer(self):
    #     """Loads the OpenAI tokenizer."""
    #     from transformers import AutoTokenizer
    #     model_name: str = self.openai_model

    #     if not model_name:
    #         raise ValueError("OpenAI model is not specified.")

    #     try:
    #         tokenizer = AutoTokenizer.from_pretrained(model_name)
    #         return tokenizer
    #     except Exception as e:
    #         logfire.info(f"Error loading tokenizer: {e}")
    #         raise e

    def load_openai_model(self):
        """Loads the OpenAI model."""
        from langchain_openai import ChatOpenAI
        model_name: str = self.openai_model
        api_key: str = self.openai_api_key
        # temperature: float = 0.7
        # top_p: float = 0.9
        max_tokens: int = None
        timeout: int = None
        max_retries: int = 2

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        if not model_name:
            raise ValueError("OpenAI model is not specified.")

        llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=api_key,
            # temperature=temperature,  # not supported for GPT-5
            # top_p=top_p,  # not supported for GPT-5
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

        return llm

    def load_ollama_embeddings_function(self):
        """Loads the Ollama embeddings function."""

        from langchain_ollama import OllamaEmbeddings
        model_name: str = self.ollama_model
        base_url: str = "https://open-webui.acdh-dev.oeaw.ac.at/ollama"
        token: str = self.jwt_token
        context_token_length: int = self.context_token_length

        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url,
            sync_client_kwargs={
                "headers": {"Authorization": f"Bearer {token}"},
            },
            num_ctx=context_token_length,
        )
        return embeddings

    def load_ollama_model(self):
        """Loads the Ollama model."""

        from langchain_ollama import OllamaLLM
        model_name: str = self.ollama_model
        token: str = self.jwt_token
        context_token_length: int = self.context_token_length
        base_url: str = "https://open-webui.acdh-dev.oeaw.ac.at/ollama"
        temperature: float = 0.7
        top_p: float = 0.9
        top_k: int = 50
        repetition_penalty: float = 1.2

        if not token:
            raise ValueError("OLLAMA_API_KEY environment variable is not set.")

        if not model_name:
            raise ValueError("Ollama model is not specified.")

        llm = OllamaLLM(
            model=model_name,
            base_url=base_url,
            sync_client_kwargs={
                "headers": {"Authorization": f"Bearer {token}"},
            },
            num_ctx=context_token_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        return llm

    def load_llama_cpp_model_tokenizer(self) -> object:
        """Loads the LlamaCpp model with memory management."""

        from llama_cpp import Llama
        model_name: str = self.hf_model
        seed: int = 1337
        n_ctx: int = self.context_token_length
        verbose: bool = False  # Reduce output spam
        n_gpu_layers: int = -1  # Use all layers on GPU

        if not model_name:
            raise ValueError("LlamaCpp model is not specified.")

        # Check memory before loading
        self.print_gpu_memory_status("Before tokenizer loading - ")

        try:
            model = Llama.from_pretrained(
                repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
                filename="Llama-3.2-3B-Instruct-Q4_0.gguf",
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                n_ctx=n_ctx,
                verbose=verbose,
            )

            self.print_gpu_memory_status("After tokenizer loading - ")
            return model
        except Exception as e:
            logfire.info(f"Error loading model: {e}")
            raise e

    def load_llama_cpp_model(
        self,
        embeddings: bool = False,
    ) -> object:
        """Loads the LlamaCpp model with memory management."""

        from llama_cpp import Llama

        model_name: str = self.hf_model
        hf_model_fn: str = self.hf_model_fn
        context_token_length: int = self.context_token_length
        aggressive_cleanup: bool = self.aggressive_cleanup
        retry_on_oom: bool = self.retry_on_oom
        # model config
        seed: int = 1337
        n_batch: int = 2048
        n_ubatch: int = 1024
        n_gpu_layers: int = -1  # Use all layers on GPU
        verbose: bool = False  # Reduce output spam

        if not model_name:
            raise ValueError("LlamaCpp model is not specified.")

        # Check memory before loading
        self.print_gpu_memory_status("Before model loading - ")

        # Clear any existing models first
        self.clear_model_from_cache()

        try:
            model = Llama.from_pretrained(
                repo_id=model_name,
                filename=hf_model_fn,
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                embeddings=embeddings,
                n_ctx=context_token_length,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                verbose=verbose,
            )

            self.print_gpu_memory_status("After model loading - ")
            return model

        except Exception as e:
            logfire.info(f"Error loading model: {e}")
            # Try with fewer GPU layers if out of memory
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logfire.info("Attempting to load with fewer GPU layers...")
                self.clear_model_from_cache()
                if aggressive_cleanup:
                    self.force_memory_cleanup()

                if retry_on_oom:
                    try:
                        model = Llama.from_pretrained(
                            repo_id=model_name,
                            filename=hf_model_fn,
                            n_gpu_layers=n_gpu_layers,  # Try to use all layers on GPU or reduce if needed
                            seed=seed,
                            embeddings=embeddings,
                            n_ctx=context_token_length,
                            n_batch=n_batch,  # Reduced batch size
                            n_ubatch=n_ubatch,  # Reduced micro batch size
                            verbose=verbose,
                        )
                        logfire.info("Successfully loaded with reduced GPU layers")
                        return model
                    except Exception as e2:
                        logfire.info(f"Failed to load with reduced settings: {e2}")
                        raise e2
            raise e

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

        logfire.info("Memory cleared successfully")

    def generate_openai(self) -> str:
        """Generates a response using the OpenAI LLM."""
        model = self.model
        conversation_messages: list[dict[str, str]] = self.conversation_messages

        try:
            return model.invoke(conversation_messages)

        except (IndexError, AttributeError):
            logfire.info("Error accessing LLM response")
            return ""

    def generate_ollama(self) -> str:
        """Generates a response using the Ollama LLM."""
        model = self.model
        conversation_messages: list[dict[str, str]] = self.conversation_messages

        try:
            return model.invoke(conversation_messages)

        except (IndexError, AttributeError):
            logfire.info("Error accessing LLM response")
            return ""

    def generate_llama_cpp(self) -> str:
        """Generates a response using the Hugging Face LLM."""
        model = self.model
        conversation_messages: list[dict[str, str]] = self.conversation_messages
        # model config:
        seed: int = 1337
        temperature: float = 0.7
        max_tokens: int = 2048
        # top_p: float = 0.9
        # top_k: int = 50
        # repeat_penalty: float = 1.2

        try:
            response = model.create_chat_completion(
                messages=conversation_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                # top_p=top_p,
                # top_k=top_k,
                # repeat_penalty=repeat_penalty,
                # response_format={
                #     "type": "json_object",
                # }
            )
            return response

        except (IndexError, AttributeError):
            logfire.info("Error accessing LLM response")
            return {"error": "Error accessing LLM response"}

    def create_conversation_messages(self) -> list[dict[str, str]]:
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

        conversation_messages: list[dict[str, str]] = self.conversation_messages

        if new_message:
            # Add the new message to the conversation
            conversation_messages.append(
                {"role": role, "content": (new_message)}
            )

    def verify_conversation_messages_length(self, prompt_tokens: int) -> bool:
        """Verifies if the conversation messages length is within the limit."""

        user_context_length: int = self.user_context_length
        max_context_length: int = self.max_context_length
        reserved_prompt_length: int = self.reserved_prompt_length

        logfire.info("Verifying conversation messages length...")
        conversation_length: int = user_context_length + prompt_tokens + 256  # buffer for system prompt

        if conversation_length > (
                max_context_length - reserved_prompt_length):
            logfire.info(f"Conversation messages length exceeds the limit:\
                {conversation_length}")
            return True, conversation_length

        logfire.info(f"Conversation messages length is within the limit:\
            {conversation_length}")

        return False, conversation_length

    def init_model_tokenizer(self) -> None:
        """Initializes the model and tokenizer based on the backend."""

        backend: str = self.backend

        match backend:
            case "ollama":
                self.tokenizer = self.load_ollama_embeddings_function()

            case "llama_cpp":
                self.tokenizer = self.load_llama_cpp_model_tokenizer()

    def truncate_text(self, tokens: str, conversation_length: int) -> str:
        """Truncates the text to fit within the context token length."""

        max_context_length: int = self.max_context_length
        reserved_prompt_length: int = self.reserved_prompt_length
        backend: str = self.backend
        tokenizer = self.tokenizer

        token_count = len(tokens)
        logfire.info(f"Token count: {token_count}")
        logfire.info(f"Max length: {max_context_length}")

        exceeded_length = conversation_length - (
            max_context_length - reserved_prompt_length)
        to_truncate = max(0, token_count - exceeded_length)

        truncated_tokens = tokens[:to_truncate]

        match backend:
            case "ollama":
                truncated_text = tokenizer.decode(truncated_tokens)
            case "llama_cpp":
                truncated_text = tokenizer.detokenize(truncated_tokens)

        logfire.info(f"Truncated text length: {len(truncated_text)}")
        return truncated_text.decode("utf-8")

    def clear_model_from_cache(self) -> None:
        """Clears the model cache to free up memory."""

        # Clear model references
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        # Force garbage collection
        gc.collect()

        # Clear GPU memory if available
        if torch.cuda.is_available():
            logfire.info("Clearing GPU memory...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Synchronize to ensure all operations are complete
            torch.cuda.synchronize()

        # Wait for memory to be fully released
        sleep(2)

    def clear_model_tokenizer_from_cache(self) -> None:
        """Clears the tokenizer cache to free up memory."""

        # Clear tokenizer references
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        gc.collect()

        # Clear GPU memory if available
        if torch.cuda.is_available():
            logfire.info("Clearing GPU memory...")
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
        logfire.info(f"Number of CUDA devices: {device_count}")

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
            logfire.info(f"Device {i}: {torch.cuda.get_device_name(device)}")
            total_memory = torch.cuda.get_device_properties(device).total_memory
            logfire.info(f"  - Total Memory: {total_memory / (1024**3):.2f} GB")
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
            logfire.info("Warning: Cannot check GPU memory, CUDA not available")
            return float(0)
        return memory_info['free_memory_gb']

    def print_gpu_memory_status(self, prefix: str = "") -> None:
        """Print current GPU memory status."""

        memory_info = self.get_gpu_memory_info()
        if "error" in memory_info:
            logfire.info(f"{prefix}GPU Memory: {memory_info['error']}")
            return

        logfire.info(f"{prefix}GPU Memory Status:")
        logfire.info(f"  - Total: {memory_info['total_memory_gb']:.2f} GB")
        logfire.info(f"  - Allocated: {memory_info['allocated_memory_gb']:.2f} GB")
        logfire.info(f"  - Free: {memory_info['free_memory_gb']:.2f} GB")
        logfire.info(f"  - Usage: {memory_info['memory_usage_percent']:.1f}%")

    def check_available_memory(self, required_memory_gb: float = 8.0) -> bool:
        """Check if enough GPU memory is available for model loading."""

        memory_info = self.get_gpu_memory_info()
        if "error" in memory_info:
            logfire.info("Warning: Cannot check GPU memory, CUDA not available")
            return True  # Assume it's okay if we can't check

        available_memory = memory_info['free_memory_gb']
        if available_memory < required_memory_gb:
            logfire.info(f"Warning: Only {available_memory:.2f} GB available, "
                         f"but {required_memory_gb:.2f} GB required")
            return False

        logfire.info(f"Memory check passed: {available_memory:.2f} GB available")
        return True

    def force_memory_cleanup(self) -> None:
        """Force aggressive memory cleanup."""

        logfire.info("Performing aggressive memory cleanup...")

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

        logfire.info("Aggressive cleanup completed")
        self.print_gpu_memory_status("After cleanup - ")

    def evaluate_gpu_memory_requirements(self) -> float:
        """Evaluates the GPU memory requirements based on model size
        and context length."""

        context_token_length: int = self.context_token_length
        model_size_gb: float = self.model_size_gb
        model_memory_usage_1k_token: float = self.model_memory_usage_1k_token

        context_memory_gb = (
            (context_token_length / 1000) * model_memory_usage_1k_token
        )
        total_memory_gb = model_size_gb + context_memory_gb

        logfire.info(f"Estimated GPU memory requirement: {total_memory_gb:.2f} GB")
        return float(total_memory_gb)

    def validate_user_input_files(self) -> dict:
        """Validates that all user input files exist."""

        user_input: list[str] = self.user_input
        backend: str = self.backend
        tokenizer = self.tokenizer  # tokenizer based on backend

        input_prompts = {}

        for file in user_input:
            fn = os.path.basename(file)
            logfire.info(f"Validating input file: {fn}")
            if not os.path.exists(file):
                raise FileNotFoundError(f"Input file {file} does not exist.")

            with open(file, "r") as f:
                text = f.read().strip()
                if not text:
                    raise ValueError(f"Input file {file} is empty.")

            match backend:
                case "openAI":
                    logfire.info("OpenAI backend selected, skipping token count for files.")
                    tokens = 0  # Skip token counting for OpenAI
                case "ollama":
                    tokens = len(tokenizer.embed_query(text.encode("utf-8")))
                case "llama_cpp":
                    tokens = len(tokenizer.tokenize(text.encode("utf-8")))

            input_prompts[fn] = {"content": text, "tokens": tokens}
            logfire.info(f"{fn} has {tokens} tokens.")
            input_prompts["total_tokens"] = input_prompts.get("total_tokens", 0) + tokens

        return input_prompts

    def validate_user_input_context(self) -> Union[int, list[int]]:
        """Validates that the user input context is not empty."""

        inputs: str = self.inputs
        backend: str = self.backend
        tokenizer = self.tokenizer  # tokenizer based on backend

        if not inputs or not inputs.strip():
            raise ValueError("User input context is empty.")

        match backend:
            case "ollama":
                tokens = tokenizer.embed_query(inputs.encode("utf-8"))
            case "llama_cpp":
                tokens = tokenizer.tokenize(inputs.encode("utf-8"))

        logfire.info(f"User input context has {tokens} tokens.")
        return len(tokens), tokens

    def openai(self) -> None:
        """Generates responses using the OpenAI LLM."""
        user_prompts: list[str] = self.user_input
        inputs: str = self.inputs
        total_pipeline_duration = logfire.metric_histogram(
            "total_pipeline_duration_ms",
            unit="ms",
            description="Duration of total pipeline processing in milliseconds",
        )
        # count time
        start_time_total = time.perf_counter()
        logfire.info("5: Starting OpenAI generation...")

        with logfire.span("create conversation messages:"):
            self.conversation_messages = self.create_conversation_messages()
            self.update_conversation_messages(new_message=inputs, role="user")
            logfire.info("6: System Msg. and Inputs set.")

        with logfire.span("process prompt files:"):
            prompts_info = self.validate_user_input_files()

        with logfire.span("Load OpenAI model:"):
            self.model = self.load_openai_model()

        with logfire.span("process each prompt file:"):
            for i, file in enumerate(user_prompts):
                logfire.info(f"Processing prompt {i + 1}/{len(user_prompts)}")
                fn_load = os.path.basename(file)

                with logfire.span("update conversation messages:"):
                    text = prompts_info[fn_load]["content"]
                    # text_tokens = prompts_info[fn_load]["tokens"]  # number of tokens in the prompt
                    self.update_conversation_messages(
                        new_message=text,
                        role="user"
                    )
                    logfire.info("10: Adding User prompt to messages.")

                try:
                    logfire.info("11: Generating response...")
                    response = self.generate_openai()
                    logfire.info("12: Response gernerated.")
                except Exception as e:
                    logfire.info(f"Error generating response: {e}")
                    response = "Error generating response."

                with logfire.span("Saving response to file:"):
                    logfire.info("13: Saving response...")
                    fn_prompt = file.split("/")[-1].replace(".txt", "")
                    fn_model = self.openai_model.replace("/", "-")
                    fn_out_dir = self.output_dir
                    fn_keyword = self.keyword
                    fn_name = f"{fn_keyword}_{fn_prompt}_{fn_model}.txt"
                    fn = os.path.join(fn_out_dir, fn_name)

                    try:
                        with open(fn, "w") as file:
                            file.write(response.text())
                        logfire.info(f"14: Response saved to {fn}")
                    except Exception as e:
                        logfire.info(f"Error writing response to file: {e}")
                        logfire.info("Response:", response)

                with logfire.span("update conversation messages:"):
                    try:
                        self.update_conversation_messages(
                            new_message=response.text(),
                            role="assistant"
                        )
                    except Exception as e:
                        logfire.info(f"Error updating conversation messages: {e}")
                sleep(5)

        with logfire.span(f"Completing generation for keyword: {self.keyword}"):
            # count time
            end_time_total = time.perf_counter()
            elapsed_time_total = (end_time_total - start_time_total)
            logfire.info(f"Time taken to process {file}: {elapsed_time_total:.2f} seconds")
            total_pipeline_duration.record(elapsed_time_total)
            self.conversation_messages.append({
                "role": "system",
                "content": "OpenAI generation completed.",
                "elapsed_time_seconds": elapsed_time_total
            })
            logfire.info("11: System Msg. added.")
            logfire.info("OpenAI generation completed.")

    def ollama(self) -> None:
        """Generates responses using the Ollama LLM."""
        reserved_prompt_length: int = self.reserved_prompt_length
        enable_memory_monitoring: bool = self.enable_memory_monitoring
        user_prompts: list[str] = self.user_input
        inputs: str = self.inputs
        total_pipeline_duration = logfire.metric_histogram(
            "total_pipeline_duration_ms",
            unit="ms",
            description="Duration of total pipeline processing in milliseconds",
        )
        # count time
        start_time_total = time.perf_counter()
        logfire.info("5: Starting LlamaCpp generation...")

        with logfire.span("create conversation messages:"):
            self.conversation_messages = self.create_conversation_messages()
            self.update_conversation_messages(new_message=inputs, role="user")
            logfire.info("6: System Msg. and Inputs set.")

        with logfire.span("initialize tokenizer:"):
            try:
                self.init_model_tokenizer()
                logfire.info("7: Tokenizer initialized.")
            except Exception as e:
                logfire.info(f"Error initializing tokenizer: {e}")

        with logfire.span("process prompt files:"):
            prompts_info = self.validate_user_input_files()
            # get token count for all prompts
            prompt_tokens = prompts_info.get("total_tokens", 0)

        with logfire.span("validate user input context:"):
            if self.ollama_model in ["gemma3:27b"]:
                self.context_token_length = 35000
            else:
                _, length = self.verify_conversation_messages_length(prompt_tokens)
                self.context_token_length = (
                    length + reserved_prompt_length)
                logfire.info(f"Context token length: {self.context_token_length}")
            self.model = self.load_ollama_model()

        with logfire.span("process each prompt file:"):
            for i, file in enumerate(user_prompts):
                logfire.info(f"Processing prompt {i + 1}/{len(user_prompts)}")
                fn_load = os.path.basename(file)

                if enable_memory_monitoring:
                    self.print_gpu_memory_status(f"Before iteration {i + 1} - ")

                with logfire.span("update conversation messages:"):
                    text = prompts_info[fn_load]["content"]
                    # text_tokens = prompts_info[fn_load]["tokens"]  # number of tokens in the prompt
                    self.update_conversation_messages(
                        new_message=text,
                        role="user"
                    )
                    logfire.info("10: Adding User prompt to messages.")

                try:
                    logfire.info("11: Generating response...")
                    response = self.generate_ollama()
                    logfire.info("12: Response gernerated.")
                except Exception as e:
                    logfire.info(f"Error generating response: {e}")
                    response = "Error generating response."

                with logfire.span("Saving response to file:"):
                    logfire.info("13: Saving response...")
                    fn_prompt = file.split("/")[-1].replace(".txt", "")
                    fn_model = self.ollama_model.replace("/", "-")
                    fn_out_dir = self.output_dir
                    fn_keyword = self.keyword
                    fn_name = f"{fn_keyword}_{fn_prompt}_{fn_model}.txt"
                    fn = os.path.join(fn_out_dir, fn_name)

                    try:
                        with open(fn, "w") as file:
                            file.write(response)
                        logfire.info(f"14: Response saved to {fn}")
                    except Exception as e:
                        logfire.info(f"Error writing response to file: {e}")
                        logfire.info("Response:", response)

                with logfire.span("update conversation messages:"):
                    self.update_conversation_messages(
                        new_message=response,
                        role="assistant"
                    )
                sleep(5)

        with logfire.span(f"Completing generation for keyword: {self.keyword}"):
            # count time
            end_time_total = time.perf_counter()
            elapsed_time_total = (end_time_total - start_time_total)
            logfire.info(f"Time taken to process {file}: {elapsed_time_total:.2f} seconds")
            total_pipeline_duration.record(elapsed_time_total)
            self.conversation_messages.append({
                "role": "system",
                "content": "Ollama generation completed.",
                "elapsed_time_seconds": elapsed_time_total
            })
            logfire.info("11: System Msg. added.")
            logfire.info("Ollama generation completed.")

    def llama_cpp(self) -> None:
        """Generates responses using the LlamaCpp LLM
        with improved memory management."""
        aggressive_cleanup: bool = self.aggressive_cleanup
        retry_on_oom: bool = self.retry_on_oom
        max_context_length: int = self.max_context_length
        reserved_prompt_length: int = self.reserved_prompt_length
        enable_memory_monitoring: bool = self.enable_memory_monitoring
        user_prompts: list[str] = self.user_input
        inputs: str = self.inputs
        total_pipeline_duration = logfire.metric_histogram(
            "total_pipeline_duration_ms",
            unit="ms",
            description="Duration of total pipeline processing in milliseconds",
        )
        # count time
        start_time_total = time.perf_counter()
        logfire.info("5: Starting LlamaCpp generation...")

        with logfire.span("create conversation messages:"):
            self.conversation_messages = self.create_conversation_messages()
            self.update_conversation_messages(
                new_message=inputs,
                role="user"
            )
            logfire.info("6: System Msg. and Inputs set.")

        with logfire.span("initialize tokenizer:"):
            try:
                self.init_model_tokenizer()
                logfire.info("7: Tokenizer initialized.")
            except Exception as e:
                logfire.info(f"Error initializing tokenizer: {e}")

        with logfire.span("process prompt files:"):
            prompts_info = self.validate_user_input_files()
            # get token count for all prompts
            prompt_tokens = prompts_info.get("total_tokens", 0)
            self.user_context_length, self.context_tokens = self.validate_user_input_context()

            with logfire.span("verify token length and truncate if needed:"):
                # truncate the input text to fit within the context length
                exceeded, length = self.verify_conversation_messages_length(prompt_tokens)
                logfire.info(f"8: Exceeded: {exceeded}, Length: {length}")

                if exceeded:
                    logfire.info("Context length exceeded, truncating...")
                    inputs = self.truncate_text(self.context_tokens, length)
                    self.conversation_messages[1]["content"] = (inputs)
                    self.context_token_length = max_context_length
                else:
                    self.context_token_length = (
                        length + reserved_prompt_length
                    )

            with logfire.span("load model and keep alive for all prompts:"):
                try:
                    self.model = self.load_llama_cpp_model()
                    logfire.info("9: Model loaded and ready.")
                except Exception as e:
                    logfire.info(f"Error loading model: {e}")
                    return  # Exit if model loading fails

            for i, file in enumerate(user_prompts):
                logfire.info(f"Processing prompt {i + 1}/{len(user_prompts)}")
                fn_load = os.path.basename(file)

                if enable_memory_monitoring:
                    self.print_gpu_memory_status(f"Before iteration {i + 1} - ")

                with logfire.span("update conversation messages:"):
                    text = prompts_info[fn_load]["content"]
                    # text_tokens = prompts_info[fn_load]["tokens"]  # number of tokens in the prompt

                    self.update_conversation_messages(
                        new_message=text,
                        role="user"
                    )
                    logfire.info("10: Adding User prompt to messsages.")

                with logfire.span("memory check before model loading:"):
                    if enable_memory_monitoring and i > 0:
                        required_memory = self.evaluate_gpu_memory_requirements()
                        logfire.info(f"Required memory for loading model:\
                            {required_memory:.2f} GB")

                        # Check if we have enough memory before loading model
                        if not self.check_available_memory(
                                required_memory_gb=required_memory):
                            logfire.info("Insufficient memory detected,\
                                performing cleanup...")

                            self.clear_model_from_cache()
                            self.clear_model_tokenizer_from_cache()
                            if aggressive_cleanup:
                                self.force_memory_cleanup()

                            # Wait a bit more for memory to be released
                            sleep(3)

                            # Check again
                            if not self.check_available_memory(
                                    required_memory_gb=required_memory):
                                logfire.info("Still insufficient memory,\
                                    using smaller context...")
                                self.context_token_length = min(
                                    self.context_token_length, 4096)

                with logfire.span("text generation:"):
                    inference_duration = logfire.metric_histogram(
                        f"inference_duration_ms_{i}",
                        unit="ms",
                        description="Duration of model inference in milliseconds",
                    )

                    # count time
                    start_time = time.perf_counter()
                    logfire.info("11: Starting LlamaCpp generation...")

                    try:
                        # generate response from the LlamaCpp model
                        response = self.generate_llama_cpp()
                        logfire.info("12: Response generated.")

                    except Exception as e:
                        logfire.info(f"Error during model loading or generation: {e}")
                        if "out of memory" in str(e).lower():
                            logfire.info("OOM detected, performing aggressive cleanup and\
                                retrying...")
                            self.clear_model_from_cache()
                            self.clear_model_tokenizer_from_cache()
                            if aggressive_cleanup:
                                self.force_memory_cleanup()
                            sleep(5)

                            if retry_on_oom:
                                # Reduce context length and try again
                                self.context_token_length = min(
                                    self.context_token_length // 2, 2048)

                                try:
                                    self.model = self.load_llama_cpp_model()
                                    logfire.info("9/a: Model re-loaded and ready.")
                                    response = self.generate_llama_cpp()
                                    logfire.info("12: Response generated after OOM recovery.")
                                except Exception as e2:
                                    logfire.info(f"Failed to recover from OOM: {e2}")
                                    continue  # Skip this file and move to next
                        else:
                            logfire.info(f"Non-memory related error: {e}")
                            continue

                    # count time
                    end_time = time.perf_counter()
                    elapsed_time = (end_time - start_time) * 1000  # convert to milliseconds
                    logfire.info(f"Time taken to process {file}: {elapsed_time:.2f} ms")
                    response["elapsed_time"] = elapsed_time
                    inference_duration.record(elapsed_time)

                with logfire.span("save response to file:"):
                    # save the response to a file
                    fn_prompt = file.split("/")[-1].replace(".txt", "")
                    fn_model = self.hf_model.replace("/", "-")
                    fn_out_dir = self.output_dir
                    fn_keyword = self.keyword
                    fn_name = f"{fn_keyword}_{fn_prompt}_{fn_model}.json"
                    fn = os.path.join(fn_out_dir, fn_name)

                    try:
                        with open(fn, "w") as file:
                            json.dump(response, file, indent=4)
                        logfire.info(f"Response saved to {fn}")

                    except Exception as e:
                        logfire.info(f"Error writing response to file: {e}")
                        logfire.info("Response:", response)

                with logfire.span("update conversation messages with response:"):
                    response = response.get(
                        "choices", [{}])[0].get("message", {}).get("content", "")

                    self.update_conversation_messages(
                        new_message=response,
                        role="assistant"
                    )
                    logfire.info("12: Updated conversation messages with response.")

                logfire.info(f"13: Iteration {i + 1} completed for file: {file}")

                # Memory status after iteration
                self.print_gpu_memory_status(f"After iteration {i + 1} - ")

                # Additional wait between iterations for memory stabilization
                if i < len(self.user_input) - 1:  # Don't wait after last iteration
                    logfire.info("Waiting for memory stabilization...")
                    sleep(3)

        with logfire.span(f"Completing generation for keyword: {self.keyword}"):
            # count time
            end_time_total = time.perf_counter()
            elapsed_time_total = (end_time_total - start_time_total)
            total_pipeline_duration.record(elapsed_time_total)
            self.conversation_messages.append({
                "role": "system",
                "content": "LlamaCpp generation completed.",
                "elapsed_time_seconds": elapsed_time_total
            })

            logfire.info("=== LlamaCpp generation completed ===")
            self.print_gpu_memory_status("Final GPU Memory - ")
            logfire.info(f"Total time taken for LlamaCpp generation: {elapsed_time_total:.2f} seconds")
