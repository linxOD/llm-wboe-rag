import os
import json
import torch
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
    max_context_length: int = 2048
    output_dir: str = "output"
    model: object = None
    tokenizer: object = None
    pipe: object = None
    inputs: str = ""

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

        if not self.jwt_token:
            raise ValueError("JWT token for Ollama API is not set.\
                Please set the OLLAMA_API_KEY environment variable.")

        if not self.hf_token:
            raise ValueError("Hugging Face token is not set.\
                Please set the HUGGINGFACE_API_KEY environment variable.")

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
            num_ctx=self.max_context_length,
            # temperature=0.4,
        )
        return embeddings

    def load_llama_cpp_embeddings_function(self):
        """Loads the LlamaCpp embeddings function."""

        from llama_cpp import Llama

        if not self.hf_model:
            raise ValueError("LlamaCpp model is not specified.")

        llm = Llama.from_pretrained(
            repo_id=self.hf_model,
            filename=self.hf_model_fn,
            n_ctx=self.max_context_length,
            n_batch=self.max_context_length,
            n_ubatch=self.max_context_length,
            n_threads=3,
            n_gpu_layers=-1,
            seed=1337,
            embedding=True,
        )

        return llm

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
            num_ctx=self.max_context_length,
            temperature=0.4,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
        )

        return llm

    def load_llama_cpp_model(self):
        """Loads the LlamaCpp model."""

        from llama_cpp import Llama

        if not self.hf_model:
            raise ValueError("LlamaCpp model is not specified.")

        llm = Llama.from_pretrained(
            repo_id=self.hf_model,
            filename=self.hf_model_fn,
            n_ctx=self.max_context_length,
            n_batch=2048,
            n_ubatch=512,
            n_gpu_layers=-1,
            seed=1337,
        )

        return llm

    def load_huggingface_model(self):
        """Loads the Hugging Face model and tokenizer."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.hf_token:
            raise ValueError("HUGGINGFACE_API_KEY environment\
                variable is not set.")

        if not self.hf_model:
            raise ValueError("Hugging Face model is not specified.")

        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_model,
            token=self.hf_token,
            device_map="auto",  # Use GPU if available
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model,
            token=self.hf_token
        )

        # tokenizer.add_special_tokens({
        #     "eos_token": "<|eot_id|>",
        #     "pad_token": "<|pad_id|>",
        #     "bos_token": "<|begin_of_text|>"
        # })

        return model, tokenizer

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
            max_length=self.max_context_length,
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

        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        if self.tokenizer is not None:
            del self.tokenizer
            torch.cuda.empty_cache()
        self.pipe = None
        self.model = None
        self.tokenizer = None
        self.user_input = None
        self.keyword = None
        self.output_dir = None

    def generate_ollama(self) -> str:
        """Generates a response using the Ollama LLM."""

        system_prompt = "Du bist ein Lexicograph und ein Experte für\
            Wortbedeutungen für deutsch bairischen Dialekt."

        context_prompt = f"Verwende ausschließlich diese Daten:\n\n\
        {self.inputs}\n\n\
        Antworte auf den folgenden Prompt:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": self.user_input}
        ]

        response = self.model.invoke(messages)

        try:
            # response = response.generations[0][0].text.strip()
            return response

        except (IndexError, AttributeError):
            print(response)
            print("Error accessing LLM response")
            return ""

    def generate_huggingface(self) -> str:
        """Generates a response using the Hugging Face LLM."""

        from langchain_core.messages import SystemMessage

        # llama_sepecial_tokens = {
        #     "eot_id": "<|eot_id|>",
        #     "start_header_id": "<|start_header_id|>",
        #     "end_header_id": "<|end_header_id|>",
        #     "begin_of_text": "<|begin_of_text|>"
        # }

        system_message_content = (
            "Du bist ein Assistent, der auf die Frage des Benutzers antwortet."
            "Du hast Zugriff auf den Kontext,\
                der aus einem Vektorstore abgerufen "
            "wurde. Deine Aufgabe ist es,\
                die Frage des Benutzers zu beantworten, "
            "indem du die bereitgestellten Informationen nutzt. "
            "Du solltest keine Informationen hinzufügen, die nicht im Kontext "
            "enthalten sind, und du solltest keine Annahmen treffen. "
            "\n\n"
            f"{self.inputs}"
        )

        # message = """
        #     <|begin_of_text|>\
        #     <|start_header_id|>system<|end_header_id|>\n\n\
        #     You are a helpful assistant.\n\n\
        #     Only use the following information:\n\n\
        #     {context}<|eot_id|>\
        #     <|start_header_id|>user<|end_header_id|>\n\n\
        #     {user_input}<|eot_id|>\n\n\
        #     <|start_header_id|>assistant<|end_header_id|>\n\n
        # """

        conversation_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": (self.user_input)},
            {"role": "assistant", "content": ("")},
        ]

        prompt = [SystemMessage(system_message_content)] + \
            conversation_messages

        # Run
        response = self.pipe.invoke(prompt)

        # response = self.pipe.generate(prompts=[message])

        try:
            return response

        except (IndexError, AttributeError):
            print(response)
            print("Error accessing LLM response")
            return ""

    def generate_llama_cpp(self) -> str:
        """Generates a response using the Hugging Face LLM."""

        system_message_content = (
            "Du bist ein Assistent, der auf die Frage des Benutzers antwortet."
            "Du hast Zugriff auf den Kontext,\
                der aus einem Vektorstore abgerufen "
            "wurde. Deine Aufgabe ist es,\
                die Frage des Benutzers zu beantworten, "
            "indem du die bereitgestellten Informationen nutzt. "
            "Du solltest keine Informationen hinzufügen, die nicht im Kontext "
            "enthalten sind, und du solltest keine Annahmen treffen. "
            "\n\n"
            f"{self.inputs}"
        )

        conversation_messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": (self.user_input)},
            {"role": "assistant", "content": ("")},
        ]

        # Run
        response = self.model.create_chat_completion(
            messages=conversation_messages,
            response_format={
                "type": "json_object",
            }
        )

        try:
            return response

        except (IndexError, AttributeError):
            print(response)
            print("Error accessing LLM response")
            return ""

    def ollama(self) -> None:
        """Generates responses using the Ollama LLM."""

        self.model = self.load_ollama_model()

        prompt_files = self.user_input

        for file in prompt_files:

            print(f"Processing file: {file}")

            with open(file, "r") as f:
                text = f.read().strip()

            self.user_input = text

            response = self.generate_ollama()

            # use the response as new context
            self.inputs += response

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

            sleep(10)  # Sleep to avoid rate limiting

    def huggingface(self) -> None:
        """Generates responses using the Hugging Face LLM."""

        # Load the Hugging Face model and tokenizer sperately
        # to allow manually unloading
        self.model, self.tokenizer = self.load_huggingface_model()

        # Create the Hugging Face pipeline
        self.pipe = self.load_huggingface_pipeline()

        prompt_files = self.user_input

        for file in prompt_files:

            with open(file, "r") as f:
                text = f.read().strip()

            self.user_input = text

            response = self.generate_huggingface()

            # use the response as new context
            self.inputs += response

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

            # Clear GPU memory after each file processing
            if torch.cuda.is_available():
                print("Clearing GPU memory...")
                torch.cuda.empty_cache()
            sleep(10)  # Sleep to avoid rate limiting

    def llama_cpp(self) -> None:
        """Generates responses using the LlamaCpp LLM."""

        # Load the LlamaCpp model
        self.model = self.load_llama_cpp_model()

        prompt_files = self.user_input

        for file in prompt_files:

            with open(file, "r") as f:
                text = f.read().strip()

            self.user_input = text

            response = self.generate_llama_cpp()

            # use the response as new context
            self.inputs += response

            # print(response)
            fn_prompt = file.split("/")[-1].replace(".txt", "")
            fn_model = self.hf_model.replace("/", "-")
            fn_out_dir = self.output_dir
            fn_keyword = self.keyword
            fn_name = f"{fn_keyword}_{fn_prompt}_{fn_model}.txt"
            fn = os.path.join(fn_out_dir, fn_name)

            try:
                with open(fn, "w") as file:
                    json.dump(response, file, indent=4)

            except Exception as e:
                print(f"Error writing response to file: {e}")
                print("Response:", response)

            # Clear GPU memory after each file processing
            if torch.cuda.is_available():
                print("Clearing GPU memory...")
                torch.cuda.empty_cache()
            sleep(5)
