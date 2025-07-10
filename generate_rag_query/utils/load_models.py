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
    max_context_length: int = 128000
    context_token_length: int = 2048
    reserved_prompt_length: int = 4096
    output_dir: str = "output"
    model: object = None
    tokenizer: object = None
    pipe: object = None
    inputs: str = ""
    conversation_messages: list[dict[str, str]] = []

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
        """Loads the LlamaCpp model."""

        from llama_cpp import Llama

        if not self.hf_model:
            raise ValueError("LlamaCpp model is not specified.")

        model = Llama.from_pretrained(
            repo_id=self.hf_model,
            filename=self.hf_model_fn,
            n_gpu_layers=-1,
            seed=1337,
            embeddings=embeddings,
            n_ctx=self.context_token_length,
            n_batch=2048,
            n_ubatch=512,
        )

        return model

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
        self.user_input_text = None
        self.keyword = None
        self.output_dir = None

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

        match self.backend:
            case "ollama":
                self.model = self.load_ollama_embeddings_function()
                conversation_length = sum(
                    len(self.model.embed_query(
                        msg["content"][0].encode("utf-8")
                        )) for msg in self.conversation_messages
                )
            case "hf_pipeline":
                # must be verified
                self.model, tokenizer = self.load_huggingface_model()
                conversation_length = sum(
                    len(tokenizer(
                        msg["content"][0].encode("utf-8"),
                        return_tensors="pt"
                        )["input_ids"]) for msg in self.conversation_messages
                )

            case "llama_cpp":
                self.model = self.load_llama_cpp_model()
                conversation_length = sum(
                    len(self.model.tokenize(
                        msg["content"][0].encode("utf-8")
                        )) for msg in self.conversation_messages
                )

        if conversation_length > (
                self.max_context_length - self.reserved_prompt_length
                ):
            print(f"Conversation messages length exceeds the limit:\
                {conversation_length}")
            return False, conversation_length

        print(f"Conversation messages length is within the limit:\
            {conversation_length}")
        return True, conversation_length

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

        return truncated_text

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
            _, length = self.verify_conversation_messages_length()
            self.context_token_length = length + self.reserved_prompt_length
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

    def llama_cpp(self) -> None:
        """Generates responses using the LlamaCpp LLM."""

        self.conversation_messages = self.create_conversation_messages()
        self.update_conversation_messages(new_message=self.inputs, role="user")

        for file in self.user_input:

            del self.model
            if torch.cuda.is_available():
                print("Clearing GPU memory...")
                torch.cuda.empty_cache()
            sleep(1)

            with open(file, "r") as f:
                text = f.read().strip()

            self.update_conversation_messages(new_message=text, role="user")

            # truncate the input text to fit within the context length
            exceeded, length = self.verify_conversation_messages_length()
            if exceeded:
                self.inputs = self.truncate_text(self.inputs, length)

                self.conversation_messages[1]["content"] = (self.inputs)

                self.context_token_length = self.max_context_length
            else:
                self.context_token_length = (
                    length + self.reserved_prompt_length
                )

            del self.model
            if torch.cuda.is_available():
                print("Clearing GPU memory...")
                torch.cuda.empty_cache()
            sleep(1)

            self.model = self.load_llama_cpp_model()
            # generate response from the LlamaCpp model
            response = self.generate_llama_cpp()

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

            except Exception as e:
                print(f"Error writing response to file: {e}")
                print("Response:", response)

            self.update_conversation_messages(
                new_message=response,
                role="assistant"
            )
            sleep(1)
