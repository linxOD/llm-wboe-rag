import os
import json
from langchain_chroma import Chroma
from pydantic import BaseModel
from typing import Generator


class WboeLoadVectorstore(BaseModel):

    collection_name: str = "wboe_word_embeddings"
    vector_store_filepath_name: str = "chroma_langchain_db_wboe_embeddings"
    jwt_token: str = os.getenv("OLLAMA_API_KEY")
    hf_token: str = os.getenv("HUGGINGFACE_API_KEY")
    vector_store: list = None
    output_dir: str = "output"
    vector_store_filepath: str = ""
    context_yield: str = None
    embeddings_yield: list = None

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

        self.vector_store_filepath = os.path.join(
            self.output_dir, self.vector_store_filepath_name)

        if not os.path.exists(self.vector_store_filepath):
            raise FileNotFoundError(f"""Vector store path \
                                    {self.vector_store_filepath}\
                                    does not exist.""")

        self.vector_store = self.load_persited_vectorstore()
        if not self.vector_store:
            raise ValueError("Failed to initialize the vector store.")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def unloading_models_and_clear_up_memory(self) -> None:
        """Unloads the Hugging Face model and tokenizer.
        This is necessary to free up GPU memory."""

        self.vector_store = None

    def load_persited_vectorstore(self) -> Chroma:

        vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.vector_store_filepath,
        )
        return vector_store

    def load_vectorstore_schema(self) -> dict:

        file_path = f"{self.vector_store_filepath}/schema.json"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vector store schema file {file_path}\
                does not exist.")

        with open(file_path, "r") as file:
            schema = json.load(file)

        return schema

    def yield_documents(self) -> Generator[dict[str, str],
                                           None, None]:

        schema = self.load_vectorstore_schema()

        for _, value in schema.items():
            self.keyword = value.get("keyword", "unknown")

            context = self.vector_store.get(
                include=["documents", "embeddings"],
                where={"keyword": self.keyword})

            self.context_yield = context["documents"][0]
            self.embeddings_yield = context["embeddings"][0]

            yield {
                "keyword": self.keyword,
                "context": self.context_yield,
                "embeddings": self.embeddings_yield
            }
