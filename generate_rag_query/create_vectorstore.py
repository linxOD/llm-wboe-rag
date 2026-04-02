import os
import json
import torch
from glob import glob
from uuid import uuid4
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
# from transformers import AutoTokenizer
# from langchain.document_loaders import DirectoryLoader
from pydantic import BaseModel
from typing import Literal


class WboeCreateVectorstore(BaseModel):

    backend: Literal["ollama", "hf_pipeline", "llama_cpp"] = "ollama"
    ollama_model: str = "llama3.2:3b"
    hf_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    collection_name: str = "wboe_word_embeddings"
    vectore_store_dir: str = "chroma_langchain_db_wboe_embeddings"
    documents_path: str = "../dboe-data-prep/output/llm_corpus"
    documents_file_type: str = "md"
    jwt_token: str = os.getenv("OLLAMA_API_KEY")
    hf_token: str = os.getenv("HUGGINGFACE_API_KEY")
    keyword: str = "grob"
    output_dir: str = "output"
    vectore_store_filepath_name: str = ""
    exclude_files: list[str] = ["Fleisch"]
    include_files: list[str] = ["Gefrette", "kauschen", "Gigel", "kommod"]

    def __init__(self, **data):
        super().__init__(**data)

        self.init_env()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.vectore_store_filepath_name = os.path.join(
            self.output_dir, self.vectore_store_dir)

    def init_env(self):

        if not self.jwt_token and self.backend == "ollama":
            raise ValueError("JWT token for Ollama API is not set.\
                Please set the OLLAMA_API_KEY environment variable.")

        if not self.hf_token and self.backend in ["hf_pipeline", "llama_cpp"]:
            raise ValueError("Hugging Face token is not set.\
                Please set the HUGGINGFACE_API_KEY environment variable.")

        if not self.ollama_model:
            raise ValueError("Ollama model is not specified.")

        if not self.hf_model:
            raise ValueError("Hugging Face model is not specified.")

        if not os.path.exists(self.documents_path):
            raise FileNotFoundError(f"Documents path {self.documents_path}\
                does not exist.")

    def init_documents_loader(self, save=False) -> list[Document] | list[str]:

        documents = {}
        schema = {}
        file_glob = glob(os.path.join(self.documents_path,
                                      f"*.{self.documents_file_type}"))
        for file in file_glob:
            filename = os.path.basename(file).replace(
                f".{self.documents_file_type}", "")
            if (self.exclude_files[0] != "none" and
                    filename in self.exclude_files):
                print(f"Excluding file {filename} from processing.")
                continue
            if (self.include_files[0] != "all" and
                    filename not in self.include_files):
                print(f"Excluding file {filename} for processing.")
                continue
            file_id = str(uuid4())
            doc = TextLoader(file, encoding="utf-8")
            document = doc.load()
            if not document:
                print(f"No content found in file {filename}. Skipping.")
                continue
            # print(document)
            document[0].id = file_id
            document[0].metadata = {
                "source": file,
                "keyword": filename,
                "id": file_id,
            }
            schema[file_id] = {
                "source": file,
                "keyword": filename,
                "id": file_id,
            }
            documents[file_id] = document[0]

        if save:
            with open(os.path.join(self.vectore_store_filepath_name,
                                   "schema.json"), "w") as file:
                json.dump(schema, file, indent=4)

        return list(documents.values()), list(documents.keys())

    def add_documents_to_vectorstore(self, vector_store):

        documents, keys = self.init_documents_loader(save=True)
        if not documents:
            print("No documents found to add to the vector store.")
            return

        vector_store.add_documents(documents=documents, ids=keys)
        print(f"Added {len(documents)} documents to the vector store.")

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
            num_ctx=128000,
            # temperature=0.4,
        )
        return embeddings

    def load_huggingface_embeddings_function(self):
        """Loads the Hugging Face embeddings function."""
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = self.hf_model
        model_kwargs = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        return hf_embeddings

    def load_llama_cpp_embeddings_function(self):
        """Loads the LlamaCpp embeddings function."""

        from llama_cpp import Llama

        if not self.hf_model:
            raise ValueError("LlamaCpp model is not specified.")

        llm = Llama.from_pretrained(
            repo_id=self.hf_model,
            filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
            n_ctx=128000,
            n_gpu_layers=-1,
            seed=1337,
            embedding=True,
        )

        return llm

    def load_persited_vectorstore_llama_cpp(self):

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.load_llama_cpp_embeddings_function(),
            persist_directory=self.vectore_store_filepath_name,
        )
        return vector_store

    def load_persited_vectorstore_ollama(self):

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.load_ollama_embeddings_function(),
            persist_directory=self.vectore_store_filepath_name,
        )
        return vector_store

    def load_persited_vectorstore_hugginface(self):

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.load_huggingface_embeddings_function(),
            persist_directory=self.vectore_store_filepath_name,
        )
        return vector_store

    def ollama(self):
        vector_store = self.load_persited_vectorstore_ollama()
        if not vector_store:
            raise ValueError("Failed to initialize the vector store.")
        self.add_documents_to_vectorstore(vector_store)

    def huggingface(self):
        vector_store = self.load_persited_vectorstore_hugginface()
        if not vector_store:
            raise ValueError("Failed to initialize the vector store.")
        self.add_documents_to_vectorstore(vector_store)

    def llama_cpp(self):
        vector_store = self.load_persited_vectorstore_llama_cpp()
        if not vector_store:
            raise ValueError("Failed to initialize the vector store.")
        self.add_documents_to_vectorstore(vector_store)

    def main(self):
        if self.backend == "ollama":
            self.ollama()
        elif self.backend == "hf_pipeline":
            self.huggingface()
        elif self.backend == "llama_cpp":
            self.llama_cpp()


if __name__ == "__main__":
    wboe_embeddings = WboeCreateVectorstore(
        backend="hf_pipeline",
        ollama_model="llama3.2:3b",
        # hf_model="lmstudio-community/Llama-3.3-70B-Instruct-GGUF",
        hf_model="sentence-transformers/all-mpnet-base-v2",
        collection_name="wboe_word_embeddings",
        vectore_store_dir="chroma_langchain_db_wboe_embeddings",
        documents_path="./llm_corpus",
        documents_file_type="toon",
        exclude_files=["none"],
        include_files=["all"],
        jwt_token=os.getenv("OLLAMA_API_KEY"),
        hf_token=os.getenv("HUGGINGFACE_API_KEY"),
        keyword="grob",
        output_dir="output"
    )
    wboe_embeddings.main()
    # clear cuda cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("CUDA cache cleared.")
    print("WBOE word embeddings initialized successfully.")
