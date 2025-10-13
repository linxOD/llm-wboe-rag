import os
import logfire
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from generate_rag_query import WboeRAGPipeline

# Configure Logfire for logging
if os.getenv("LOGFIRE_TOKEN") is None:
    raise ValueError("LOGFIRE_TOKEN environment variable is not set.")

logfire.configure()

# Global variable to store running tasks
running_tasks: Dict[str, Dict[str, Any]] = {}


class RAGPipelineRequest(BaseModel):
    """Request model for RAG pipeline processing."""

    backend: str = Field(
        default="ollama",
        description="Backend type: 'ollama', 'llama_cpp', or 'openAI'"
    )

    # Model configurations
    ollama_model: Optional[str] = Field(
        default="llama3.2:3b",
        description="Ollama model name"
    )
    hf_model: Optional[str] = Field(
        default="bartowski/Llama-3.2-3B-Instruct-GGUF",
        description="HuggingFace model name"
    )
    hf_model_fn: Optional[str] = Field(
        default="Llama-3.2-3B-Instruct-Q4_0.gguf",
        description="HuggingFace model file name"
    )
    openai_model: Optional[str] = Field(
        default="gpt-4o",
        description="OpenAI model name"
    )

    # Vector store configuration
    collection_name: str = Field(
        default="wboe_word_embeddings",
        description="Vector store collection name"
    )
    vector_store_filepath_name: str = Field(
        default="chroma_langchain_db_wboe_embeddings",
        description="Vector store file path"
    )

    # Processing configuration
    user_input: List[str] = Field(
        default=["prompt1.txt", "prompt2.txt", "prompt3.txt", "prompt4.txt"],
        description="List of prompt files to use"
    )
    keywords_to_process: List[str] = Field(
        default=[],
        description="Specific keywords to process (empty list processes all)"
    )

    # Performance configuration
    max_context_length: int = Field(
        default=128000,
        description="Maximum context length in tokens"
    )
    model_memory_usage: float = Field(
        default=4.0,
        description="Model memory usage in GB"
    )
    gpu_memory_threshold: float = Field(
        default=0.9,
        description="Maximum GPU memory utilization (0.0-1.0)"
    )

    # Output configuration
    output_dir: str = Field(
        default="output",
        description="Output directory for results"
    )

    # Feature flags
    enable_memory_monitoring: bool = Field(
        default=True,
        description="Enable memory usage monitoring"
    )
    aggressive_cleanup: bool = Field(
        default=True,
        description="Enable aggressive memory cleanup"
    )
    retry_on_oom: bool = Field(
        default=True,
        description="Retry on out-of-memory errors"
    )


class RAGPipelineResponse(BaseModel):
    """Response model for RAG pipeline processing."""

    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Task status: 'started', 'running', 'completed', 'failed'")
    message: str = Field(description="Status message")
    created_at: datetime = Field(description="Task creation timestamp")
    backend: str = Field(description="Backend used for processing")
    keywords_count: int = Field(description="Number of keywords to process")


class RAGPipelineStatus(BaseModel):
    """Status model for checking pipeline progress."""

    task_id: str
    status: str
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RAGPipelineResult(BaseModel):
    """Result model for completed pipeline."""

    task_id: str
    status: str
    pipeline_config: Dict[str, Any]
    conversations: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    output_files: List[str]


async def run_rag_pipeline(task_id: str, request: RAGPipelineRequest) -> None:
    """Run the RAG pipeline asynchronously."""

    try:
        # Update task status
        running_tasks[task_id]["status"] = "running"
        running_tasks[task_id]["progress"]["current_step"] = "initializing"

        logfire.info(f"Starting RAG Pipeline for task {task_id}")

        # Validate environment variables based on backend
        if request.backend == "ollama" and not os.getenv("OLLAMA_API_KEY"):
            raise ValueError("OLLAMA_API_KEY environment variable is required for Ollama backend")

        if request.backend == "llama_cpp" and not os.getenv("HUGGINGFACE_API_KEY"):
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required for Llama CPP backend")

        if request.backend == "openAI" and not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI backend")

        # Create pipeline instance
        model_handler = WboeRAGPipeline(
            backend=request.backend,
            openai_model=request.openai_model,
            hf_model=request.hf_model,
            hf_model_fn=request.hf_model_fn,
            ollama_model=request.ollama_model,
            collection_name=request.collection_name,
            vector_store_filepath_name=request.vector_store_filepath_name,
            jwt_token=os.getenv("OLLAMA_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            hf_token=os.getenv("HUGGINGFACE_API_KEY"),
            user_input=request.user_input,
            keywords_to_process=request.keywords_to_process,
            max_context_length=request.max_context_length,
            model_memory_usage=request.model_memory_usage,
            output_dir=request.output_dir,
            gpu_memory_threshold=request.gpu_memory_threshold,
            enable_memory_monitoring=request.enable_memory_monitoring,
            aggressive_cleanup=request.aggressive_cleanup,
            retry_on_oom=request.retry_on_oom
        )

        # Update progress
        running_tasks[task_id]["progress"]["current_step"] = "memory_handling"

        # Memory handling
        model_handler.model_memory_handling()

        if request.enable_memory_monitoring:
            logfire.info("Initial Memory Status:")
            model_handler.print_gpu_memory_status()
            memory_info = model_handler.get_gpu_memory_info()
            running_tasks[task_id]["progress"]["memory_info"] = memory_info

        # Load documents
        running_tasks[task_id]["progress"]["current_step"] = "loading_documents"
        logfire.info("Loading documents from the vector store...")

        model_handler.get_documents()
        logfire.info("Documents loaded successfully.")

        # Generate responses
        running_tasks[task_id]["progress"]["current_step"] = "generating_responses"
        logfire.info("Generating LLM responses...")

        model_handler.generate()
        logfire.info("RAG pipeline generation completed successfully.")

        # Save chat history
        running_tasks[task_id]["progress"]["current_step"] = "saving_results"
        model_handler.save_chat_history()

        # Cleanup
        running_tasks[task_id]["progress"]["current_step"] = "cleanup"
        model_handler.unloading_vector_store_and_clear_up_memory()
        model_handler.unloading_models_and_clear_up_memory()

        # Prepare results
        result = {
            "pipeline_config": {
                "backend": request.backend,
                "ollama_model": request.ollama_model,
                "hf_model": request.hf_model,
                "max_context_length": request.max_context_length,
                "keywords_processed": len(request.keywords_to_process) if request.keywords_to_process else "all"
            },
            "conversations": model_handler.conversations,
            "statistics": {
                "total_documents_processed": len(model_handler.conversations),
                "total_user_inputs": len(request.user_input),
                "total_keywords": len(request.keywords_to_process) if request.keywords_to_process else "all"
            },
            "output_directory": request.output_dir
        }

        # Update final status
        running_tasks[task_id]["status"] = "completed"
        running_tasks[task_id]["result"] = result
        running_tasks[task_id]["progress"]["current_step"] = "completed"

        logfire.info(f"RAG Pipeline completed successfully for task {task_id}")

    except Exception as e:
        logfire.error(f"RAG Pipeline failed for task {task_id}: {str(e)}")
        running_tasks[task_id]["status"] = "failed"
        running_tasks[task_id]["error"] = str(e)
        running_tasks[task_id]["progress"]["current_step"] = "failed"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logfire.info("WBOE RAG API starting up...")
    yield
    # Shutdown
    logfire.info("WBOE RAG API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="WBOE RAG Pipeline API",
    description="Austrian Dialect Word Embeddings Processing Pipeline",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "WBOE RAG Pipeline API",
        "version": "1.0.0",
        "description": "Austrian Dialect Word Embeddings Processing Pipeline",
        "endpoints": {
            "process": "/rag/process",
            "status": "/rag/status/{task_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "running_tasks": len(running_tasks)
    }


@app.post("/rag/process", response_model=RAGPipelineResponse)
async def process_rag_pipeline(
    request: RAGPipelineRequest,
    background_tasks: BackgroundTasks
) -> RAGPipelineResponse:
    """
    Start RAG pipeline processing with the provided configuration.

    This endpoint initiates the RAG pipeline processing in the background
    and returns a task ID for tracking progress.
    """

    # Validate backend
    if request.backend not in ["ollama", "llama_cpp", "openAI"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported backend: {request.backend}. "
                   f"Supported backends are: 'ollama', 'llama_cpp', 'openAI'"
        )

    # Check required environment variables
    required_env_vars = {
        "ollama": "OLLAMA_API_KEY",
        "llama_cpp": "HUGGINGFACE_API_KEY",
        "openAI": "OPENAI_API_KEY"
    }

    required_env = required_env_vars.get(request.backend)
    if required_env and not os.getenv(required_env):
        raise HTTPException(
            status_code=400,
            detail=f"{required_env} environment variable is required for {request.backend} backend"
        )

    # Generate unique task ID
    task_id = f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(running_tasks)}"

    # Initialize task tracking
    running_tasks[task_id] = {
        "status": "started",
        "created_at": datetime.now(),
        "request": request,
        "progress": {
            "current_step": "queued",
            "steps_completed": 0,
            "total_steps": 6
        },
        "result": None,
        "error": None
    }

    # Start background task
    background_tasks.add_task(run_rag_pipeline, task_id, request)

    return RAGPipelineResponse(
        task_id=task_id,
        status="started",
        message="RAG pipeline processing started successfully",
        created_at=running_tasks[task_id]["created_at"],
        backend=request.backend,
        keywords_count=len(request.keywords_to_process) if request.keywords_to_process else 0
    )


@app.get("/rag/status/{task_id}", response_model=RAGPipelineStatus)
async def get_pipeline_status(task_id: str) -> RAGPipelineStatus:
    """
    Get the status and progress of a RAG pipeline task.
    """

    if task_id not in running_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    task = running_tasks[task_id]

    return RAGPipelineStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=task.get("result"),
        error=task.get("error")
    )


@app.get("/rag/result/{task_id}", response_model=RAGPipelineResult)
async def get_pipeline_result(task_id: str) -> RAGPipelineResult:
    """
    Get the complete result of a completed RAG pipeline task.
    """

    if task_id not in running_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    task = running_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task {task_id} is not completed. Current status: {task['status']}"
        )

    result = task["result"]

    return RAGPipelineResult(
        task_id=task_id,
        status=task["status"],
        pipeline_config=result["pipeline_config"],
        conversations=result["conversations"],
        statistics=result["statistics"],
        output_files=[result["output_directory"]]
    )


@app.get("/rag/tasks")
async def list_tasks():
    """
    List all tasks with their current status.
    """

    tasks_summary = []
    for task_id, task in running_tasks.items():
        tasks_summary.append({
            "task_id": task_id,
            "status": task["status"],
            "created_at": task["created_at"].isoformat(),
            "backend": task["request"].backend,
            "current_step": task["progress"].get("current_step", "unknown")
        })

    return {
        "total_tasks": len(running_tasks),
        "tasks": tasks_summary
    }


@app.delete("/rag/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task from the tracking system.
    """

    if task_id not in running_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    del running_tasks[task_id]

    return {
        "message": f"Task {task_id} deleted successfully"
    }


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
