# WBOE RAG Pipeline with FastAPI

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for processing Austrian dialect dictionary entries using multiple language model backends with advanced memory management, vector database integration, and a FastAPI web service.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- Access tokens for chosen backends (Ollama, HuggingFace, OpenAI)

### Installation

```bash
# Clone the repository
git clone https://github.com/linxOD/llm-wboe-rag.git
cd vectorstore_word_embeddings

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env with your API tokens
nano .env
```

Required environment variables:
```bash
LOGFIRE_TOKEN=your_logfire_token_here

# At least one backend API key:
OLLAMA_API_KEY=your_jwt_token_here          # For Ollama backend
HUGGINGFACE_API_KEY=your_hf_token_here      # For Llama CPP backend  
OPENAI_API_KEY=your_openai_api_key_here     # For OpenAI backend
```

### Quick Usage

#### Option 1: FastAPI Web Service (Recommended)

```bash
# Start the API server
./start_api.sh

# In another terminal, test the API
python test_api.py

# Process documents via API
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama",
    "ollama_model": "llama3.2:3b",
    "keywords_to_process": ["43218__Gefrette_Simplex"]
  }'
```

#### Option 2: Direct Python Usage

```bash
# Create vector store (first time only)
uv run generate_rag_query/create_vectorstore.py

# Run RAG pipeline directly
uv run generate_rag_query/generate_rag_query.py
```

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Backends](#backends)
- [Memory Management](#memory-management)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

The WBOE (Wörterbuch der bairischen Mundarten in Österreich) RAG system processes Austrian dialect dictionary entries using advanced language models. It combines vector database search capabilities with multiple LLM backends to generate contextual analyses of dialect terms and their usage patterns.

**NEW**: Now includes a FastAPI web service for easy integration and remote processing!

### Key Components

- **FastAPI Web Service**: RESTful API for remote processing and integration
- **Vector Database**: Chroma-based storage for document embeddings
- **Multi-Backend LLM Support**: Ollama, HuggingFace, and OpenAI
- **Memory Management**: Intelligent GPU memory handling with dynamic optimization
- **Batch Processing**: Efficient processing of large document collections
- **Structured Output**: JSON-formatted results with conversation history

## ✨ Features

### 🌐 **NEW: FastAPI Web Service**
- **RESTful API**: Process documents remotely via HTTP endpoints
- **Async Processing**: Background task processing with status tracking
- **Interactive Documentation**: Auto-generated API docs at `/docs`
- **Health Monitoring**: Built-in health checks and system status
- **Error Handling**: Comprehensive error handling and validation
- **Task Management**: Track multiple concurrent processing tasks

### 🎯 Core Functionality
- **Multi-Backend Support**: Choose between Ollama, HuggingFace Pipeline, or OpenAI
- **Vector Database Integration**: Efficient document retrieval using Chroma embeddings
- **Batch Processing**: Process multiple documents with configurable filtering
- **Memory Optimization**: Dynamic GPU memory management with automatic cleanup
- **Conversation History**: Complete interaction logging for analysis and debugging

### 🛡️ Reliability & Performance
- **Error Recovery**: Automatic retry mechanisms for OOM and network errors
- **Context Management**: Intelligent text truncation to fit model limits
- **Memory Monitoring**: Real-time GPU memory usage tracking
- **Structured Logging**: Comprehensive logging with Logfire integration
- **Graceful Degradation**: Fallback handling for model loading failures

### 🔧 Flexibility
- **Configurable Prompts**: Four customizable prompt templates
- **Keyword Filtering**: Process specific subsets of documents
- **Output Customization**: Configurable output directories and formats
- **Backend Switching**: Easy switching between different LLM backends
- **Model Selection**: Support for various model sizes and quantizations

## 🌐 API Documentation

### Starting the API Server

```bash
# Quick start with environment checking
./start_api.sh

# Manual startup
cd generate_rag_query
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check and system status |
| `/docs` | GET | Interactive API documentation |
| `/rag/process` | POST | Start RAG pipeline processing |
| `/rag/status/{task_id}` | GET | Check processing status |
| `/rag/result/{task_id}` | GET | Get completed results |
| `/rag/tasks` | GET | List all tasks |
| `/rag/tasks/{task_id}` | DELETE | Delete a task |

### Example API Usage

#### Start Processing
```bash
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama",
    "ollama_model": "llama3.2:3b",
    "keywords_to_process": ["43218__Gefrette_Simplex"],
    "max_context_length": 32000,
    "enable_memory_monitoring": true,
    "output_dir": "output_api"
  }'
```

#### Check Status
```bash
curl http://localhost:8000/rag/status/rag_20250930_143022_0
```

#### Get Results
```bash
curl http://localhost:8000/rag/result/rag_20250930_143022_0
```

### Python Client Example

```python
import requests
import time

# Start processing
response = requests.post("http://localhost:8000/rag/process", json={
    "backend": "openAI",
    "openai_model": "gpt-4o",
    "keywords_to_process": ["43218__Gefrette_Simplex"]
})

task_data = response.json()
task_id = task_data["task_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/rag/status/{task_id}").json()
    print(f"Status: {status['status']}")
    
    if status["status"] == "completed":
        results = requests.get(f"http://localhost:8000/rag/result/{task_id}").json()
        print("Processing completed!")
        break
    elif status["status"] == "failed":
        print(f"Processing failed: {status.get('error')}")
        break
    
    time.sleep(5)
```

For complete API documentation, see [API_GUIDE.md](./API_GUIDE.md).

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
WboeRAGPipeline
├── WboeBaseRAG (Base configuration and validation)
├── WboeLoadVectorstore (Vector database operations)
└── WboeLoadModels (Multi-backend LLM operations)

FastAPI Service
├── Background Tasks (Async processing)
├── Task Management (Status tracking)  
├── Error Handling (Validation & recovery)
└── API Endpoints (RESTful interface)
```

For detailed architecture documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

### System Flow

1. **API Request**: Receive processing request via FastAPI endpoint
2. **Validation**: Validate configuration and check required API keys
3. **Background Task**: Start async processing with task ID
4. **Memory Management**: Analyze GPU memory and calculate optimal settings
5. **Vector Loading**: Load Chroma database with document embeddings
6. **Document Processing**: Iterate through filtered documents
7. **LLM Processing**: Generate responses using selected backend
8. **Status Updates**: Real-time progress tracking via API
9. **Output Generation**: Save results and provide download links

## 🔧 Installation & Setup

### System Requirements

- **Python**: 3.11 or higher
- **GPU Memory**: 4GB+ VRAM (recommended for local models)
- **Disk Space**: 10GB+ for models and vector database
- **RAM**: 16GB+ system RAM recommended

### Detailed Installation

1. **Clone Repository**
   ```bash
   git clone https://github.com/linxOD/llm-wboe-rag.git
   cd vectorstore_word_embeddings
   ```

2. **Python Environment Setup**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   uv sync

   # Using pip
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

3. **Environment Configuration**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit with your credentials
   nano .env
   ```

4. **Vector Database Setup**
   ```bash
   # Create vector store with your corpus
   uv run generate_rag_query/create_vectorstore.py
   ```

5. **Test Installation**
   ```bash
   # Start API server
   ./start_api.sh &
   
   # Run tests
   python test_api.py
   ```

## 📖 Usage Guide

### API-First Usage (Recommended)

#### 1. Start the API Server
```bash
./start_api.sh
```

#### 2. Submit Processing Request
```bash
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama",
    "ollama_model": "llama3.2:3b",
    "keywords_to_process": ["43218__Gefrette_Simplex"],
    "enable_memory_monitoring": true
  }'
```

#### 3. Monitor Progress
```bash
# Check status
curl http://localhost:8000/rag/status/{task_id}

# List all tasks  
curl http://localhost:8000/rag/tasks

# Access interactive docs
open http://localhost:8000/docs
```

### Direct Python Usage

#### 1. Prepare Your Data
Ensure your text corpus is in the `llm_corpus/` directory:
```
llm_corpus/
├── word1.txt
├── word2.txt
└── ...
```

#### 2. Configure Processing
Edit the configuration in `generate_rag_query.py`:
```python
model_handler = WboeRAGPipeline(
    backend="ollama",  # Choose: ollama, llama_cpp, openAI
    ollama_model="llama3.2:3b",
    keywords_to_process=["specific_keyword"],  # Filter documents
    max_context_length=128000,
    model_memory_usage=4.0,  # GB
    gpu_memory_threshold=0.9,  # Use 90% max GPU memory
    enable_memory_monitoring=True,
    aggressive_cleanup=True
)
```

#### 3. Run Processing
```bash
# Full pipeline
uv run generate_rag_query/generate_rag_query.py

# Small subset (for testing)
uv run generate_rag_query/generate_rag_query_small.py
```

### Advanced API Usage

#### Batch Processing Multiple Keywords
```json
{
  "backend": "ollama",
  "ollama_model": "llama3.2:3b",
  "keywords_to_process": [
    "43218__Gefrette_Simplex",
    "44358__geilig_Simplex", 
    "44224__Köder_Simplex"
  ],
  "aggressive_cleanup": true,
  "output_dir": "output_batch"
}
```

#### Memory-Constrained Processing
```json
{
  "backend": "llama_cpp",
  "hf_model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
  "hf_model_fn": "Llama-3.2-3B-Instruct-Q4_0.gguf",
  "model_memory_usage": 2.0,
  "gpu_memory_threshold": 0.7,
  "max_context_length": 32000,
  "aggressive_cleanup": true
}
```

#### OpenAI Backend Processing
```json
{
  "backend": "openAI",
  "openai_model": "gpt-4o",
  "keywords_to_process": ["43218__Gefrette_Simplex"],
  "max_context_length": 64000,
  "output_dir": "output_openai"
}
```

## ⚙️ Configuration

### API Configuration

The FastAPI service can be configured via environment variables:

```bash
# Server settings
HOST=0.0.0.0
PORT=8000
WORKERS=1
RELOAD=true
LOG_LEVEL=info

# Required tokens
LOGFIRE_TOKEN=your_token_here

# Backend API keys (at least one required)
OLLAMA_API_KEY=your_ollama_token
HUGGINGFACE_API_KEY=your_hf_token
OPENAI_API_KEY=your_openai_key
```

### Processing Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | `"ollama"` | LLM backend (`ollama`, `llama_cpp`, `openAI`) |
| `max_context_length` | int | `128000` | Maximum context length in tokens |
| `model_memory_usage` | float | `4.0` | Model memory usage in GB |
| `gpu_memory_threshold` | float | `0.9` | Maximum GPU memory utilization |
| `output_dir` | str | `"output"` | Output directory for results |
| `enable_memory_monitoring` | bool | `True` | Enable memory usage monitoring |
| `aggressive_cleanup` | bool | `True` | Force memory cleanup between documents |
| `retry_on_oom` | bool | `True` | Retry on out-of-memory errors |

### Backend-Specific Configuration

#### Ollama
```json
{
  "backend": "ollama",
  "ollama_model": "llama3.2:3b"
}
```

#### OpenAI
```json
{
  "backend": "openAI", 
  "openai_model": "gpt-4o"
}
```

#### Llama CPP
```json
{
  "backend": "llama_cpp",
  "hf_model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
  "hf_model_fn": "Llama-3.2-3B-Instruct-Q4_0.gguf"
}
```

## 🔗 Backends

### Ollama Backend

**Advantages:**
- Remote processing capability
- No local GPU memory requirements
- Centralized model management
- Built-in API authentication

**Use Case:** When you have access to a remote Ollama instance with powerful hardware.

**API Configuration:**
```json
{
  "backend": "ollama",
  "ollama_model": "llama3.2:3b"
}
```

### OpenAI Backend

**Advantages:**
- Highest quality responses
- No local hardware requirements
- Latest model capabilities
- Reliable and fast processing

**Use Case:** When you want the best quality results and have OpenAI API access.

**API Configuration:**
```json
{
  "backend": "openAI",
  "openai_model": "gpt-4o"
}
```

### Llama CPP Backend

**Advantages:**
- Memory-efficient quantized models
- CPU/GPU hybrid execution
- GGUF format support
- Optimal for resource-constrained environments

**Use Case:** When working with limited GPU memory or need efficient quantized models.

**API Configuration:**
```json
{
  "backend": "llama_cpp",
  "hf_model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
  "hf_model_fn": "Llama-3.2-3B-Instruct-Q4_0.gguf"
}
```

## 🧠 Memory Management

The system includes sophisticated memory management for both direct usage and API processing:

### Automatic Memory Detection
- Real-time GPU memory monitoring
- Dynamic context length adjustment
- Memory usage prediction and optimization
- Out-of-memory error recovery

### API Memory Monitoring
- Background memory status tracking
- Progress updates with memory information
- Automatic cleanup between tasks
- Memory threshold enforcement

### Memory Optimization Features
- **Automatic Cleanup**: Models unloaded after each document
- **GPU Cache Management**: Strategic `torch.cuda.empty_cache()` calls
- **Context Truncation**: Automatic text truncation when exceeding limits
- **OOM Recovery**: Retry mechanisms with reduced memory usage
- **Real-time Monitoring**: Memory usage tracking via API endpoints

## 📊 Output Format

### API Response Format

**Task Creation Response:**
```json
{
  "task_id": "rag_20250930_143022_0",
  "status": "started",
  "message": "RAG pipeline processing started successfully",
  "created_at": "2025-09-30T14:30:22.123456",
  "backend": "ollama",
  "keywords_count": 1
}
```

**Status Response:**
```json
{
  "task_id": "rag_20250930_143022_0", 
  "status": "running",
  "progress": {
    "current_step": "generating_responses",
    "steps_completed": 3,
    "total_steps": 6,
    "memory_info": {
      "gpu_memory_used": "8.2GB",
      "gpu_memory_total": "24GB"
    }
  }
}
```

**Results Response:**
```json
{
  "task_id": "rag_20250930_143022_0",
  "status": "completed", 
  "pipeline_config": {
    "backend": "ollama",
    "ollama_model": "llama3.2:3b"
  },
  "conversations": [...],
  "statistics": {
    "total_documents_processed": 1,
    "total_user_inputs": 4
  },
  "output_files": ["output_api"]
}
```

### Individual Response Files

Each document/prompt combination generates a JSON file:
```json
{
  "model": "llama3.2:3b",
  "keyword": "43218__Gefrette_Simplex",
  "prompt": "prompt1",
  "response": "Generated analysis...",
  "timestamp": "2025-09-30T14:30:22Z",
  "elapsed_time": 15.42
}
```

## 🔧 Troubleshooting

### API Issues

#### Cannot Connect to API Server
**Problem:** `ConnectionError: Cannot connect to API server`

**Solutions:**
1. Check if API server is running: `./start_api.sh`
2. Verify port is not in use: `lsof -i :8000`
3. Check firewall settings
4. Ensure correct host/port in client code

#### Environment Variable Errors
**Problem:** `JWT token must be set` or similar API key errors

**Solutions:**
1. Check `.env` file exists and has correct values
2. Verify environment variables are loaded: `env | grep API_KEY`
3. Restart API server after updating `.env`
4. Check API key format and permissions

### Processing Issues

#### Out of Memory Errors
**Problem:** `RuntimeError: CUDA out of memory`

**API Solution:**
```json
{
  "model_memory_usage": 2.0,
  "gpu_memory_threshold": 0.7,
  "aggressive_cleanup": true,
  "max_context_length": 32000
}
```

#### Model Loading Failures
**Problem:** Model fails to load or times out

**API Solution:**
```json
{
  "backend": "openAI",
  "openai_model": "gpt-4o"
}
```

### Testing and Debugging

```bash
# Test API connectivity
python test_api.py

# Check API server logs
./start_api.sh  # Check console output

# Test specific backend
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{"backend": "openAI", "openai_model": "gpt-4o", "keywords_to_process": ["test"]}'

# Monitor system resources
watch -n 1 "nvidia-smi && curl -s http://localhost:8000/health"
```

## 📚 Examples

### Example 1: Quick API Test

```bash
# Start server
./start_api.sh &

# Quick test with minimal config
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "openAI",
    "openai_model": "gpt-4o", 
    "keywords_to_process": ["43218__Gefrette_Simplex"],
    "user_input": ["prompt1.txt"]
  }'
```

### Example 2: Batch Processing via API

```python
import requests

# Configuration for batch processing
config = {
    "backend": "ollama",
    "ollama_model": "llama3.2:3b",
    "keywords_to_process": [
        "43218__Gefrette_Simplex",
        "44358__geilig_Simplex", 
        "44224__Köder_Simplex"
    ],
    "aggressive_cleanup": True,
    "enable_memory_monitoring": True
}

# Start processing
response = requests.post("http://localhost:8000/rag/process", json=config)
task_id = response.json()["task_id"]
print(f"Started batch processing: {task_id}")
```

### Example 3: Memory-Optimized Processing

```json
{
  "backend": "llama_cpp",
  "hf_model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
  "hf_model_fn": "Llama-3.2-3B-Instruct-Q4_0.gguf", 
  "keywords_to_process": [],
  "model_memory_usage": 2.0,
  "gpu_memory_threshold": 0.7,
  "max_context_length": 32000,
  "aggressive_cleanup": true,
  "retry_on_oom": true
}
```

## 🧪 Testing

### API Testing

```bash
# Basic API tests
python test_api.py

# Start API and run comprehensive tests
./start_api.sh &
sleep 5
python test_api.py

# Test specific backend
OPENAI_API_KEY=your_key python test_api.py
```

### Direct Pipeline Testing

```bash
# Run pipeline tests
uv run pytest

# Test specific functionality  
uv run pytest tests/test_pipeline.py

# Run with coverage
uv run pytest --cov=generate_rag_query
```

## 🤝 Contributing

We welcome contributions! The project now includes both API and direct pipeline functionality.

### Development Setup

1. Fork and clone the repository
2. Install development dependencies:
   ```bash
   uv sync --dev
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Code Quality

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings for public methods
- Write tests for new features
- Run linting before commits:
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```

### Testing

```bash
# Test API functionality
python test_api.py

# Test direct pipeline
uv run pytest

# Run specific test file
uv run pytest tests/test_pipeline.py

# Run with coverage
uv run pytest --cov=generate_rag_query
```

### Submitting Changes

1. Create a feature branch
2. Make your changes with tests
3. Update documentation (including API docs)
4. Test both API and direct usage
5. Submit a pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WBOE Team**: For providing the Austrian dialect corpus
- **Chroma**: For the vector database infrastructure  
- **LangChain**: For the RAG framework components
- **FastAPI**: For the excellent web framework
- **HuggingFace**: For model hosting and transformers library
- **Ollama**: For the model serving infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/linxOD/llm-wboe-rag/issues)
- **API Documentation**: http://localhost:8000/docs (when server is running)
- **Discussions**: [GitHub Discussions](https://github.com/linxOD/llm-wboe-rag/discussions)
- **Email**: [Project Maintainer](mailto:maintainer@example.com)

## 🔄 Changelog

### v1.0.0 - FastAPI Integration (Current)
- **NEW**: FastAPI web service with RESTful API
- **NEW**: Async background processing with task tracking
- **NEW**: Interactive API documentation
- **NEW**: Comprehensive API testing suite
- **NEW**: Environment validation and startup scripts
- Advanced memory management system
- Multi-backend support (Ollama, OpenAI, Llama CPP)
- Comprehensive error handling and recovery
- Structured output and conversation logging
- Vector database integration with Chroma

### v0.1.0 - Initial Release
- Basic pipeline with multi-backend support
- Memory management system
- Error handling and recovery
- Vector database integration

---

**Made with ❤️ for Austrian dialect research - Now with FastAPI! 🚀**
