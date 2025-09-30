# WBOE Vectorstore Word Embeddings

A sophisticated Retrieval-Augmented Generation (RAG) pipeline for processing Austrian dialect dictionary entries using multiple language model backends with advanced memory management and vector database integration.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- Access tokens for chosen backends (Ollama, HuggingFace)

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
OLLAMA_API_KEY=your_jwt_token_here
HUGGINGFACE_API_KEY=your_hf_token_here
LOGFIRE_TOKEN=your_logfire_token_here
OPENAI_API_KEY=your_openai_key_here
```

### Basic Usage

```bash
# Create vector store (first time only)
uv run generate_rag_query/create_vectorstore.py

# Run RAG pipeline
uv run generate_rag_query/generate_rag_query.py
```

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
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

### Key Components

- **Vector Database**: Chroma-based storage for document embeddings
- **Multi-Backend LLM Support**: Ollama, HuggingFace, and Llama CPP
- **Memory Management**: Intelligent GPU memory handling with dynamic optimization
- **Batch Processing**: Efficient processing of large document collections
- **Structured Output**: JSON-formatted results with conversation history

## ✨ Features

### 🎯 Core Functionality
- **Multi-Backend Support**: Choose between Ollama, HuggingFace Pipeline, or Llama CPP
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

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
WboeRAGPipeline
├── WboeBaseRAG (Base configuration and validation)
├── WboeLoadVectorstore (Vector database operations)
└── WboeLoadModels (Multi-backend LLM operations)
```

For detailed architecture documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

### System Flow

1. **Initialization**: Validate configuration and setup backends
2. **Memory Management**: Analyze GPU memory and calculate optimal settings
3. **Vector Loading**: Load Chroma database with document embeddings
4. **Document Processing**: Iterate through filtered documents
5. **LLM Processing**: Generate responses using selected backend
6. **Output Generation**: Save results and conversation history

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

## 📖 Usage Guide

### Basic Workflow

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
    backend="ollama",  # Choose: ollama, hf_pipeline, llama_cpp
    ollama_model="deepseek-r1:70b",
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

### Advanced Usage

#### Custom Prompt Engineering

Create custom prompt files:

```bash
# prompt1.txt - Base instructions
You are an expert linguist analyzing Austrian dialect terms...

# prompt2.txt - Specific analysis tasks
Analyze the semantic field and etymology of this term...

# prompt3.txt - Output formatting
Format your response as structured JSON with...

# prompt4.txt - Contextual questions
Based on the provided context, explain...
```

#### Batch Processing with Filtering

```python
# Process specific keywords only
keywords_to_process = [
    "43218__Gefrette_Simplex",
    "44358__geilig_Simplex",
    "44224__Köder_Simplex"
]

# Process all documents (remove or empty list)
keywords_to_process = []
```

#### Memory-Constrained Environments

```python
# Configuration for limited GPU memory
model_handler = WboeRAGPipeline(
    backend="llama_cpp",  # Most memory efficient
    hf_model="bartowski/Llama-3.2-3B-Instruct-GGUF",  # Smaller model
    model_memory_usage=2.0,  # Adjust based on available memory
    gpu_memory_threshold=0.8,  # Conservative memory usage
    aggressive_cleanup=True,  # Force cleanup between documents
    max_context_length=32000  # Reduced context window
)
```

## ⚙️ Configuration

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | `"ollama"` | LLM backend (`ollama`, `llama_cpp`) |
| `max_context_length` | int | `128000` | Maximum context length in tokens |
| `model_memory_usage` | float | `4.0` | Model memory usage in GB |
| `gpu_memory_threshold` | float | `0.9` | Maximum GPU memory utilization |
| `output_dir` | str | `"output"` | Output directory for results |
| `enable_memory_monitoring` | bool | `True` | Enable memory usage monitoring |
| `aggressive_cleanup` | bool | `True` | Force memory cleanup between documents |
| `retry_on_oom` | bool | `True` | Retry on out-of-memory errors |

### Backend-Specific Configuration

#### Ollama
```python
ollama_model = "deepseek-r1:70b"
jwt_token = os.getenv("OLLAMA_API_KEY")
base_url = "https://open-webui.acdh-dev.oeaw.ac.at/ollama"
```

#### Llama CPP
```python
hf_model = "bartowski/Llama-3.2-3B-Instruct-GGUF"
hf_model_fn = "Llama-3.2-3B-Instruct-Q4_0.gguf"
n_gpu_layers = -1  # Use all GPU layers
n_ctx = 128000
```

## 🔗 Backends

### Ollama Backend

**Advantages:**
- Remote processing capability
- No local GPU memory requirements
- Centralized model management
- Built-in API authentication

**Use Case:** When you have access to a remote Ollama instance with powerful hardware.

**Setup:**
```python
backend = "ollama"
ollama_model = "deepseek-r1:70b"
jwt_token = "your_jwt_token"
```

### Llama CPP Backend

**Advantages:**
- Memory-efficient quantized models
- CPU/GPU hybrid execution
- GGUF format support
- Optimal for resource-constrained environments

**Use Case:** When working with limited GPU memory or need efficient quantized models.

**Setup:**
```python
backend = "llama_cpp"
hf_model = "bartowski/Llama-3.2-3B-Instruct-GGUF"
hf_model_fn = "Llama-3.2-3B-Instruct-Q4_0.gguf"
```

## 🧠 Memory Management

The system includes sophisticated memory management to handle large models efficiently:

### Automatic Memory Detection

```python
def model_memory_handling(self) -> None:
    """Automatically detects and configures memory settings."""
    self.total_available_gpu_memory = self.check_free_gpu_memory()
    self.model_memory_usage_1k_token = (40 / 128000) * 1000
    # Calculates optimal settings based on available memory
```

### Dynamic Context Adjustment

```python
def calc_max_context_length(self) -> int:
    """Dynamically adjusts context length based on available memory."""
    avail_gpu_memory = self.total_available_gpu_memory - self.model_memory_usage
    token_per_gb = self.max_context_length / usage_for_max_length
    return int(token_per_gb * avail_gpu_memory)
```

### Memory Optimization Features

- **Automatic Cleanup**: Models unloaded after each document
- **GPU Cache Management**: `torch.cuda.empty_cache()` called strategically  
- **Context Truncation**: Automatic text truncation when exceeding limits
- **OOM Recovery**: Retry mechanisms for out-of-memory errors
- **Memory Monitoring**: Real-time memory usage tracking

## 📊 Output Format

### Individual Response Files

Each document/prompt combination generates a JSON file:

```json
{
    "model": "deepseek-r1:70b",
    "keyword": "43218__Gefrette_Simplex",
    "prompt": "prompt1",
    "response": "Generated analysis...",
    "timestamp": "2025-09-29T10:30:00Z",
    "elapsed_time": 15.42,
    "context_length": 1247,
    "memory_usage": {
        "gpu_memory_used": "8.2GB",
        "gpu_memory_total": "24GB"
    }
}
```

### Conversation History

Complete interaction log saved as `conversation_history.json`:

```json
{
    "pipeline_config": {
        "backend": "ollama",
        "model": "deepseek-r1:70b",
        "max_context_length": 128000
    },
    "conversations": [
        {
            "keyword": "43218__Gefrette_Simplex",
            "messages": [
                {
                    "role": "system",
                    "content": "System prompt..."
                },
                {
                    "role": "user", 
                    "content": "Context and prompt..."
                },
                {
                    "role": "assistant",
                    "content": "Generated response..."
                }
            ]
        }
    ],
    "statistics": {
        "total_documents_processed": 25,
        "total_prompts_processed": 100,
        "average_response_time": 12.8,
        "total_tokens_processed": 145000
    }
}
```

### Directory Structure

```
output/
├── conversation_history.json
├── 43218__Gefrette_Simplex_prompt1_deepseek-r1-70b.json
├── 43218__Gefrette_Simplex_prompt2_deepseek-r1-70b.json
├── 44358__geilig_Simplex_prompt1_deepseek-r1-70b.json
└── ...
```

## 🔧 Troubleshooting

### Common Issues

#### Out of Memory Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `model_memory_usage` parameter
2. Lower `gpu_memory_threshold` 
3. Use smaller model or Llama CPP backend
4. Enable `aggressive_cleanup`
5. Reduce `max_context_length`

```python
# Memory-conservative configuration
model_memory_usage = 2.0
gpu_memory_threshold = 0.7
aggressive_cleanup = True
max_context_length = 32000
```

#### Model Loading Failures

**Problem:** `ValueError: Hugging Face model must be specified`

**Solutions:**
1. Check model name spelling
2. Verify HuggingFace token permissions
3. Ensure model exists and is accessible
4. Check internet connection for downloads

#### Vector Store Issues

**Problem:** `FileNotFoundError: Vector store path does not exist`

**Solutions:**
1. Run `create_vectorstore.py` first
2. Check `vector_store_filepath_name` configuration
3. Verify `output_dir` permissions
4. Ensure corpus files exist in `llm_corpus/`

#### Authentication Errors

**Problem:** `ValueError: JWT token for Ollama API must be set`

**Solutions:**
1. Set environment variables in `.env` file
2. Verify token validity and permissions
3. Check network connectivity to Ollama server
4. Ensure correct token format

### Performance Optimization

#### Speed Optimization
- Use quantized models (GGUF format) with Llama CPP
- Enable aggressive cleanup for memory-constrained systems
- Process in smaller batches
- Use SSD storage for vector database

#### Quality Optimization  
- Use larger models when memory allows
- Increase context length for complex documents
- Fine-tune prompt templates for specific use cases
- Enable memory monitoring for optimal settings

### Debug Mode

Enable detailed logging:

```python
import logfire
logfire.configure(level="DEBUG")

# Additional debug settings
enable_memory_monitoring = True
retry_on_oom = True
```

## 📚 Examples

### Example 1: Quick Analysis

Process a single document with all prompts:

```python
model_handler = WboeRAGPipeline(
    backend="llama_cpp",
    hf_model="bartowski/Llama-3.2-3B-Instruct-GGUF",
    hf_model_fn="Llama-3.2-3B-Instruct-Q4_0.gguf",
    keywords_to_process=["43218__Gefrette_Simplex"],
    max_context_length=32000,
    output_dir="output_quick"
)
model_handler.main()
```

### Example 2: Batch Processing

Process multiple documents with memory optimization:

```python
model_handler = WboeRAGPipeline(
    backend="ollama",
    ollama_model="deepseek-r1:70b", 
    keywords_to_process=[
        "43218__Gefrette_Simplex",
        "44358__geilig_Simplex", 
        "44224__Köder_Simplex"
    ],
    aggressive_cleanup=True,
    enable_memory_monitoring=True,
    output_dir="output_batch"
)
model_handler.main()
```

### Example 3: Custom Prompts

Create domain-specific analysis:

```python
# Create custom prompt1.txt
with open("prompt1.txt", "w") as f:
    f.write("""
    Analyze this Austrian dialect term focusing on:
    1. Etymology and historical development
    2. Geographic distribution within Austria
    3. Semantic relationships to Standard German
    4. Contemporary usage patterns
    """)

# Run with custom prompts
model_handler = WboeRAGPipeline(
    backend="llama_cpp",
    hf_model="bartowski/Llama-3.2-3B-Instruct-GGUF",
    hf_model_fn="Llama-3.2-3B-Instruct-Q4_0.gguf",
    user_input=["prompt1.txt"],  # Use only custom prompt
    output_dir="output_etymology"
)
model_handler.main()
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

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
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_pipeline.py

# Run with coverage
uv run pytest --cov=generate_rag_query
```

### Submitting Changes

1. Create a feature branch
2. Make your changes with tests
3. Update documentation
4. Submit a pull request with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WBOE Team**: For providing the Austrian dialect corpus
- **Chroma**: For the vector database infrastructure  
- **LangChain**: For the RAG framework components
- **HuggingFace**: For model hosting and transformers library
- **Ollama**: For the model serving infrastructure

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/linxOD/llm-wboe-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/linxOD/llm-wboe-rag/discussions)
- **Email**: [Project Maintainer](mailto:daniel.elsner@oeaw.ac.at)

## 🔄 Changelog

### v0.1.0 (Current)
- Initial release with multi-backend support
- Advanced memory management system
- Comprehensive error handling and recovery
- Structured output and conversation logging
- Vector database integration with Chroma

---

**Made with ❤️ for Austrian dialect research**
**README generated with Claude AI**
