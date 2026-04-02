# WBOE RAG Pipeline API Guide

## Quick Start

### 1. Environment Setup

Copy the example environment file and configure your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
nano .env
```

### 2. Start the API Server

```bash
# Using the startup script (recommended)
./start_api.sh

# Or manually
cd generate_rag_query
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Process RAG Pipeline

**Endpoint**: `POST /rag/process`

**Example Request** (Ollama backend):
```bash
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "ollama",
    "ollama_model": "llama3.2:3b",
    "keywords_to_process": ["43218__Gefrette_Simplex"],
    "max_context_length": 32000,
    "enable_memory_monitoring": true
  }'
```

**Example Request** (OpenAI backend):
```bash
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "openAI",
    "openai_model": "gpt-4o",
    "keywords_to_process": ["43218__Gefrette_Simplex"],
    "output_dir": "output_openai"
  }'
```

**Example Request** (Llama CPP backend):
```bash
curl -X POST "http://localhost:8000/rag/process" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "llama_cpp",
    "hf_model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "hf_model_fn": "Llama-3.2-3B-Instruct-Q4_0.gguf",
    "keywords_to_process": [],
    "aggressive_cleanup": true,
    "gpu_memory_threshold": 0.8
  }'
```

**Response**:
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

### 3. Check Processing Status

**Endpoint**: `GET /rag/status/{task_id}`

```bash
curl http://localhost:8000/rag/status/rag_20250930_143022_0
```

**Response**:
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

### 4. Get Results

**Endpoint**: `GET /rag/result/{task_id}`

```bash
curl http://localhost:8000/rag/result/rag_20250930_143022_0
```

### 5. List All Tasks

**Endpoint**: `GET /rag/tasks`

```bash
curl http://localhost:8000/rag/tasks
```

### 6. Delete Task

**Endpoint**: `DELETE /rag/tasks/{task_id}`

```bash
curl -X DELETE http://localhost:8000/rag/tasks/rag_20250930_143022_0
```

## Configuration Parameters

### Backend Selection

| Backend | Description | Required Environment Variable |
|---------|-------------|------------------------------|
| `ollama` | Remote Ollama instance | `OLLAMA_API_KEY` |
| `llama_cpp` | Local GGUF models | `HUGGINGFACE_API_KEY` |
| `openAI` | OpenAI API | `OPENAI_API_KEY` |

### Model Configuration

```json
{
  "ollama_model": "llama3.2:3b",           // For Ollama backend
  "hf_model": "bartowski/Llama-3.2-3B-Instruct-GGUF",  // For Llama CPP
  "hf_model_fn": "Llama-3.2-3B-Instruct-Q4_0.gguf",   // GGUF filename
  "openai_model": "gpt-4o"                 // For OpenAI backend
}
```

### Processing Configuration

```json
{
  "keywords_to_process": [                 // Specific keywords (empty = all)
    "43218__Gefrette_Simplex",
    "44358__geilig_Simplex"
  ],
  "user_input": [                          // Prompt files to use
    "prompt1.txt",
    "prompt2.txt", 
    "prompt3.txt",
    "prompt4.txt"
  ]
}
```

### Performance Configuration

```json
{
  "max_context_length": 128000,           // Maximum tokens
  "model_memory_usage": 4.0,              // Model memory in GB
  "gpu_memory_threshold": 0.9,            // Max GPU usage (0.0-1.0)
  "enable_memory_monitoring": true,       // Memory monitoring
  "aggressive_cleanup": true,             // Force cleanup
  "retry_on_oom": true                    // Retry on OOM errors
}
```

## Python Client Example

```python
import requests
import time
import json

# Configuration
API_BASE = "http://localhost:8000"

# Start processing
response = requests.post(f"{API_BASE}/rag/process", json={
    "backend": "ollama",
    "ollama_model": "llama3.2:3b",
    "keywords_to_process": ["43218__Gefrette_Simplex"],
    "enable_memory_monitoring": True,
    "output_dir": "output_api"
})

task_data = response.json()
task_id = task_data["task_id"]
print(f"Started task: {task_id}")

# Poll for completion
while True:
    status_response = requests.get(f"{API_BASE}/rag/status/{task_id}")
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']}")
    print(f"Step: {status_data['progress']['current_step']}")
    
    if status_data["status"] == "completed":
        # Get results
        result_response = requests.get(f"{API_BASE}/rag/result/{task_id}")
        results = result_response.json()
        print("Processing completed!")
        print(f"Processed {results['statistics']['total_documents_processed']} documents")
        break
    elif status_data["status"] == "failed":
        print(f"Processing failed: {status_data.get('error', 'Unknown error')}")
        break
    
    time.sleep(5)  # Wait 5 seconds before next check
```

## Async Python Client Example

```python
import asyncio
import aiohttp
import json

async def process_rag_async():
    async with aiohttp.ClientSession() as session:
        # Start processing
        async with session.post("http://localhost:8000/rag/process", json={
            "backend": "openAI",
            "openai_model": "gpt-4o",
            "keywords_to_process": ["43218__Gefrette_Simplex"],
            "max_context_length": 64000
        }) as response:
            task_data = await response.json()
            task_id = task_data["task_id"]
        
        # Monitor progress
        while True:
            async with session.get(f"http://localhost:8000/rag/status/{task_id}") as response:
                status_data = await response.json()
                
                print(f"Status: {status_data['status']}")
                
                if status_data["status"] == "completed":
                    # Get results
                    async with session.get(f"http://localhost:8000/rag/result/{task_id}") as response:
                        results = await response.json()
                        return results
                elif status_data["status"] == "failed":
                    raise Exception(status_data.get("error", "Processing failed"))
                
                await asyncio.sleep(5)

# Run async processing
results = asyncio.run(process_rag_async())
print(json.dumps(results, indent=2))
```

## Error Handling

### Common HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)  
- **404**: Task not found
- **422**: Validation Error
- **500**: Internal Server Error

### Error Response Format

```json
{
  "detail": "Unsupported backend: invalid_backend. Supported backends are: 'ollama', 'llama_cpp', 'openAI'"
}
```

## Environment Variables

Create a `.env` file with your configuration:

```bash
# Required
LOGFIRE_TOKEN=your_logfire_token

# Backend API Keys (at least one required)
OLLAMA_API_KEY=your_jwt_token
HUGGINGFACE_API_KEY=your_hf_token  
OPENAI_API_KEY=your_openai_key

# Optional Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1
RELOAD=true
LOG_LEVEL=info
```

## Monitoring and Logging

The API integrates with Logfire for comprehensive logging and monitoring. All processing steps, errors, and performance metrics are logged for debugging and analysis.

### Log Categories

- **Pipeline Execution**: Step-by-step processing logs
- **Memory Management**: GPU memory usage and optimization
- **Model Operations**: Model loading, inference, and cleanup
- **Error Handling**: Detailed error context and recovery attempts
- **Performance Metrics**: Processing times and resource usage

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "generate_rag_query.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using systemd

```ini
[Unit]
Description=WBOE RAG API
After=network.target

[Service]
Type=exec
User=wboe
WorkingDirectory=/path/to/vectorstore_word_embeddings
Environment=PATH=/path/to/vectorstore_word_embeddings/.venv/bin
ExecStart=/path/to/vectorstore_word_embeddings/.venv/bin/uvicorn generate_rag_query.api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## Security Considerations

- Keep API keys secure and never commit them to version control
- Use environment variables or secure key management systems
- Consider implementing API authentication for production use
- Monitor API usage and implement rate limiting if needed
- Regularly update dependencies for security patches

## Performance Tips

- Use appropriate GPU memory thresholds based on your hardware
- Enable aggressive cleanup for memory-constrained environments
- Process documents in smaller batches for better resource management
- Monitor memory usage during processing to optimize settings
- Use quantized models (GGUF) for better memory efficiency
