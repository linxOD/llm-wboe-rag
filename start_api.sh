#!/usr/bin/env bash

# WBOE RAG Pipeline API Startup Script
# This script starts the FastAPI server for the RAG pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
RELOAD=${RELOAD:-true}
LOG_LEVEL=${LOG_LEVEL:-"info"}

echo -e "${GREEN}🚀 Starting WBOE RAG Pipeline API...${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Host: $HOST"
echo -e "  Port: $PORT"
echo -e "  Workers: $WORKERS"
echo -e "  Reload: $RELOAD"
echo -e "  Log Level: $LOG_LEVEL"
echo ""

# Check if environment variables are set
echo -e "${YELLOW}🔍 Checking environment variables...${NC}"

if [ -z "$LOGFIRE_TOKEN" ]; then
    echo -e "${RED}❌ LOGFIRE_TOKEN is not set${NC}"
    echo -e "   Please set it in your .env file or environment"
    exit 1
else
    echo -e "${GREEN}✅ LOGFIRE_TOKEN is set${NC}"
fi

# Check if at least one backend API key is available
BACKEND_AVAILABLE=false

if [ ! -z "$OLLAMA_API_KEY" ]; then
    echo -e "${GREEN}✅ OLLAMA_API_KEY is set - Ollama backend available${NC}"
    BACKEND_AVAILABLE=true
fi

if [ ! -z "$HUGGINGFACE_API_KEY" ]; then
    echo -e "${GREEN}✅ HUGGINGFACE_API_KEY is set - Llama CPP backend available${NC}"
    BACKEND_AVAILABLE=true
fi

if [ ! -z "$OPENAI_API_KEY" ]; then
    echo -e "${GREEN}✅ OPENAI_API_KEY is set - OpenAI backend available${NC}"
    BACKEND_AVAILABLE=true
fi

if [ "$BACKEND_AVAILABLE" = false ]; then
    echo -e "${RED}❌ No backend API keys are set${NC}"
    echo -e "   Please set at least one of:"
    echo -e "   - OLLAMA_API_KEY (for Ollama backend)"
    echo -e "   - HUGGINGFACE_API_KEY (for Llama CPP backend)"
    echo -e "   - OPENAI_API_KEY (for OpenAI backend)"
    exit 1
fi

# Check if vector store exists
VECTOR_STORE_PATH="output/chroma_langchain_db_wboe_embeddings"
if [ ! -d "$VECTOR_STORE_PATH" ]; then
    echo -e "${YELLOW}⚠️  Vector store not found at $VECTOR_STORE_PATH${NC}"
    echo -e "   You may need to create it first using:"
    echo -e "   ${GREEN}uv run generate_rag_query/create_vectorstore.py${NC}"
    echo ""
fi

# Check if prompt files exist
PROMPT_FILES=("prompt1.txt" "prompt2.txt" "prompt3.txt" "prompt4.txt")
MISSING_PROMPTS=()

for prompt_file in "${PROMPT_FILES[@]}"; do
    if [ ! -f "$prompt_file" ]; then
        MISSING_PROMPTS+=("$prompt_file")
    fi
done

if [ ${#MISSING_PROMPTS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some prompt files are missing:${NC}"
    for missing in "${MISSING_PROMPTS[@]}"; do
        echo -e "   - $missing"
    done
    echo -e "   The API will still work, but you'll need to provide custom prompts"
    echo ""
fi

# Install dependencies if needed
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ uv is not installed${NC}"
    echo -e "   Please install it first: https://docs.astral.sh/uv/"
    exit 1
fi

# Sync dependencies
echo -e "${GREEN}📦 Syncing dependencies...${NC}"
uv sync --quiet

echo -e "${GREEN}🎯 Starting API server...${NC}"
echo -e "${YELLOW}Access the API at: http://localhost:$PORT${NC}"
echo -e "${YELLOW}API Documentation: http://localhost:$PORT/docs${NC}"
echo ""

# Start the server
cd generate_rag_query

if [ "$RELOAD" = "true" ]; then
    uv run uvicorn api:app --host "$HOST" --port "$PORT" --reload --log-level "$LOG_LEVEL"
else
    uv run uvicorn api:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-level "$LOG_LEVEL"
fi
