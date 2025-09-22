# Creating a Vectorstore with ChromaDB for Retrieval Augmented Generation (RAG) with Open Source LLMs

# WBOE RAG Pipeline Flowchart

```mermaid
flowchart TD
    A[Start WBOE RAG Pipeline] --> B{Select Backend}
    B --> C[Ollama]
    B --> D[HuggingFace Pipeline]
    B --> E[Llama CPP]
    
    C --> F[Validate Ollama Model & JWT Token]
    D --> G[Validate HF Model & HF Token]
    E --> H[Validate HF Model, Model File & HF Token]
    
    F --> I[Initialize WboeRAGPipeline]
    G --> I
    H --> I
    
    I --> J[Model Memory Handling]
    J --> K[Check GPU Memory Status]
    K --> L[Calculate Available GPU Memory]
    L --> M[Calculate Max Context Length]
    
    M --> N[Load Vector Store Documents]
    N --> O[Initialize Chroma Vector Store]
    O --> P[Yield Documents from Vector Store]
    P --> Q{Documents Found?}
    
    Q -->|No| R[Raise Error: No Documents Found]
    Q -->|Yes| S[Process Keywords]
    
    S --> T{For Each Document}
    T --> U[Check if Keyword in Process List]
    U -->|No| V[Skip Document]
    U -->|Yes| W[Extract Document Context & Embeddings]
    
    W --> X{Select Processing Backend}
    X --> Y[Process with Ollama]
    X --> Z[Process with HuggingFace]
    X --> AA[Process with Llama CPP]
    
    Y --> BB[Generate RAG Query with Ollama API]
    Z --> CC[Load HF Model & Generate Query]
    AA --> DD[Load Llama CPP Model & Generate Query]
    
    BB --> EE[Store Conversation Messages]
    CC --> EE
    DD --> EE
    
    EE --> FF[Sleep 5 seconds]
    FF --> GG{More Documents?}
    
    GG -->|Yes| T
    GG -->|No| HH[Save Chat History to JSON]
    
    HH --> II[Unload Models & Clear Memory]
    II --> JJ[End Pipeline]
    
    V --> GG
    R --> KK[End with Error]
    
    subgraph "Vector Store Creation (Separate Process)"
        LL[Start Vector Store Creation] --> MM[Load Text Documents from llm_corpus]
        MM --> NN[Create Document Embeddings]
        NN --> OO[Store in Chroma Vector Database]
        OO --> PP[Save Vector Store to Disk]
    end
    
    subgraph "Memory Management"
        QQ[Monitor GPU Memory] --> RR[Calculate Memory Usage per 1K Tokens]
        RR --> SS[Adjust Context Length Based on Available Memory]
        SS --> TT[Enable Aggressive Cleanup if Needed]
        TT --> UU[Retry on Out of Memory Errors]
    end
    
    subgraph "Input Files"
        VV[prompt1.txt] --> WW[User Prompts]
        XX[prompt2.txt] --> WW
        YY[prompt3.txt] --> WW
        ZZ[prompt4.txt] --> WW
        WW --> AAA[Fed to LLM for Context]
    end
    
    subgraph "Output"
        BBB[Conversation History JSON] --> CCC[Generated RAG Responses]
        CCC --> DDD[Word Embedding Analysis Results]
    end
    
    style A fill:#e1f5fe
    style JJ fill:#c8e6c9
    style KK fill:#ffcdd2
    style I fill:#fff3e0
    style N fill:#f3e5f5
    style EE fill:#e8f5e8
```

## Pipeline Overview

### Main Components:

1. **Initialization Phase**
   - Backend selection (Ollama, HuggingFace Pipeline, or Llama CPP)
   - Model and token validation
   - Memory management setup

2. **Vector Store Integration**
   - Loads documents from Chroma vector database
   - Extracts relevant embeddings for specified keywords
   - Filters documents based on keyword processing list

3. **Processing Phase**
   - For each document/keyword:
     - Extracts context and embeddings
     - Generates RAG queries using selected backend
     - Stores conversation messages
     - Implements memory-aware processing

4. **Memory Management**
   - GPU memory monitoring
   - Dynamic context length calculation
   - Model loading/unloading optimization
   - Out-of-memory error handling

5. **Output Generation**
   - Saves conversation history to JSON
   - Generates word embedding analysis results
   - Cleanup and resource management

### Key Features:

- **Multi-backend Support**: Supports Ollama, HuggingFace, and Llama CPP backends
- **Memory Optimization**: Intelligent GPU memory management with configurable thresholds
- **Scalable Processing**: Processes multiple documents with configurable keyword filtering
- **Error Handling**: Comprehensive error handling and retry mechanisms
- **Persistent Storage**: Saves results and conversation history for later analysis

### Input Data:
- **Text Corpus**: Austrian dialect dictionary entries in `llm_corpus/` directory
- **Prompts**: User-defined prompts in `prompt1.txt` through `prompt4.txt`
- **Keywords**: Configurable list of keywords to process

### Output Data:
- **Conversation History**: JSON file containing all LLM interactions
- **RAG Responses**: Generated responses for word embedding analysis
- **Memory Statistics**: GPU usage and performance metrics
