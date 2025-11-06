# CineMind

An intelligent movie recommendation system powered by semantic search and large language models. CineMind uses vector embeddings and retrieval-augmented generation (RAG) to provide personalized, context-aware movie recommendations.

## Overview

CineMind combines the power of:
- **FAISS** for efficient vector similarity search
- **OpenAI Embeddings** for semantic understanding of movie content
- **GPT-4** for intelligent, conversational recommendations
- **LangChain** for orchestrating the RAG pipeline

## Features

- Semantic movie search based on plot descriptions and themes
- Context-aware recommendations using RAG architecture
- Fast vector similarity search across large movie databases
- Natural language query interface

## Project Structure

```
CineMind/
├── agent/                    # RAG inference agents
│   └── retrieval_inference.py   # Main recommendation engine
├── backend/                  # Data processing and embeddings
│   ├── embeddings_setup.py      # Initial embedding pipeline
│   └── create_embeddings_faiss.py  # FAISS index generation
├── data/                     # Data directory (excluded from git)
│   ├── movies_metadata.csv      # Raw movie data
│   ├── ratings_small.csv        # User ratings
│   ├── cleaned_movies.csv       # Processed dataset
│   ├── embeddings_corpus.jsonl  # Prepared corpus
│   └── faiss_index/             # Vector database
├── frontend/                 # (In development)
├── utils/                    # Utility functions
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (not in git)
```

## Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv
- OpenAI API key

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CineMind
```

### 2. Create Environment

Using Conda (recommended):
```bash
conda create -n cinemind python=3.10
conda activate cinemind
```

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

**Note:** Never commit your `.env` file to version control.

### 5. Prepare Dataset

Place your movie dataset files in the `data/` directory:
- `movies_metadata.csv`
- `ratings_small.csv`

Run the dataset preparation script:

```bash
python data/dataset_setup.py
```

### 6. Build Embeddings and Vector Store

Generate embeddings and create the FAISS index:

```bash
python backend/create_embeddings_faiss.py
```

This process may take several minutes depending on dataset size and will use OpenAI API credits.

## Usage

### Running the Recommendation System

```python
from agent.retrieval_inference import cine_recommend

# Get movie recommendations
cine_recommend("I loved Interstellar and Arrival — recommend similar thought-provoking sci-fi")
```

Or run the example queries:

```bash
python agent/retrieval_inference.py
```

### Example Queries

- "Movies like Inception with mind-bending plots"
- "Suggest light-hearted romantic comedies from the 2000s"
- "Films with artificial intelligence themes but emotionally deep"
- "Dark psychological thrillers similar to Shutter Island"

## Architecture

### Data Flow

```
User Query
    ↓
[Embedding Model] → Convert query to vector
    ↓
[FAISS Index] → Find top-k similar movies
    ↓
[Context Formatter] → Prepare retrieved movies
    ↓
[GPT-4 + Prompt] → Generate personalized recommendation
    ↓
Response + Source Movies
```

### Key Components

**1. Vector Embeddings**
- Uses OpenAI's `text-embedding-3-large` model
- Captures semantic meaning of movie plots and metadata
- 1536-dimensional vectors for each movie

**2. FAISS Vector Store**
- Facebook AI Similarity Search library
- Enables sub-second similarity search across thousands of movies
- Supports efficient k-NN (k-nearest neighbors) queries

**3. RAG Pipeline (LCEL)**
- Built with LangChain Expression Language
- Retrieves relevant movies based on semantic similarity
- Augments LLM context with retrieved information
- Generates contextual, factually grounded recommendations

## Technologies Used

- **Python 3.10+**
- **LangChain 0.3+** - LLM application framework
- **OpenAI API** - Embeddings and GPT-4
- **FAISS** - Vector similarity search
- **Pandas** - Data manipulation
- **python-dotenv** - Environment management

## Configuration

### Model Settings

Edit these in the respective files:

**Embeddings Model** (`backend/create_embeddings_faiss.py`):
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

**LLM Model** (`agent/retrieval_inference.py`):
```python
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)
```

**Retrieval Settings** (`agent/retrieval_inference.py`):
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

## Cost Considerations

- **Embeddings**: ~$0.13 per 1M tokens (text-embedding-3-large)
- **GPT-4 Turbo**: ~$10 per 1M input tokens, ~$30 per 1M output tokens
- One-time embedding cost for dataset
- Per-query cost for recommendations

## Troubleshooting

### Module Not Found Errors

Ensure you've activated the correct environment and installed all dependencies:
```bash
conda activate cinemind
pip install -r requirements.txt
```

### API Key Issues

Verify your `.env` file exists and contains a valid OpenAI API key:
```bash
cat .env  # Should show OPENAI_API_KEY=sk-...
```

### FAISS Index Not Found

Run the embedding generation script:
```bash
python backend/create_embeddings_faiss.py
```

## Development Status

This project is currently in active development. Planned features include:

- Web interface for interactive recommendations
- User preference learning
- Multi-modal search (posters, trailers)
- Collaborative filtering integration
- Deployment pipeline

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MovieLens dataset for movie metadata
- OpenAI for embeddings and language models
- Facebook Research for FAISS
- LangChain for the RAG framework

---

**Note:** This is an educational/research project. Ensure you comply with OpenAI's usage policies and dataset licenses when using this system.
