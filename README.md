# Code Search and Analysis System

A powerful code search and analysis system that uses embeddings and vector databases to enable semantic code search across multiple programming languages. This system allows you to search through your codebase using natural language queries and find relevant code snippets efficiently.

## Features

- **Multi-language Support**: Works with Python, JavaScript, Java, Go, C, C++, Ruby, and TypeScript
- **Semantic Code Search**: Find code using natural language queries
- **Code Embeddings**: Generate embeddings for code snippets using CodeBERT
- **Vector Database Storage**: Store and retrieve code embeddings using ChromaDB
- **Two-Pass Ranking**: Advanced search algorithm combining embedding similarity and keyword matching
- **Code Description Generation**: Automatically generate descriptions for code snippets
- **REST API**: Easy-to-use FastAPI endpoints for all operations
- **Efficient Processing**: Multi-threaded processing for handling large codebases

## Architecture

The system consists of several key components:

1. **Embeddings Generator**: Uses CodeBERT to generate embeddings for code snippets
2. **Vector Database**: Stores and manages code embeddings using ChromaDB
3. **Code Parser**: Parses code files using tree-sitter for different programming languages
4. **Search Engine**: Implements a two-pass ranking system for accurate code search
5. **REST API**: Provides endpoints for code search and management

## Setup

### Prerequisites

- Python 3.8+
- Git
- Tree-sitter parsers for supported languages

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```
GROQ_API_KEY=your_groq_api_key
```

### Configuration

The system can be configured through the following files:
- `app.py`: API endpoints and server configuration
- `database.py`: Vector database settings
- `code_embeddings.py`: Embedding generation settings

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://127.0.0.1:8080`

### API Endpoints

1. **Create Code Embeddings**
```bash
POST /make-code-embeddings
{
    "base_dir": "path/to/codebase",
    "ignored_items": ["node_modules", ".git"],
    "output_file": "code_files.json"
}
```

2. **Search Code**
```bash
POST /search-code
{
    "query": "find code that handles user authentication"
}
```

3. **Delete Collections**
```bash
POST /delete-all-collections
```

## Project Structure

```
project/
├── app.py                 # FastAPI application and endpoints
├── database.py           # Vector database operations
├── code_embeddings.py    # Code embedding generation
├── file_filter.py        # File filtering utilities
├── logger.py            # Logging configuration
├── models.py            # Data models
├── chroma_db/          # Vector database storage
└── .env                # Environment variables
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CodeBERT for code embeddings
- ChromaDB for vector database
- Tree-sitter for code parsing
- FastAPI for the web framework
