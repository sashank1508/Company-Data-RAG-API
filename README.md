# 📊 Company Data RAG API

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logo=database&logoColor=white)](https://www.trychroma.com/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge&logo=python&logoColor=white)](https://www.nltk.org/)
[![Yahoo Finance](https://img.shields.io/badge/Yahoo%20Finance-6001D2?style=for-the-badge&logo=yahoo&logoColor=white)](https://finance.yahoo.com/)
[![Alpha Vantage](https://img.shields.io/badge/Alpha%20Vantage-0066CC?style=for-the-badge&logo=alpha&logoColor=white)](https://www.alphavantage.co/)

> **A comprehensive FastAPI-based Retrieval Augmented Generation (RAG) system for intelligent company data analysis, combining traditional NLP with cutting-edge AI technologies.**

## 🚀 Overview

The Company Data RAG API is a sophisticated web service designed to revolutionize how you manage, analyze, and extract insights from company-related data. By seamlessly integrating multiple data sources including URLs, PDFs, financial news, and historical stock data, this system provides intelligent, context-aware responses to complex business queries.

### ✨ Key Highlights

- **🧠 Advanced RAG Architecture**: Combines ChromaDB vector storage with OpenAI's GPT models for intelligent document retrieval and response generation
- **📈 Financial Intelligence**: Real-time stock data and news analysis using Yahoo Finance and Alpha Vantage APIs
- **🔍 Multi-Modal Processing**: Handles both web content and PDF documents with advanced text chunking
- **⚡ High Performance**: Asynchronous processing with rate limiting and concurrent request handling
- **🎯 Smart Analytics**: Dual keyword extraction using both NLTK and LLM-based approaches

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Scraping  │    │   PDF Processing │    │  Financial APIs │
│   (BeautifulSoup│ -> │   (LangChain)    │ -> │ (Yahoo/Alpha)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                       │                       │
          v                       v                       v
┌─────────────────────────────────────────────────────────────────┐
│                    ChromaDB Vector Store                        │
│              (OpenAI Embeddings + Metadata)                    │
└─────────────────────────────────────────────────────────────────┘
          │
          v
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Query Engine                             │
│           (Retrieval + OpenAI GPT Generation)                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Backend Framework** | FastAPI, Uvicorn |
| **AI/ML** | OpenAI GPT-3.5-turbo, ChromaDB, LangChain |
| **NLP** | NLTK, BeautifulSoup4 |
| **Financial Data** | Yahoo Finance (yfinance), Alpha Vantage API |
| **Database** | ChromaDB (Vector Database), Redis (Rate Limiting) |
| **File Processing** | PyPDF2, aiofiles |
| **Async/Concurrency** | asyncio, aiohttp |
| **Data Processing** | pandas, pydantic |

## 📋 Prerequisites

- **Python 3.8+**
- **OpenAI API Key** ([Get yours here](https://platform.openai.com/api-keys))
- **Alpha Vantage API Key** ([Register here](https://www.alphavantage.co/support/#api-key))
- **Redis Server** (Optional, for rate limiting)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd company-data-rag-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment setup**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
   ```

4. **NLTK Data Download**
   The application automatically downloads required NLTK data on startup.

## 🚀 Quick Start

1. **Start the application**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 1236
   ```

2. **Access the API Documentation**
   Open your browser and navigate to: `http://localhost:1236`
   
   The interactive Swagger UI will provide complete API documentation and testing interface.

3. **Your first company setup**
   ```bash
   curl -X POST "http://localhost:1236/create_company_folder" \
        -H "Content-Type: application/json" \
        -d '{"company_name": "Apple Inc", "company_description": "Technology company"}'
   ```

## 📚 Core Features

### 🏢 Company Data Management
- **Folder Creation**: Organized storage for each company
- **Multi-Source Ingestion**: URLs and PDF documents
- **Metadata Tracking**: Automatic summary and keyword generation
- **Content Deduplication**: Smart handling of duplicate sources

### 🔍 Intelligent Querying
- **Vector Search**: Semantic similarity using OpenAI embeddings
- **Context-Aware Responses**: LLM-powered answer generation
- **Source Attribution**: Optional source tracking in responses
- **Configurable Results**: Adjustable result count and detail level

### 📊 Financial Intelligence
- **Historical Stock Data**: Complete OHLCV data with custom intervals
- **News Analysis**: AI-powered financial news summarization
- **Market Sentiment**: Ticker-specific relevance scoring
- **Keyword Extraction**: Financial context-aware keyword generation

### 🔧 Processing Pipeline
- **Async Web Scraping**: Concurrent URL processing with retry logic
- **Smart Text Chunking**: Recursive character splitting for optimal embeddings
- **Dual Keyword Extraction**: NLTK + LLM hybrid approach
- **Summary Generation**: AI-powered content summarization

## 🎯 API Endpoints

### 📁 Company Management
| Endpoint | Method | Description |
|----------|---------|-------------|
| `/create_company_folder` | POST | Initialize company data structure |
| `/save_company_data` | POST | Comprehensive data upload (URLs + PDFs) |
| `/upload_company_pdfs` | POST | PDF document processing |
| `/upload_company_urls` | POST | Web content ingestion |
| `/list_companies` | GET | Paginated company listing |
| `/uploaded_data` | GET | Retrieve company data inventory |
| `/delete_company_data` | DELETE | Selective data removal |

### 🔍 Retrieval & Analysis
| Endpoint | Method | Description |
|----------|---------|-------------|
| `/query_company_data` | GET | RAG-powered intelligent querying |
| `/get_company_info` | GET | Document-level summaries and keywords |
| `/get_company_summary` | GET | Overall company analysis |

### 📈 Financial Services
| Endpoint | Method | Description |
|----------|---------|-------------|
| `/historical_stock_data` | GET | Yahoo Finance stock data retrieval |
| `/financial_news_summary` | GET | Alpha Vantage news analysis |

## 💡 Usage Examples

### Adding Company Data
```python
import requests

# Create company folder
response = requests.post("http://localhost:1236/create_company_folder", 
                        json={"company_name": "Tesla Inc"})

# Upload URLs and PDFs
files = {'pdf_files': open('tesla_report.pdf', 'rb')}
data = {
    'company_name': 'Tesla Inc',
    'urls': 'https://tesla.com, https://ir.tesla.com'
}
response = requests.post("http://localhost:1236/save_company_data", 
                        files=files, data=data)
```

### Querying Company Data
```python
# Intelligent querying with RAG
params = {
    'company_name': 'Tesla Inc',
    'query': 'What are the main business segments and revenue drivers?',
    'n_results': 5,
    'include_sources': True
}
response = requests.get("http://localhost:1236/query_company_data", params=params)
print(response.json())
```

### Financial Data Analysis
```python
# Get historical stock data
params = {
    'ticker': 'TSLA',
    'mode': 'range',
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'interval': 'Daily'
}
response = requests.get("http://localhost:1236/historical_stock_data", params=params)

# Get financial news summary
news_params = {
    'ticker': 'TSLA',
    'start_date': '2024-01-01',
    'end_date': '2024-01-07'
}
response = requests.get("http://localhost:1236/financial_news_summary", params=news_params)
```

## 🔧 Configuration

### Environment Variables
```env
# Required
OPENAI_API_KEY=sk-...                    # OpenAI API key
ALPHA_VANTAGE_API_KEY=your_key_here     # Alpha Vantage API key

# Optional
REDIS_URL=redis://localhost:6379        # Redis for rate limiting
MAX_CONCURRENT_REQUESTS=5               # Concurrent processing limit
CHUNK_SIZE=1000                         # Text chunking size
CHUNK_OVERLAP=100                       # Text chunk overlap
```

### Directory Structure
```
company_data/
├── {company_name}/
│   ├── company_description.txt
│   ├── desired_urls.txt
│   ├── uploaded_urls.txt
│   ├── {company_name}_pdfs.txt
│   ├── company_summary.json
│   └── {pdf_files}
db/
└── {chroma_collections}
```

## 🛡️ Security & Performance

- **Rate Limiting**: Redis-based request throttling (10 requests/minute)
- **Async Processing**: Non-blocking I/O for improved performance
- **Concurrent Controls**: Semaphore-based request limiting
- **Error Handling**: Comprehensive exception handling and logging
- **User Agent Rotation**: Anti-bot detection for web scraping

## 🔍 Monitoring & Logging

The application includes comprehensive logging for:
- Request processing status
- Web scraping results
- AI model interactions
- File processing operations
- Error tracking and debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
