# Company Data RAG API

This project is a FastAPI-based web service designed to manage and analyze company-related data. The API is capable of processing uploaded URLs and PDFs, generating summaries and keywords using both traditional NLP techniques (NLTK) and OpenAI's GPT models. Additionally, the system handles historical stock data and financial news summaries using external APIs such as **Alpha Vantage API** and **Yahoo Finance**.

## Features

- **Company Data Management**: Upload and manage URLs and PDFs for company data.
- **Text Summarization and Keyword Extraction**: Generates summaries and keywords using both NLTK and OpenAI's GPT models.
- **ChromaDB for Vector Storage**: Stores the embedded content from URLs and PDFs for querying using ChromaDB.
- **Rate Limiting**: Prevents excessive requests using Redis-based rate limiting.
- **Financial Data via Yahoo Finance**: Fetches and resamples historical stock data from Yahoo Finance.
- **News Summarization via Alpha Vantage API**: Extracts and summarizes financial news articles using the Alpha Vantage API and OpenAI.

## Pipeline

1. **Company Data Upload**: URLs and PDFs are uploaded to the API. PDFs are processed, and their content is split into manageable chunks, which are then embedded and stored in ChromaDB.
2. **Text Scraping**: The uploaded URLs are scraped for text data. Content from the website is extracted, summarized, and keywords are generated.
3. **Text Storage**: Summaries and keywords (both NLTK-based and LLM-based) are stored in JSON files for each company.
4. **Querying**: The stored content is queried using ChromaDB, and the results are enhanced with an LLM to generate a final response.
5. **Stock Data via Yahoo Finance**: Fetch historical stock data from Yahoo Finance, resample based on a defined interval, and return it in a user-friendly format.
6. **Financial News via Alpha Vantage API**: Fetch financial news from the Alpha Vantage API and summarize key articles with key financial insights and relevant keywords.

## API Documentation

### Endpoints

1. **`/query_company_data`** (`GET`): 
   - **Description**: Queries stored data for a specific company using ChromaDB and LLM-based results.
   - **Parameters**: `company_name`, `query`, `n_results`
   - **Response**: Returns a response generated from the query using embedded data and the LLM.

2. **`/save_company_data`** (`POST`):
   - **Description**: Saves company-related data including URLs, PDFs, and a company description.
   - **Parameters**: `company_name`, `company_description` (optional), `urls`, `pdf_files`
   - **Response**: Returns a message indicating successful processing of URLs and PDFs.

3. **`/create_company_folder`** (`POST`):
   - **Description**: Creates or updates a folder for a company.
   - **Parameters**: `company_name`, `company_description` (optional)
   - **Response**: Message confirming the folder creation or update.

4. **`/upload_company_pdfs`** (`POST`):
   - **Description**: Uploads and processes PDF files for a company, storing the content in ChromaDB.
   - **Parameters**: `company_name`, `pdf_files`
   - **Response**: Message indicating the success or failure of PDF uploads.

5. **`/upload_company_urls`** (`POST`):
   - **Description**: Uploads and processes URLs for a company. The URLs are scraped for content and stored in ChromaDB.
   - **Parameters**: `company_name`, `urls`
   - **Response**: Lists successfully processed, failed, and skipped URLs.

6. **`/list_companies`** (`GET`):
   - **Description**: Lists all companies stored in the system.
   - **Parameters**: `page`, `page_size`
   - **Response**: A paginated list of companies with metadata (URLs and PDFs count).

7. **`/retrieve_company_data`** (`GET`):
   - **Description**: Retrieves all stored data for a company.
   - **Parameters**: `company_name`
   - **Response**: Returns the company's uploaded URLs, PDFs, and description.

8. **`/delete_company_data`** (`DELETE`):
   - **Description**: Deletes specific URLs or PDFs for a company.
   - **Parameters**: `company_name`, `pdfs_to_remove`, `urls_to_remove`
   - **Response**: Lists successfully removed and not found items (PDFs or URLs).

9. **`/get_company_summary`** (`GET`):
   - **Description**: Generates an overall summary and keywords for a company based on processed documents.
   - **Parameters**: `company_name`
   - **Response**: Returns an overall summary and a list of keywords for the company.

10. **`/historical_stock_data`** (`GET`):
    - **Description**: Fetches historical stock data for a company ticker using Yahoo Finance.
    - **Parameters**: `ticker`, `start_date`, `end_date` (optional), `interval`
    - **Response**: Returns historical stock data in JSON format based on the specified interval.

11. **`/financial_news_summary`** (`GET`):
    - **Description**: Fetches and summarizes financial news articles using the Alpha Vantage API and generates keywords.
    - **Parameters**: `ticker`, `start_date`, `end_date`
    - **Response**: Returns a weekly summary of financial news and top articles.

### Utility Functions

- **`ensure_directory_exists(directory: Path)`**: Ensures that the specified directory exists.
- **`clean_text(text: str)`**: Cleans unwanted characters from text.
- **`read_file(file_path: Path)`**: Asynchronously reads lines from a file.
- **`write_to_file(file_path: Path, lines: List[str], mode: str = 'a')`**: Asynchronously writes lines to a file.
- **`scrape_website(url: str, max_retries: int)`**: Scrapes a website and returns the cleaned content.
- **`generate_llm_response(query: str, context: List[str])`**: Generates a response from the LLM based on the query and context.
- **`generate_summary(text: str)`**: Generates a summary of a given text using the LLM.
- **`extract_keywords_nltk(text: str, num_keywords: int)`**: Extracts keywords from the text using NLTK.
- **`generate_keywords_llm(text: str, num_keywords: int)`**: Generates keywords from the text using the LLM.
- **`update_company_summary_json(company_name: str, document_name: str, summary: str, keywords_nltk: List[str], keywords_llm: List[str])`**: Updates the company summary JSON file.
- **`read_company_summary(company_name: str)`**: Reads the company summary JSON file.
- **`generate_overall_summary(company_name: str, summaries: List[Dict])`**: Generates an overall summary using LLM based on individual document summaries.
- **`generate_overall_keywords(company_name: str, summaries: List[Dict])`**: Generates overall keywords using LLM based on individual keywords.
- **`fetch_news(ticker: str, start_date: datetime, end_date: datetime)`**: Fetches financial news from the Alpha Vantage API for a company ticker.
- **`summarize_content(content: str, ticker: str)`**: Summarizes content focusing on a company's financial market using the LLM.
- **`generate_keywords(content: str, ticker: str)`**: Generates keywords from financial news content.
  
### Core Functions

- **`create_or_update_company_folder(company_name: str, urls: Optional[List[str]], pdf_files: Optional[List[UploadFile]])`**: Creates or updates a folder for a company, processing URLs and PDFs.
- **`process_urls(urls: List[str], company_name: str, debug: bool = False)`**: Scrapes and processes URLs for a company, storing content in ChromaDB.
- **`process_pdfs(pdf_names: List[str], company_name: str, folder_path: Path)`**: Processes and stores PDF content for a company in ChromaDB.
- **`resample_data(data: DataFrame, interval: Interval)`**: Resamples historical stock data based on a given interval using Yahoo Finance data.
  
## API Integrations

### Alpha Vantage API

The Alpha Vantage API is used for fetching financial news related to a specific company ticker. It provides real-time news sentiment analysis and article summaries.

To use the Alpha Vantage API, you must obtain an API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key) and set it in the `.env` file.

### Yahoo Finance

The Yahoo Finance API is used to fetch historical stock data for a given company ticker. It allows fetching stock data based on various intervals (daily, weekly, monthly, etc.).

## Environment Setup

Clone the repository
cd company-data-rag-api
Install the dependencies:
pip install -r requirements.txt
Set up environment variables by creating a .env file in the root of the project and adding your API keys:
OPENAI_API_KEY=your_openai_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
Start the application:
uvicorn main:app --reload