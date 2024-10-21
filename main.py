import os
from typing import List, Optional, Union, Dict, Set
from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb.config import Settings
from openai import AsyncOpenAI
import logging
from pydantic import BaseModel, Field
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import aiohttp
from bs4 import BeautifulSoup
import random
import asyncio
import aiofiles
from urllib.parse import urljoin
from pathlib import Path
from functools import lru_cache
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
# from redis import asyncio as aioredis
from contextlib import asynccontextmanager
from starlette.exceptions import HTTPException as StarletteHTTPException
from collections import Counter
import json
import requests
from datetime import datetime, date, time, timedelta
import yfinance as yf
from enum import Enum
from pandas import DataFrame

# NLTK setup
nltk_data_dir = Path.home() / 'nltk_data'
os.environ['NLTK_DATA'] = str(nltk_data_dir)

def download_nltk_data():
    import nltk
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.append(str(nltk_data_dir))

    required_packages = ['punkt', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True, download_dir=str(nltk_data_dir))

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configuration
class Config:
    VECTOR_STORE_FOLDER = Path("db")
    COMPANY_DATA_FOLDER = Path("company_data")
    MAX_CONCURRENT_REQUESTS = 5
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    MAX_RETRIES = 3

# Environment Setup
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY is missing in environment variables.")

alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
if not alpha_vantage_api_key:
    raise EnvironmentError("ALPHA_VANTAGE_API_KEY is missing in environment variables.")

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CompanyDataAPI")

## Rate limiting setup
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     redis = aioredis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
#     await FastAPILimiter.init(redis)
#     yield
#     # Shutdown
#     await FastAPILimiter.close()

# FastAPI Application Setup
# app = FastAPI(docs_url="/", title="Company Data RAG API", version="1.0", lifespan=lifespan)
app = FastAPI(docs_url="/", title="Company Data RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Dependency for services (Initialize ChromaDB client and OpenAI client)
@lru_cache()
def get_embedding_function():
    return OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-large"
    )

@lru_cache()
def get_chroma_client():
    return chromadb.PersistentClient(
        path=str(Config.VECTOR_STORE_FOLDER),
        settings=Settings(anonymized_telemetry=False)
    )

@lru_cache()
def get_openai_client():
    return AsyncOpenAI(
        api_key=openai_api_key
    )

# Semaphore for limiting concurrent requests
semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)

# Reusable rate limiter dependency
rate_limit_dependency = Depends(RateLimiter(times=10, seconds=60))

# User Agents for Web Scraping
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:114.0) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Brave/114.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 OPR/94.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Brave/114.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:114.0) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Brave/114.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36 SamsungBrowser/18.0",
    "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Mobile Safari/537.36"
]

# Utility Functions
def ensure_directory_exists(directory: Path):
    """Ensures that the specified directory exists."""
    if not directory.exists():
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def clean_text(text: str) -> str:
    """Cleans unwanted characters from text."""
    return " ".join(text.split())

async def read_file(file_path: Path) -> List[str]:
    """Reads lines from a file asynchronously."""
    if not file_path.exists():
        return []
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    return content.splitlines()

async def write_to_file(file_path: Path, lines: List[str], mode: str = 'a'):
    """Writes lines to a file asynchronously."""
    async with aiofiles.open(file_path, mode) as f:
        await f.write("\n".join(lines) + "\n")

async def scrape_website(url: str, max_retries: int = Config.MAX_RETRIES) -> Optional[str]:
    """Scrapes a website for text and additional info, with retry logic."""
    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                headers = {"User-Agent": random.choice(USER_AGENTS)}
                await asyncio.sleep(random.uniform(1, 2))
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=15) as response:
                        response.raise_for_status()
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")
                        base_url = url
                        text = ' '.join(soup.stripped_strings)
                        cleaned_text = clean_text(text)
                        anchor_tags = [f"URL: {urljoin(base_url, a.get('href', ''))} - {clean_text(a.get_text())}" for a in soup.find_all('a', href=True)]
                        image_tags = [f"Image: {urljoin(base_url, img.get('src', ''))} - {clean_text(img.get('alt', 'No description'))}" for img in soup.find_all('img', src=True)]
                        all_info = [cleaned_text] + anchor_tags + image_tags
                        return "\n".join(all_info)
            except aiohttp.ClientError as e:
                logger.error(f"Attempt {attempt} - Error scraping {url}: {e}")
                if attempt == max_retries:
                    return None
                sleep_time = min(2 ** attempt, 30)
                await asyncio.sleep(sleep_time)
    return None

async def generate_llm_response(query: str, context: List[str]) -> str:
    """Generates a response using the LLM based on the provided query and context."""
    prompt = f"Based on the context provided: '{' '.join(context)}', {query}?"
    try:
        chat_completion = await get_openai_client().chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ])
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate response"
        )

async def generate_summary(text: str) -> str:
    """Generates a summary of the given text using OpenAI's API."""
    prompt = f"Please summarize the following text in a concise manner:\n\n{text[:4000]}"
    try:
        response = await get_openai_client().chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that summarizes text."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return ""

def extract_keywords_nltk(text: str, num_keywords: int = 10, excluded_keywords: Optional[Set[str]] = None) -> List[str]:
    """Extracts keywords from the given text using NLTK, excluding specific keywords."""
    if excluded_keywords is None:
        excluded_keywords = {"https", "http", "url", "www", "image"}  # Default excluded keywords
    # Tokenize and filter out stop words and unwanted keywords
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [
        word for word in words
        if word.isalpha() and word not in stop_words and word not in excluded_keywords
    ]
    # Count word frequencies
    word_freq = Counter(words)
    # Return the most common keywords
    return [word for word, _ in word_freq.most_common(num_keywords)]

async def generate_keywords_llm(text: str, num_keywords: int = 10) -> List[str]:
    """Generates keywords from the given text using OpenAI's API."""
    prompt = f"Please generate {num_keywords} keywords or key phrases that best represent the main topics and concepts in the following text. Separate the keywords with commas:\n\n{text[:4000]}"
    try:
        response = await get_openai_client().chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that generates keywords from text."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100
        )
        keywords = response.choices[0].message.content.strip().split(',')
        return [keyword.strip() for keyword in keywords]
    except Exception as e:
        logger.error(f"Error generating keywords with LLM: {e}")
        return []

async def update_company_summary_json(company_name: str, document_name: str, summary: str, keywords_nltk: List[str], keywords_llm: List[str]):
    """Updates the company_summary.json file with new document information."""
    json_path = Config.COMPANY_DATA_FOLDER / company_name / "company_summary.json"
    try:
        if json_path.exists():
            async with aiofiles.open(json_path, 'r') as f:
                data = json.loads(await f.read())
        else:
            data = []

        new_entry = {
            "document_name": document_name,
            "summary": summary,
            "keywords_nltk": keywords_nltk,
            "keywords_llm": keywords_llm
        }
        # Update existing entry or add new one
        for entry in data:
            if entry["document_name"] == document_name:
                entry.update(new_entry)
                break
        else:
            data.append(new_entry)

        async with aiofiles.open(json_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

    except Exception as e:
        logger.error(f"Error updating company_summary.json for {company_name}: {e}")

async def read_company_summary(company_name: str) -> Dict:
    """Reads the company_summary.json file for a given company."""
    file_path = Config.COMPANY_DATA_FOLDER / company_name / "company_summary.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Summary file for company '{company_name}' not found")
    
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
    return json.loads(content)

async def generate_overall_summary(company_name: str, summaries: List[Dict]) -> str:
    """Generates an overall summary using the LLM based on individual summaries."""
    combined_summaries = "\n".join([f"Document: {item['document_name']}\nSummary: {item['summary']}" for item in summaries])
    prompt = f"Please provide a comprehensive overall summary for the company '{company_name}' based on the following individual document summaries:\n\n{combined_summaries}\n\nOverall Summary:"

    try:
        response = await get_openai_client().chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that creates concise but comprehensive summaries."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating overall summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate overall summary")

async def generate_overall_keywords(company_name: str, summaries: List[Dict]) -> List[str]:
    """Generates overall keywords using the LLM based on individual keywords."""
    all_keywords = [keyword for item in summaries for keyword_list in [item['keywords_nltk'], item['keywords_llm']] for keyword in keyword_list]
    combined_keywords = ", ".join(set(all_keywords))  # Remove duplicates
    prompt = f"Based on the following keywords extracted from various documents for the company '{company_name}':\n\n{combined_keywords}\n\nPlease provide a refined list of 10-15 most relevant and important keywords or key phrases that best represent the company's overall focus and activities. Separate the keywords with commas:"

    try:
        response = await get_openai_client().chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that extracts and refines keywords."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100
        )
        return [keyword.strip() for keyword in response.choices[0].message.content.strip().split(',')]
    except Exception as e:
        logger.error(f"Error generating overall keywords: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate overall keywords")

# Alpha Vantage API Functions
async def fetch_news(ticker: str, start_date: datetime, end_date: datetime, limit: int = 1000):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start_date.strftime("%Y%m%dT%H%M"),
        "time_to": end_date.strftime("%Y%m%dT%H%M"),
        "limit": limit,
        "sort": "CHRONOLOGICAL",
        "apikey": alpha_vantage_api_key
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise HTTPException(status_code=response.status, detail="Failed to fetch news from Alpha Vantage API")

async def summarize_content(content: str, ticker: str):
    prompt = f"Summarize the following content, focusing on {ticker}'s financial market and stock details:\n\n{content[:4000]}"
    response = await get_openai_client().chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a financial analyst summarizing news articles."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

async def generate_keywords(content: str, ticker: str):
    prompt = f"Generate 5 keywords relevant to {ticker}'s financial market and stock details from the following content:\n\n{content[:4000]}"
    response = await get_openai_client().chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a financial analyst extracting keywords from news articles."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )
    return [keyword.strip() for keyword in response.choices[0].message.content.split(',')]

# Models
class SourceInfo(BaseModel):
    pdf_url: str
    content: str

class QueryResponse(BaseModel):
    query: str
    response: str
    n_value: int
    sources: Optional[List[SourceInfo]] = None

class CompanyData(BaseModel):
    company_name: str
    company_description: Optional[str] = None
    uploaded_urls: List[str]
    uploaded_pdfs: List[str]

class DeleteResponse(BaseModel):
    pdfs_removed: List[str] = Field(default_factory=list)
    pdfs_not_found: List[str] = Field(default_factory=list)
    urls_removed: List[str] = Field(default_factory=list)
    urls_not_found: List[str] = Field(default_factory=list)

class OverallSummary(BaseModel):
    company_name: str
    summary: str
    keywords: List[str]

class ErrorResponse(BaseModel):
    message: str

class NewsArticle(BaseModel):
    title: str
    url: str
    time_published: str
    summary: str
    ticker_relevance_score: float
    content_summary: Optional[str] = None
    keywords: Optional[List[str]] = None

class FinancialNewsSummary(BaseModel):
    week_summary: str
    week_keywords: List[str]
    top_articles: List[NewsArticle]

class Interval(str, Enum):
    daily = "Daily"
    total = "Total"

class Mode(str, Enum):
    single_day = "single_day"
    range = "range"

# Function to resample data based on the 'Total' interval
def calculate_total_data(data: DataFrame) -> Dict[str, Union[float, int]]:
    return {
        "Open": float(data['Open'].iloc[0]),            # Opening value from the first day
        "High": float(data['High'].max()),              # Highest value over the time frame
        "Low": float(data['Low'].min()),                # Lowest value over the time frame
        "Close": float(data['Close'].iloc[-1]),         # Closing value on the last day
        "Adj Close": float(data['Adj Close'].iloc[-1]), # Adjusted Close on the last day
        "Volume": int(data['Volume'].sum())             # Total volume over the time frame
    }

# Core Functions
async def create_or_update_company_folder(company_name: str, urls: Optional[List[str]] = None, pdf_files: Optional[List[UploadFile]] = None):
    """Creates or updates a company folder with specified URLs and PDFs."""
    company_path = Config.COMPANY_DATA_FOLDER / company_name
    ensure_directory_exists(company_path)

    file_paths = {
        "desired_urls": company_path / "desired_urls.txt",
        "uploaded_urls": company_path / "uploaded_urls.txt",
        "pdfs": company_path / f"{company_name}_pdfs.txt"
    }

    # Ensure required files exist
    for path in file_paths.values():
        if not path.exists():
            path.touch()

    if urls:
        # Append URLs to the desired_urls.txt
        await write_to_file(file_paths["desired_urls"], urls)

    pdf_names = []
    if pdf_files:
        # Save PDF files
        for pdf_file in pdf_files:
            file_path = company_path / pdf_file.filename
            content = await pdf_file.read()
            if len(content) > 0:
                async with aiofiles.open(file_path, "wb") as buffer:
                    await buffer.write(content)
                pdf_names.append(pdf_file.filename)
            else:
                logger.warning(f"Skipped empty PDF file: {pdf_file.filename}")

    return company_path, pdf_names

async def process_single_url(url: str, company_name: str, collection, debug: bool, existing_uploaded_urls: List[str], debug_folder: Path):
    """Processes a single URL and adds the content to the collection."""
    if url in existing_uploaded_urls:
        logger.info(f"Skipped already processed URL: {url}")
        return None, url, None

    try:
        content = await scrape_website(url)
        if content:
            if debug:
                safe_url = url.replace('/', '_').replace(':', '_')
                debug_file_path = debug_folder / f"{safe_url}_content.txt"
                async with aiofiles.open(debug_file_path, mode='w', encoding='utf-8') as f:
                    await f.write(f"URL: {url}\n\n")
                    await f.write(content)
                logger.info(f"Exported scraped content to {debug_file_path}")

            summary = await generate_summary(content)
            keywords_nltk = extract_keywords_nltk(content)
            keywords_llm = await generate_keywords_llm(content)
            
            await update_company_summary_json(company_name, url, summary, keywords_nltk, keywords_llm)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            splits = text_splitter.split_text(content)
            new_ids = [f"{url}-{i}" for i in range(len(splits))]
            collection.add(
                documents=splits,
                ids=new_ids,
                metadatas=[{
                    "url": url,
                    "summary": summary,
                    "keywords_nltk": ", ".join(keywords_nltk),
                    "keywords_llm": ", ".join(keywords_llm)
                } for _ in splits]
            )
            logger.info(f"Processed and added content from {url} to collection {company_name}")
            return url, None, None  # Successfully processed
        else:
            raise ValueError(f"Failed to scrape content from {url}")
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return None, url, None  # Failed

async def process_urls(urls: List[str], company_name: str, debug: bool = False):
    ensure_directory_exists(Config.COMPANY_DATA_FOLDER)
    company_path = Config.COMPANY_DATA_FOLDER / company_name
    collection = get_chroma_client().get_or_create_collection(
        name=company_name,
        embedding_function=get_embedding_function()
    )
    existing_uploaded_urls = await read_file(company_path / "uploaded_urls.txt")
    # Create a debug folder if debug mode is on
    if debug:
        debug_folder = company_path / "debug_exports"
        ensure_directory_exists(debug_folder)
    else:
        debug_folder = None
    # Process URLs concurrently using asyncio.gather
    tasks = [
        process_single_url(url, company_name, collection, debug, existing_uploaded_urls, debug_folder)
        for url in urls
    ]
    results = await asyncio.gather(*tasks)
    successfully_processed_urls = [url for url, failed_url, skipped_url in results if url]
    failed_urls = [failed_url for url, failed_url, skipped_url in results if failed_url]
    skipped_urls = [skipped_url for url, failed_url, skipped_url in results if skipped_url]
    # Append successfully processed URLs to uploaded_urls.txt
    if successfully_processed_urls:
        await write_to_file(company_path / "uploaded_urls.txt", successfully_processed_urls, mode='a')

    return successfully_processed_urls, failed_urls, skipped_urls
    
async def process_single_pdf(pdf_name: str, company_name: str, folder_path: Path, collection):
    """Processes a single PDF and adds the content to the collection."""
    try:
        file_path = folder_path / pdf_name
        if not file_path.exists() or file_path.stat().st_size == 0:
            logger.error(f"PDF file {pdf_name} is missing or empty")
            return None  # Failed

        loader = PyPDFLoader(str(file_path))
        pages = loader.load_and_split()

        if not pages:
            logger.warning(f"No content extracted from PDF {pdf_name}")
            return None  # Failed

        full_text = " ".join([page.page_content for page in pages])
        summary = await generate_summary(full_text)
        keywords_nltk = extract_keywords_nltk(full_text)
        keywords_llm = await generate_keywords_llm(full_text)

        await update_company_summary_json(company_name, pdf_name, summary, keywords_nltk, keywords_llm)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(pages)
        new_ids = [f"{pdf_name}_{i}" for i in range(len(splits))]
        collection.add(
            documents=[split.page_content for split in splits],
            ids=new_ids,
            metadatas=[{
                "filename": pdf_name,
                "summary": summary,
                "keywords_nltk": ", ".join(keywords_nltk),
                "keywords_llm": ", ".join(keywords_llm)
            } for _ in splits]
        )
        logger.info(f"Processed and added content from PDF {pdf_name} to collection {company_name}")
        return pdf_name  # Successfully processed
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_name}: {e}")
        return None  # Failed

async def process_pdfs(pdf_names: List[str], company_name: str, folder_path: Path):
    collection = get_chroma_client().get_or_create_collection(
        name=company_name,
        embedding_function=get_embedding_function()
    )
    # Process PDFs concurrently using asyncio.gather
    tasks = [
        process_single_pdf(pdf_name, company_name, folder_path, collection)
        for pdf_name in pdf_names
    ]
    results = await asyncio.gather(*tasks)
    successfully_processed_pdfs = [pdf_name for pdf_name in results if pdf_name]
    # Append successfully processed PDFs to the PDF file
    await write_to_file(folder_path / f"{company_name}_pdfs.txt", successfully_processed_pdfs)

    return successfully_processed_pdfs

# Routes
@app.get("/query_company_data", response_model=QueryResponse, summary="Query Company Data", dependencies=[rate_limit_dependency], tags=["Retrieval Augmented Generation"])
async def query_company(
    company_name: str,
    query: str = Query(..., min_length=1, max_length=1000),
    n_results: int = Query(default=3, ge=1, le=10),
    include_sources: bool = Query(default=False)  # Toggle for including sources
):
    """
    ## **Query Company Data**

    **Description:**  
    
    Retrieves augmented data related to a specified company using **retrieval-augmented generation** methods. This is particularly useful for pulling **context-specific information** using large language models (LLMs).

    ### **Request Body**:  
    - **N/A**

    ### **Query Parameters**:
    - **`company_name`**:  The registered name of the company.
    
    - **`query`**:  The specific query or request about the company data.
    
    - **`n_results`**:  The number of results to return. The default is **3**, but this can be adjusted based on requirements.
    
    - **`include_sources`**:  Whether to include sources in the response. Defaults to **False**.

    ### **Response Body**:
    - **`query`**:  The original user query echoed back.
    
    - **`response`**:  The LLM-generated response based on the queried company data.
    
    - **`n_value`**:  The number of documents processed to generate the response.
    
    - **`sources`**:  A list of sources used to generate the response, including URLs or content from PDF documents. **Only shown if `include_sources=True`.**

    ---
    """
    try:
        collection = get_chroma_client().get_collection(
            name=company_name,
            embedding_function=get_embedding_function()
        )
    except ValueError:
        return {
            "query": query,
            "response": f"Company '{company_name}' not found. Please check the company name and try again."
        }

    try:
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        sources = []
        if include_sources:
            for i, doc_list in enumerate(results["documents"]):
                for doc, metadata in zip(doc_list, results["metadatas"][i]):
                    pdf_url = metadata.get("url") or metadata.get("filename", "Unknown")
                    source_info = SourceInfo(
                        pdf_url=pdf_url,
                        content=doc
                    )
                    sources.append(source_info)

        # Generate response from LLM using the context (documents)
        context = [doc for doc_list in results["documents"] for doc in doc_list]
        llm_response = await generate_llm_response(query, context)
        
        # Build the response
        response = QueryResponse(
            query=query,
            response=clean_text(llm_response),
            n_value=n_results
        )
        
        # Only include sources if the toggle is turned on
        if include_sources:
            response.sources = sources

        return response

    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query"
        )

@app.post("/create_company_folder", summary="Create/Update Company Folder", dependencies=[rate_limit_dependency], tags=["Company Data Management"])
async def create_company_folder(
    company_name: str,
    company_description: Optional[str] = None
):
    """
## **Create/Update Company Folder**

**Description:**  

Creates or updates a storage folder for a specified company. This endpoint is used to initialize or update the storage for company-specific data.

### **Request Body**:
- **`company_name`**:  The registered name of the company.
  
- **`company_description`**:  *(Optional)* A brief description of the company.

### **Response Body**:
- **`message`**:  A confirmation message detailing the actions taken, such as folder creation or description update.

---
"""
    try:
        folder_path = Config.COMPANY_DATA_FOLDER / company_name
        description_file_path = folder_path / "company_description.txt"
        folder_existed = folder_path.exists()

        ensure_directory_exists(folder_path)

        # Create empty text files
        file_paths = [
            folder_path / "desired_urls.txt",
            folder_path / "uploaded_urls.txt",
            folder_path / f"{company_name}_pdfs.txt"
        ]

        for file_path in file_paths:
            if not file_path.exists():
                file_path.touch()

        if company_description is not None:
            await write_to_file(description_file_path, [company_description], mode='w')
            description_action = "updated" if folder_existed else "created"
        else:
            description_action = "unchanged"

        if folder_existed:
            message = f"Company data '{company_name}' already exists. "
        else:
            message = f"Folder for company '{company_name}' created successfully. "

        message += f"Company description {description_action}. "

        return {"message": message}
    except Exception as e:
        action = "updating" if folder_existed else "creating"
        logger.error(f"Error {action} folder for company '{company_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to {action} folder for company '{company_name}'"
        )

@app.post("/save_company_data", summary="Save Company Data", dependencies=[rate_limit_dependency], tags=["Company Data Management"])
async def save_company_data(
    company_name: str = Form(..., 
        description="The name of the company.",
        examples=[
            {
                "example": "Meta",
                "summary": "Example Company Name",
                "description": "A typical example of a company name."
            }
        ]
    ),
    company_description: Optional[str] = Form(None, 
        description="A brief description of the company.",
        examples=[
            {
                "example": "American multinational technology conglomerate based in Menlo Park, California.",
                "summary": "Example Company Description",
                "description": "A typical example of a company description."
            }
        ]
    ),
    urls: Optional[str] = Form(None, 
        description="Comma-separated URLs of the company's web pages.",
        examples=[
            {
                "example": "https://meta.com, https://about.meta.com/company-info/",
                "summary": "Example URLs",
                "description": "A typical example of URLs related to the company."
            }
        ]
    ),
    pdf_files: List[UploadFile] = File([], 
        description="PDF files related to the company.",
        examples=[
            {
                "example": ["Meta_Platforms_Inc.pdf", "Meta_Annual_Report.pdf"],
                "summary": "Example PDF Files",
                "description": "A typical example of file names for PDF uploads."
            }
        ]
    )
):
    """
## **Save Company Data**

**Description:**  

Saves or updates comprehensive data about a company, including web URLs and PDF files.

### **Request Body**:
- **`company_name`**:  The name of the company.
  
- **`company_description`**:  *(Optional)* A brief description of the company.
  
- **`urls`**:  *(Optional)* Comma-separated URLs associated with the company.
  
- **`pdf_files`**:  *(Optional)* A list of PDF files related to the company.

### **Response Body**:
- **`message`**:  Status message about the process completion.
  
- **`processed_urls`**:  List of URLs that were successfully processed.
  
- **`failed_urls`**:  List of URLs that failed during processing.
  
- **`skipped_urls`**:  URLs skipped due to previous processing.
  
- **`processed_pdfs`**:  List of PDFs that were successfully processed.
  
- **`company_description_updated`**:  Boolean indicating if the company description was updated.

---
"""
    try:
        url_list = [url.strip() for url in urls.split(',')] if urls else []

        # Handle PDF files
        pdf_files = pdf_files or []

        folder_path, pdf_names = await create_or_update_company_folder(company_name, url_list, pdf_files)

        # Handle company description
        description_file_path = folder_path / "company_description.txt"
        if company_description is not None:
            await write_to_file(description_file_path, [company_description], mode='w')
            logger.info(f"Company description for {company_name} created/updated.")

        processed_urls = []
        failed_urls = []
        skipped_urls = []
        processed_pdfs = []

        if url_list:
            processed_urls, failed_urls, skipped_urls = await process_urls(url_list, company_name)

        if pdf_files:
            processed_pdfs = await process_pdfs(pdf_names, company_name, folder_path)

        if not url_list and not pdf_files and company_description is None:
            return {"message": f"Company folder for {company_name} created or updated. No data to process."}

        return {
            "message": "Company data processed successfully",
            "processed_urls": processed_urls,
            "failed_urls": failed_urls,
            "skipped_urls": skipped_urls,
            "processed_pdfs": processed_pdfs,
            "company_description_updated": company_description is not None
        }
    except Exception as e:
        logger.error(f"Error processing company data: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/upload_company_pdfs", summary="Upload Company PDFs", dependencies=[rate_limit_dependency], tags=["Company Data Management"])
async def upload_company_pdfs(
    company_name: str = Form(...),
    pdf_files: List[UploadFile] = File(...)
):
    """
## **Upload Company PDFs**

**Description:**  

Allows uploading and processing of multiple PDF documents for a specified company.

### **Request Body**:
- **`company_name`**:  The name of the company.
  
- **`pdf_files`**:  A list of PDF files to be uploaded.

### **Response Body**:
- **`message`**:  Confirmation message of the upload and processing status.
  
- **`uploaded_pdfs`**:  List of PDFs that were successfully uploaded and processed.

---
"""
    try:
        folder_path = Config.COMPANY_DATA_FOLDER / company_name
        
        # Check if the company folder exists
        if not folder_path.exists():
            return {
                "message": f"Folder for company '{company_name}' does not exist."
            }

        uploaded_pdfs = []

        # Use a set to keep track of unique filenames
        processed_filenames = set()

        for pdf_file in pdf_files:
            if pdf_file.filename.endswith('.pdf'):
                if pdf_file.filename not in processed_filenames:
                    file_path = folder_path / pdf_file.filename
                    async with aiofiles.open(file_path, "wb") as buffer:
                        content = await pdf_file.read()
                        await buffer.write(content)

                    # Check the file size to ensure it was saved correctly
                    file_size = os.stat(file_path)
                    if file_size.st_size > 0:
                        uploaded_pdfs.append(pdf_file.filename)
                        processed_filenames.add(pdf_file.filename)
                    else:
                        logger.error(f"Error: PDF file {pdf_file.filename} appears to be empty after upload.")
                else:
                    logger.warning(f"Skipped duplicate PDF file: {pdf_file.filename}")
            else:
                logger.warning(f"Skipped non-PDF file: {pdf_file.filename}")

        # Process PDFs and add to vector store
        if uploaded_pdfs:
            processed_pdfs = await process_pdfs(uploaded_pdfs, company_name, folder_path)
        else:
            processed_pdfs = []

        return {
            "message": f"PDFs uploaded and processed successfully for company '{company_name}'",
            "uploaded_pdfs": processed_pdfs
        }
    except Exception as e:
        logger.error(f"Error uploading PDFs for company '{company_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload PDFs for company '{company_name}'"
        )

@app.post("/upload_company_urls", summary="Upload Company URLs", dependencies=[rate_limit_dependency], tags=["Company Data Management"])
async def upload_company_urls(
    company_name: str = Form(...),
    urls: str = Form(...)
):
    """
## **Upload Company URLs**

**Description:**  

Allows uploading and processing of multiple URLs associated with a specified company.

### **Request Body**:
- **`company_name`**:  The name of the company.
  
- **`urls`**:  Comma-separated list of URLs to be processed.

### **Response Body**:
- **`message`**:  Confirmation message detailing the processing results.
  
- **`processed_urls`**:  List of URLs that were successfully processed.
  
- **`failed_urls`**:  List of URLs that failed to process.
  
- **`skipped_urls`**:  List of URLs that were skipped during processing.

---
"""
    try:
        folder_path = Config.COMPANY_DATA_FOLDER / company_name
        
        # Check if the company folder exists
        if not folder_path.exists():
            return {
                "message": f"Folder for company '{company_name}' does not exist."
            }

        # Split the input URLs and remove duplicates
        url_list = list(set(url.strip() for url in urls.split(',')))

        desired_urls_path = folder_path / "desired_urls.txt"
        uploaded_urls_path = folder_path / "uploaded_urls.txt"

        # Ensure the files exist
        if not desired_urls_path.exists():
            desired_urls_path.touch()

        if not uploaded_urls_path.exists():
            uploaded_urls_path.touch()

        # Append the URLs to desired_urls.txt
        await write_to_file(desired_urls_path, url_list, mode='a')

        # Process the URLs
        successfully_processed_urls, failed_urls, skipped_urls = await process_urls(url_list, company_name)

        # Update desired_urls.txt to remove failed URLs
        if failed_urls:
            current_desired_urls = await read_file(desired_urls_path)
            updated_desired_urls = [url for url in current_desired_urls if url not in failed_urls]
            await write_to_file(desired_urls_path, updated_desired_urls, mode='w')  # Overwrite with successful and skipped URLs

        return {
            "message": f"URLs processed successfully for company '{company_name}'",
            "processed_urls": successfully_processed_urls,
            "failed_urls": failed_urls,
            "skipped_urls": skipped_urls
        }
    except Exception as e:
        logger.error(f"Error processing URLs for company '{company_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process URLs for company '{company_name}'"
        )

@app.get("/list_companies", summary="List All Companies", dependencies=[rate_limit_dependency], tags=["Company Data Management"])
async def list_companies(
    page: int = 1, 
    page_size: int = 10
):
    """
## **List All Companies**

**Description:**  

Lists all companies for which data is stored in the system.

### **Request Body**:  
- **N/A**

### **Query Parameters**:
- **`page`**:  Specifies the page number for pagination.
  
- **`page_size`**:  Specifies the number of items per page.

### **Response Body**:
- **`companies`**:  List of companies with basic details like name, description status, URL count, and PDF count.
  
- **`total_companies`**:  Total number of companies available.
  
- **`page`**:  Current page number.
  
- **`page_size`**:  Number of items per page.

---
"""
    try:
        ensure_directory_exists(Config.COMPANY_DATA_FOLDER)
        companies = []

        for company_folder in Config.COMPANY_DATA_FOLDER.iterdir():
            if company_folder.is_dir():
                company_info = {
                    "name": company_folder.name,
                    "has_description": (company_folder / "company_description.txt").exists(),
                    "url_count": 0,
                    "pdf_count": 0
                }

                # Count URLs
                uploaded_urls_path = company_folder / "uploaded_urls.txt"
                existing_urls = await read_file(uploaded_urls_path)
                company_info["url_count"] = len(existing_urls)

                # Count PDFs
                pdf_file_path = company_folder / f"{company_folder.name}_pdfs.txt"
                existing_pdfs = await read_file(pdf_file_path)
                company_info["pdf_count"] = len(existing_pdfs)

                companies.append(company_info)

        # Pagination
        total_companies = len(companies)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_companies = companies[start:end]

        return {
            "companies": paginated_companies,
            "total_companies": total_companies,
            "page": page,
            "page_size": page_size,
        }
    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list companies"
        )

@app.get("/uploaded_data", summary="Retrieve Uploaded Data", response_model=Union[CompanyData, ErrorResponse], dependencies=[rate_limit_dependency], tags=["Company Data Management"])
async def get_company_data(company_name: str):
    """
## **Retrieve Uploaded Data**

**Description:**  

Retrieves all data uploaded for a specified company, including URLs, PDFs, and company descriptions.

### **Request Body**:
- **`company_name`**:  The name of the company.

### **Response Body**:
- **`company_name`**:  Name of the company.
  
- **`company_description`**:  Description of the company, if available.
  
- **`uploaded_urls`**:  List of URLs uploaded for the company.
  
- **`uploaded_pdfs`**:  List of PDFs uploaded for the company.

---
"""
    try:
        folder_path = Config.COMPANY_DATA_FOLDER / company_name
        if not folder_path.exists():
            return ErrorResponse(message=f"Company '{company_name}' not found. Please check the company name and try again.")

        # Get uploaded URLs
        uploaded_urls = await read_file(folder_path / "uploaded_urls.txt")

        # Get uploaded PDFs
        uploaded_pdfs = await read_file(folder_path / f"{company_name}_pdfs.txt")

        # Get company description if it exists
        description_file_path = folder_path / "company_description.txt"
        company_description = None
        if description_file_path.exists():
            company_description = "\n".join(await read_file(description_file_path))

        return CompanyData(
            company_name=company_name,
            company_description=company_description,
            uploaded_urls=uploaded_urls,
            uploaded_pdfs=uploaded_pdfs
        )

    except HTTPException as http_exc:
        if http_exc.status_code == 404:
            return {"message": f"Company '{company_name}' not found. Please check the company name and try again."}
        raise http_exc
    except Exception as e:
        logger.error(f"Error retrieving data for company '{company_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve data for company '{company_name}'"
        )

@app.delete("/delete_company_data", summary="Delete Company Data", response_model=DeleteResponse, dependencies=[rate_limit_dependency], tags=["Company Data Management"])
async def delete_company_data(
    company_name: str,
    pdfs_to_remove: Optional[str] = Query(None, description="Comma-separated list of PDF names to remove"),
    urls_to_remove: Optional[str] = Query(None, description="Comma-separated list of URLs to remove")
):
    """
## **Delete Company Data**

**Description:**  

Deletes specified URLs and/or PDFs from a company's data repository.

### **Request Body**:
- **`company_name`**:  The name of the company.
  
- **`pdfs_to_remove`**:  *(Optional)* Comma-separated list of PDF names to be removed.
  
- **`urls_to_remove`**:  *(Optional)* Comma-separated list of URLs to be removed.

### **Response Body**:
- **`pdfs_removed`**:  List of PDFs successfully removed.
  
- **`pdfs_not_found`**:  List of PDFs that were not found and thus not removed.
  
- **`urls_removed`**:  List of URLs successfully removed.
  
- **`urls_not_found`**:  List of URLs that were not found and thus not removed.

---
"""
    try:
        ensure_directory_exists(Config.COMPANY_DATA_FOLDER)
        folder_path = Config.COMPANY_DATA_FOLDER / company_name
        if not folder_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Company folder for '{company_name}' not found"
            )

        collection = get_chroma_client().get_collection(
            name=company_name,
            embedding_function=get_embedding_function()
        )

        response = DeleteResponse()

        if pdfs_to_remove:
            pdf_list = [pdf.strip() for pdf in pdfs_to_remove.split(',')]
            pdf_file_path = folder_path / f"{company_name}_pdfs.txt"

            existing_pdfs = await read_file(pdf_file_path)

            pdfs_to_keep = []
            for pdf in existing_pdfs:
                if pdf in pdf_list:
                    # Remove from ChromaDB
                    collection.delete(where={"filename": pdf})
                    # Delete the actual PDF file
                    pdf_path = folder_path / pdf
                    if pdf_path.exists():
                        os.remove(pdf_path)
                    response.pdfs_removed.append(pdf)
                else:
                    pdfs_to_keep.append(pdf)

            await write_to_file(pdf_file_path, pdfs_to_keep, mode='w')

            response.pdfs_not_found = list(set(pdf_list) - set(response.pdfs_removed))

        if urls_to_remove:
            url_list = [url.strip() for url in urls_to_remove.split(',')]
            uploaded_urls_path = folder_path / "uploaded_urls.txt"
            desired_urls_path = folder_path / "desired_urls.txt"

            existing_urls = await read_file(uploaded_urls_path)

            for url in url_list:
                if url in existing_urls:
                    # Remove from ChromaDB
                    collection.delete(where={"url": url})
                    response.urls_removed.append(url)
                else:
                    response.urls_not_found.append(url)

            await write_to_file(uploaded_urls_path, [url for url in existing_urls if url not in response.urls_removed], mode='w')
            await write_to_file(desired_urls_path, [url for url in existing_urls if url not in response.urls_removed], mode='w')

        return response
    except HTTPException as http_exc:
        if http_exc.status_code == 404:
            return {"message": f"Company '{company_name}' not found. Please check the company name and try again."}
        raise http_exc
    except Exception as e:
        logger.error(f"Error deleting data for company '{company_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete data"
        )

@app.get("/get_company_info", summary="Get Company Information", dependencies=[rate_limit_dependency], tags=["Company Information, Summary And Keywords"])
async def get_company_summary(company_name: str):
    """
## **Get Company Information**

**Description:**  

Retrieves detailed information about a company, including summaries and keywords extracted from processed documents.

### **Request Body**:
- **`company_name`**:  The name of the company.

### **Response Body**:  
A JSON array containing detailed summaries and keywords for each document related to the company.

Each element in the array has the following structure:
- **`document_name`**:  The name of the document.
  
- **`summary`**:  A summarized description of the document content.
  
- **`keywords_nltk`**:  A list of keywords extracted using traditional NLP techniques.
  
- **`keywords_llm`**:  A list of keywords generated using an LLM (Large Language Model).

---
"""
    try:
        json_path = Config.COMPANY_DATA_FOLDER / company_name / "company_summary.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail=f"Summary for company '{company_name}' not found")

        async with aiofiles.open(json_path, 'r') as f:
            data = json.loads(await f.read())
        return data
    except Exception as e:
        logger.error(f"Error retrieving company summary for {company_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve company summary")

@app.get("/get_company_summary", response_model=OverallSummary, summary="Get Company Summary and Keywords", dependencies=[rate_limit_dependency], tags=["Company Information, Summary And Keywords"])
async def get_overall_summary(company_name: str):
    """
## **Get Company Summary and Keywords**

**Description:**  

Generates a comprehensive summary and list of keywords for the specified company based on all processed documents.

### **Request Body**:
- **`company_name`**:  The name of the company.

### **Response Body**:
- **`company_name`**:  Name of the company.
  
- **`summary`**:  Generated summary of the company.
  
- **`keywords`**:  List of important keywords associated with the company.

---
"""
    try:
        company_data = await read_company_summary(company_name)
        overall_summary = await generate_overall_summary(company_name, company_data)
        overall_keywords = await generate_overall_keywords(company_name, company_data)

        return OverallSummary(
            company_name=company_name,
            summary=overall_summary,
            keywords=overall_keywords
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error generating overall summary for company '{company_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate overall summary for company '{company_name}'")

@app.get("/historical_stock_data", summary="Get Historical Stock Data For A Ticker", dependencies=[rate_limit_dependency], tags=["Financial Data Services"])
async def get_historical_stock_data(
    ticker: str = Query(..., description="Stock Ticker Symbol"),
    mode: Mode = Query(..., description="Mode to fetch stock data: 'Single Day' or 'Range'"),
    date: str = Query(None, description="Date for 'Single Day' mode (YYYY-MM-DD)"),
    start_date: str = Query(None, description="Start Date for 'Range' mode (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End Date for 'Range' mode (YYYY-MM-DD). If not provided, the current date will be used."),
    interval: Interval = Query(Interval.daily, description="Data interval ('Daily' or 'Total')")
):
    """
    ## **Get Historical Stock Data For A Ticker**

    **Description:** 

    Retrieves historical stock data for a specified ticker symbol from **Yahoo Finance** over a given time period. This endpoint is useful for financial analysis and tracking stock performance over specific intervals.

    ### **Request Body**:  
    - **N/A**

    ### **Query Parameters**:
    - **`ticker`**:  The stock ticker symbol, e.g., 'AAPL' for Apple Inc.

    - **`mode`**:  Select whether to fetch data for a 'Single Day' or a 'Range'.

    - **`date`**:  The date to fetch stock data for (used when `mode=single_day`).

    - **`start_date`**:  Start date for the data retrieval (used when `mode=range`), formatted as YYYY-MM-DD.

    - **`end_date`**:  *(Optional)* End date for the data retrieval (used when `mode=range`), formatted as YYYY-MM-DD. If not provided, the current date is used.

    - **`interval`**:  Specifies the data interval. Options include 'Daily' for daily stock prices or 'Total' for cumulative data over the period.

    ### **Response Body**:
    - **`ticker`**:  The ticker symbol for which data was requested.

    - **`start_date` or `date`**:  The start date (or specific date for 'Single Day') for the historical data retrieval.

    - **`end_date`**:  The end date for the historical data retrieval (for 'Range' mode).

    - **`interval`**:  The interval of the data retrieved, specified by the user ('Daily' or 'Total').
    
    - **`historical_data`**:  An array of objects, each representing stock data within the specified range or on a single day.

    ---
    """
    try:
        if mode == Mode.single_day:
            if not date:
                raise HTTPException(status_code=400, detail="Date must be provided for 'Single Day' mode.")
            
            # Validate and parse date
            try:
                single_day = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

            # Fetch data for a single day
            historical_data = yf.download(ticker, start=single_day, end=single_day + timedelta(days=1))
            
            # Return the data for the single day
            historical_data_dict = historical_data.reset_index().to_dict(orient='records')
            return {
                "ticker": ticker,
                "date": single_day.isoformat(),
                "interval": interval,
                "historical_data": historical_data_dict
            }

        elif mode == Mode.range:
            if not start_date:
                raise HTTPException(status_code=400, detail="Start date must be provided for 'Range' mode.")
            
            # Validate and parse start_date
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Please use YYYY-MM-DD.")

            # Handle end_date
            if end_date:
                try:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    raise HTTPException(status_code=400, detail="Invalid end_date format. Please use YYYY-MM-DD.")
            else:
                end_date = date.today()

            # Fetch historical data
            historical_data = yf.download(ticker, start=start_date, end=end_date)

            # Handle the 'total' interval case
            if interval == Interval.total:
                total_data = calculate_total_data(historical_data)
                return {
                    "ticker": ticker,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "interval": interval,
                    "total_data": total_data
                }

            # For 'daily' interval
            else:
                historical_data_dict = historical_data.reset_index().to_dict(orient='records')
                return {
                    "ticker": ticker,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "interval": interval,
                    "historical_data": historical_data_dict
                }
    
    except Exception as e:
        logger.error(f"Error fetching historical stock data for ticker '{ticker}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch historical stock data for ticker '{ticker}'"
        )

@app.get("/financial_news_summary", response_model=FinancialNewsSummary, summary='Get Financial News Summary And Keywords For A Ticker', dependencies=[rate_limit_dependency], tags=["Financial Data Services"])
async def get_financial_news_summary(
    ticker: str = Query(..., description="Company Ticker Symbol"),
    start_date: str = Query(..., description="Start Date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End Date (YYYY-MM-DD)")
):
    """
## **Get Financial News Summary And Keywords For A Ticker**

**Description:**  

Fetches and summarizes financial news for a specified ticker, generating an overall summary and keywords based on the content.

### **Request Body**:  
- **N/A**

### **Query Parameters**:
- **`ticker`**:  The company ticker symbol.
  
- **`start_date`**:  Start date for the news retrieval.
  
- **`end_date`**:  End date for the news retrieval.

### **Response Body**:
- **`week_summary`**:  A summary of the week's news concerning the ticker.
  
- **`week_keywords`**:  Keywords extracted from the news content.
  
- **`top_articles`**:  List of top articles with relevant details such as title, summary, and URL.

---
"""
    non_scrapable_urls = ["barrons.com", "example.com", "othersite.com"]
    try:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    news_data = await fetch_news(ticker, start_date, end_date)
    if 'feed' not in news_data:
        raise HTTPException(status_code=404, detail="No news articles found or there was an error in the API response.")

    filtered_articles = [
        NewsArticle(
            title=article['title'],
            url=article['url'],
            time_published=article['time_published'],
            summary=article['summary'],
            ticker_relevance_score=next(
                (float(item['relevance_score']) for item in article['ticker_sentiment'] if item['ticker'] == ticker),
                0.0
            )
        )
        for article in news_data['feed']
    ]
    filtered_articles.sort(key=lambda x: x.ticker_relevance_score, reverse=True)

    # Scrape and summarize articles
    successful_articles = []
    skipped_articles = []
    for article in filtered_articles:
        if len(successful_articles) >= 5:
            break

        # Check if the URL is in the non-scrapable list
        if any(non_scrapable_url in article.url for non_scrapable_url in non_scrapable_urls):
            logger.info(f"Skipping article from non-scrapable URL: {article.url}")
            skipped_articles.append(article)
            continue

        content = await scrape_website(article.url)
        if content:
            article.content_summary = await summarize_content(content, ticker)
            article.keywords = await generate_keywords(content, ticker)
            successful_articles.append(article)
        else:
            logger.warning(f"Failed to scrape content from {article.url}")

    if not successful_articles:
        raise HTTPException(status_code=500, detail="Failed to scrape content from all articles.")

    if len(successful_articles) < 5:
        logger.warning(f"Only {len(successful_articles)} articles were successfully scraped and processed.")
        logger.info(f"Skipped {len(skipped_articles)} non-scrapable articles.")

    # Generate overall summary and keywords
    overall_summary = await generate_overall_summary(
        ticker,
        [{"document_name": article.title, "summary": article.content_summary} for article in successful_articles]
    )
    overall_keywords = await generate_overall_keywords(
        ticker,
        [{"keywords_nltk": article.keywords, "keywords_llm": article.keywords} for article in successful_articles]
    )

    return FinancialNewsSummary(
        week_summary=overall_summary,
        week_keywords=overall_keywords,
        top_articles=successful_articles
    )

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc, tags=["System Management"]):
    if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"message": "Too Many Requests. Rate Limit Exceeded"},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1236, reload=True)
