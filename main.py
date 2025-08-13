import os
import asyncio
import pdfplumber
import logging
import requests
import json
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict, deque

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Langchain imports
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings as LangchainEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys and constants
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
EXPECTED_TOKEN = os.environ.get("FARMER_BOT_TOKEN", "sample_token_123")
CACHE_DIR = os.path.join("/tmp", "farmer_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farmer_bot")

# App setup
app = FastAPI(title="Farmer Assistant Chatbot")
router = APIRouter()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=6)

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
VECTORSTORE_PATH = os.path.join(CACHE_DIR, "vectorstore.faiss")
DOCS_METADATA_PATH = os.path.join(CACHE_DIR, "docs_metadata.json")
processed_docs: List[Document] = []
vectorstore = None

# -----------------------------
# Conversation memory (rolling)
# -----------------------------
MAX_HISTORY = 6  # number of messages (user+assistant turns) to keep
conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


def build_conversation_context(session_id: str) -> str:
    """Build a simple chat transcript string from recent turns."""
    history = conversation_history.get(session_id, [])
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in history)


# Authentication
# Return the token when valid so we can use it as session_id

def verify_bearer_token(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return token


# Models
class ChatRequest(BaseModel):
    message: str


class SentenceTransformerEmbeddings(LangchainEmbeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()


embeddings = SentenceTransformerEmbeddings()


# PDF Processing
async def extract_text_from_pdf_async(path: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: _extract_text_sync(path))


def _extract_text_sync(filename: str) -> str:
    text_parts = []
    with pdfplumber.open(filename) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text.strip())
                else:
                    logger.warning(f"⚠️ Page {i} in '{os.path.basename(filename)}' could not be parsed.")
                    text_parts.append(f"[[UNREADABLE PAGE {i}]]")
            except Exception as e:
                logger.error(f"❌ Error reading page {i} in '{os.path.basename(filename)}': {e}")
                text_parts.append(f"[[UNREADABLE PAGE {i} - ERROR: {e}]]")
    return "\n".join(text_parts)



def create_semantic_chunks(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]


async def get_or_create_document_pipeline(document_path: str):
    text = await extract_text_from_pdf_async(document_path)

    if not text.strip():
        logger.warning(f"⚠️ No text extracted from {document_path}. Skipping...")
        return None, []

    documents = create_semantic_chunks(text)
    if not documents:
        logger.warning(f"⚠️ No chunks created for {document_path}. Skipping...")
        return None, []

    loop = asyncio.get_event_loop()
    vectorstore = await loop.run_in_executor(
        executor,
        lambda: FAISS.from_documents(documents, embeddings)
    )
    return vectorstore, documents


def create_hybrid_retriever(documents: List[Document], vectorstore, k: int = 6):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = max(2, k // 3)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k - bm25_retriever.k})
    return EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.3, 0.7])


# Model Router
class ModelRouter:
    def __init__(self):
        self.fast_model = ChatGroq(
            model=os.environ.get("GROQ_FAST_MODEL", "llama3-8b-8192"),
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1, max_tokens=1024
        )
        self.powerful_model = ChatGroq(
            model=os.environ.get("GROQ_POWER_MODEL", "llama3-70b-8192"),
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.1, max_tokens=2048
        )

    def route_model(self, question: str, context_length: int = 0):
        if len(question.split()) > 15 or context_length > 2000:
            return self.powerful_model
        return self.fast_model


model_router = ModelRouter()


# Weather
async def get_weather_forecast(location: str, days: int = 5) -> str:
    if not OPENWEATHER_API_KEY:
        return "Weather API key not set."
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location},IN&limit=1&appid={OPENWEATHER_API_KEY}"
    geo_data = requests.get(geo_url).json()
    if not geo_data:
        return "Could not find location."
    lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    forecast_data = requests.get(forecast_url).json()
    forecasts = []
    for entry in forecast_data['list'][:days*8]:
        date = entry['dt_txt']
        temp = entry['main']['temp']
        desc = entry['weather'][0]['description']
        forecasts.append(f"{date}: {temp:.1f}°C, {desc}")
    return "\n".join(forecasts)


# Startup - process or load documents
@app.on_event("startup")
async def load_or_process_documents():
    global processed_docs, vectorstore

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    pdf_files = sorted([os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")])

    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCS_METADATA_PATH):
        with open(DOCS_METADATA_PATH, "r") as f:
            metadata = json.load(f)
        cached_files = metadata.get("files", [])
        if cached_files == pdf_files:
            logger.info("✅ Using cached vectorstore.")
            vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            processed_docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in metadata["docs"]]
            return

    logger.info("📄 Processing new or updated documents...")
    all_docs = []
    all_vs = []

    for p in pdf_files:
        if os.path.exists(DOCS_METADATA_PATH):
            with open(DOCS_METADATA_PATH) as f:
                meta = json.load(f)
            # Skip processing if already in cache
            if any(doc["metadata"].get("source") == p for doc in meta.get("docs", [])):
                continue

        vs, docs = await get_or_create_document_pipeline(p)
        if docs:
            for d in docs:
                d.metadata["source"] = p
            all_docs.extend(docs)
            all_vs.append(vs)

    if not all_vs:
        logger.error("❌ No valid PDFs found or all were unreadable.")
        return

    combined_vs = all_vs[0]
    for vs in all_vs[1:]:
        combined_vs.merge_from(vs)

    combined_vs.save_local(VECTORSTORE_PATH)
    processed_docs = all_docs
    vectorstore = combined_vs

    with open(DOCS_METADATA_PATH, "w") as f:
        json.dump({
            "files": pdf_files,
            "docs": [{"content": d.page_content, "metadata": d.metadata} for d in all_docs]
        }, f)

    logger.info("✅ Documents processed and cached.")



# Keywords
WEATHER_KEYWORDS = ["weather", "rain", "temperature", "forecast", "irrigate", "frost", "humidity"]
AGRICULTURE_KEYWORDS = ["crop", "farming", "harvest", "fertilizer", "pesticide", "soil", "yield", "agriculture", "irrigation"]


# Chat endpoint (with rolling memory + end_chat cleanup)
@router.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    location: Optional[str] = Query(None),
    end_chat: Optional[bool] = Query(False),
    session_token: str = Depends(verify_bearer_token),
):
    message = req.message.strip()
    weather_context = ""
    weather_raw = None
    current_datetime = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

    # Use token as session id (isolates memory per authenticated user)
    session_id = session_token or (EXPECTED_TOKEN if EXPECTED_TOKEN != "sample_token_123" else "default_user")

    # Build conversation history context
    history_str = build_conversation_context(session_id)

    # Weather logic
    if any(kw in message.lower() for kw in WEATHER_KEYWORDS):
        if not location:
            return {
                "response": "🌱 **KrishiBandhu Advice** 🌱\n\nPlease provide your location or postal code to get accurate weather details.",
                "weather_raw": None
            }
        weather_raw = await get_weather_forecast(location)
        weather_context = f"Weather forecast for {location} as of {current_datetime}:\n{weather_raw}"

    is_agriculture_query = any(kw in message.lower() for kw in AGRICULTURE_KEYWORDS)

    # Agriculture + weather + docs pipeline
    if is_agriculture_query and processed_docs and vectorstore:
        docs = processed_docs.copy()
        if weather_context:
            docs.append(Document(page_content=weather_context, metadata={"source": "weather_api"}))

        retriever = create_hybrid_retriever(docs, vectorstore)
        llm = model_router.route_model(message, sum(len(d.page_content) for d in docs))

        prompt = PromptTemplate(
            template=(
                "You are **KrishiBandhu**, an AI assistant for Indian farmers.\n"
                "Today is {date_time}.\n"
                "Use the context from agricultural documents and the latest weather forecast to give helpful, clear, and actionable answers.\n"
                "Conversation so far:\n{history}\n"
                "Format your answer in a **systematic, well-structured report** with clear sections:\n"
                "1. Short answer first if applicable.\n"
                "2. Detailed reasoning.\n"
                "3. Stats and percentages if relevant.\n"
                "4. Prominently mention weather impact.\n"
                "5. Cautions or advice.\n"
                "Mention sources when applicable.\n\n"
                "**Context:**\n{context}\n\n"
                "**Question:** {question}\n\n"
                "**Answer:**"
            ),
            input_variables=["context", "question", "date_time", "history"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt.partial(date_time=current_datetime, history=history_str)},
            return_source_documents=True
        )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, lambda: qa_chain.invoke({"query": message}))
        final_response_text = result["result"]

    else:
        # General queries
        llm = model_router.route_model(message)
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            executor,
            lambda: llm.invoke(
                f"[Current Date & Time: {current_datetime}]\n{history_str}\nUser: {message}\n{weather_context}"
            )
        )
        final_response_text = res.content

    # Save conversation turn (rolling memory)
    conversation_history[session_id].append({"role": "user", "content": message})
    conversation_history[session_id].append({"role": "assistant", "content": final_response_text})

    # If chat is ending, clear memory for this session after preparing response
    if end_chat:
        conversation_history.pop(session_id, None)

    return {"response": final_response_text, "weather_raw": weather_raw}


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


app.include_router(router, prefix="/api/v1/farmer")
