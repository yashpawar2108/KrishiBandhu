# app.py
import os
import re
import json
import hashlib
import argparse
import asyncio
import logging
from typing import List, Optional, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict, deque

import requests
import pdfplumber
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Pydantic
from pydantic import BaseModel

# LangChain / Vector stuff (ensure installed and compatible versions)
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings as LangchainEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# ChatGroq wrapper (ensure you have this package or replace with your LLM wrapper)
from langchain_groq import ChatGroq

# SentenceTransformers
from sentence_transformers import SentenceTransformer

# -------------------- Config & Logging --------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farmer_bot")

# Keep everything in the SAME directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Artifacts we will COMMIT to Hugging Face
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore.faiss")
VECTORSTORE_META_PATH = os.path.join(BASE_DIR, "vectorstore.pkl")
PROCESSED_TEXT_PATH = os.path.join(BASE_DIR, "processed_text.json")

# Optional local PDFs dir (NOT committed)
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

# Static (for any assets you might want to serve)
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Secrets / env
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
EXPECTED_TOKEN = os.environ.get("FARMER_BOT_TOKEN", "sample_token_123")

# Concurrency - use threadpool for blocking IO
executor = ThreadPoolExecutor(max_workers=6)

# -------------------- FastAPI App --------------------
app = FastAPI(title="Farmer Assistant Chatbot")
router = APIRouter()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# -------------------- Conversation Memory --------------------
MAX_HISTORY = 6
conversation_history: defaultdict = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

def build_conversation_context(session_id: str) -> str:
    history = conversation_history.get(session_id, [])
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in history)

def verify_bearer_token(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    token = authorization.split(" ")[1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return token

class ChatRequest(BaseModel):
    message: str

# -------------------- Embeddings --------------------
class SentenceTransformerEmbeddings(LangchainEmbeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # returns list of vector lists
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

embeddings = SentenceTransformerEmbeddings()

# -------------------- Weather --------------------
async def _sync_get(url: str, params: Dict[str, Any] = None, timeout: int = 10):
    """Helper to run blocking requests in executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: requests.get(url, params=params, timeout=timeout))

async def get_weather_forecast(location: str, days: int = 5) -> str:
    if not OPENWEATHER_API_KEY:
        return "Weather API key not set."
    try:
        geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        geo_params = {"q": f"{location},IN", "limit": 1, "appid": OPENWEATHER_API_KEY}
        geo_resp = await _sync_get(geo_url, params=geo_params)
        geo_data = geo_resp.json()
        if not geo_data:
            return "Could not find location."
        lat, lon = geo_data[0]['lat'], geo_data[0]['lon']

        forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        forecast_params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        forecast_resp = await _sync_get(forecast_url, params=forecast_params)
        forecast_data = forecast_resp.json()

        forecasts = []
        # forecast endpoint returns 3-hourly entries; take up to days * 8 entries
        for entry in forecast_data.get('list', [])[: max(0, days * 8)]:
            date = entry.get('dt_txt', "")
            temp = entry.get('main', {}).get('temp')
            desc = entry.get('weather', [{}])[0].get('description', "")
            if temp is not None and date:
                forecasts.append(f"{date}: {temp:.1f}°C, {desc}")
        return "\n".join(forecasts) if forecasts else "No forecast data available."
    except Exception as e:
        logger.exception("Error fetching weather")
        return f"Weather fetch error: {e}"

# -------------------- Text Extraction + Filtering --------------------
AGRI_KEYWORDS = {
    "agriculture","crop","crops","farmer","farmers","farming","harvest","sowing","irrigation","irrigate",
    "fertilizer","fertiliser","pesticide","insecticide","herbicide","fungicide","soil","soil health",
    "yield","seed","variety","disease","pest","weather","rain","monsoon","drought","humidity","temperature",
    "msp","market","storage","post-harvest","horticulture","kharif","rabi","zayed","livestock","dairy"
}

def normalize_text(t: str) -> str:
    # light normalization for dedup + filtering
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def is_relevant_chunk(text: str) -> bool:
    # Keep chunks that are reasonably sized and contain at least one agri keyword
    if len(text) < 120:      # too small → often headings/garbage
        return False
    if len(text) > 2000:     # too big → likely noisy OCR page dump
        return False
    lower = text.lower()
    return any(k in lower for k in AGRI_KEYWORDS)

def extract_text_from_pdf(filename: str) -> str:
    parts: List[str] = []
    with pdfplumber.open(filename) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                page_text = normalize_text(page_text)
                if page_text:
                    parts.append(page_text)
                else:
                    parts.append(f"[[UNREADABLE PAGE {i}]]")
            except Exception as e:
                logger.warning(f"Error reading page {i} in '{os.path.basename(filename)}': {e}")
                parts.append(f"[[UNREADABLE PAGE {i} - ERROR: {e}]]")
    return "\n".join(parts)

def semantic_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    return splitter.split_text(text)

def dedup_chunks(chunks: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in chunks:
        h = hashlib.md5(c.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out

def filter_top_chunks_per_doc(chunks: List[str], max_keep: int = 400) -> List[str]:
    # Filter by relevance, then keep up to max_keep per document after dedup
    relevant = [c for c in chunks if is_relevant_chunk(c)]
    relevant = dedup_chunks(relevant)
    if len(relevant) > max_keep:
        # crude scoring: presence count of keywords
        def score(c: str) -> int:
            lower = c.lower()
            return sum(lower.count(k) for k in AGRI_KEYWORDS)
        relevant.sort(key=score, reverse=True)
        relevant = relevant[:max_keep]
    return relevant

async def ingest_pdfs_to_artifacts(source_dir: str) -> Tuple[List[Document], FAISS]:
    logger.info(f"Ingesting PDFs from: {source_dir}")
    pdf_files = sorted(
        [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
    )
    if not pdf_files:
        raise RuntimeError("No PDFs found to ingest. Put PDFs in ./documents or pass a directory with --ingest.")

    all_docs: List[Document] = []
    for p in pdf_files:
        logger.info(f"Reading {os.path.basename(p)}")
        raw = extract_text_from_pdf(p)
        chunks = semantic_chunks(raw)
        chunks = filter_top_chunks_per_doc(chunks, max_keep=400)  # tune to control final size
        for ch in chunks:
            all_docs.append(Document(page_content=ch, metadata={"source": os.path.basename(p)}))

    if not all_docs:
        raise RuntimeError("No relevant chunks were produced. Adjust filters/keywords.")

    logger.info(f"Total relevant chunks: {len(all_docs)} (creating FAISS index)")
    loop = asyncio.get_event_loop()
    # FAISS.from_documents is blocking, so run in executor
    vectorstore: FAISS = await loop.run_in_executor(executor, lambda: FAISS.from_documents(all_docs, embeddings))

    # Save artifacts
    try:
        vectorstore.save_local(BASE_DIR)  # writes vectorstore.faiss + vectorstore.pkl to BASE_DIR
    except Exception:
        logger.exception("Could not save vectorstore using save_local; continuing (check library compatibility).")

    with open(PROCESSED_TEXT_PATH, "w", encoding="utf-8") as f:
        json.dump([{"content": d.page_content, "metadata": d.metadata} for d in all_docs], f, ensure_ascii=False, indent=2)

    logger.info(f"Artifacts saved (if supported by your vectorstore implementation).")
    return all_docs, vectorstore

# -------------------- Retrieval Setup --------------------
def create_hybrid_retriever(documents: List[Document], vectorstore: FAISS, k: int = 6):
    # BM25
    bm25 = BM25Retriever.from_documents(documents)
    bm25_k = max(2, k // 3)
    bm25.k = bm25_k
    # Faiss retriever
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": max(1, k - bm25_k)})
    return EnsembleRetriever(retrievers=[bm25, faiss_retriever], weights=[0.3, 0.7])

class ModelRouter:
    def __init__(self):
        # Make sure these env vars exist if you plan to call the models
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
        try:
            # routing rule: longer questions or heavy context -> powerful model
            if len(question.split()) > 15 or context_length > 2000:
                return self.powerful_model
        except Exception:
            # fallback
            return self.fast_model
        return self.fast_model

model_router = ModelRouter()

# -------------------- Runtime State --------------------
processed_docs: List[Document] = []
vectorstore: Optional[FAISS] = None

WEATHER_KEYWORDS = ["weather", "rain", "temperature", "forecast", "irrigate", "frost", "humidity"]
AGRICULTURE_KEYWORDS = ["crop", "farming", "harvest", "fertilizer", "pesticide", "soil", "yield", "agriculture", "irrigation"]

# -------------------- Startup Loader --------------------
@app.on_event("startup")
async def load_artifacts_if_present():
    """On HF Spaces, PDFs won't exist. We only load the prebuilt artifacts if available."""
    global processed_docs, vectorstore
    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(VECTORSTORE_META_PATH) and os.path.exists(PROCESSED_TEXT_PATH):
        logger.info("Loading preprocessed chunks & FAISS index from repo artifacts.")
        try:
            with open(PROCESSED_TEXT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            processed_docs = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]
            # load_local may be blocking - run in executor
            loop = asyncio.get_event_loop()
            vectorstore = await loop.run_in_executor(executor, lambda: FAISS.load_local(BASE_DIR, embeddings, allow_dangerous_deserialization=True))
            logger.info(f"Loaded {len(processed_docs)} chunks.")
        except Exception as e:
            logger.exception("Failed to load artifacts; start with empty index or run ingestion locally.")
            processed_docs = []
            vectorstore = None
    else:
        logger.warning("Artifacts not found. If running locally, run: python app.py --ingest ./documents")

# -------------------- Chat Endpoint --------------------
@router.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    location: Optional[str] = Query(None),
    end_chat: Optional[bool] = Query(False),
    session_token: str = Depends(verify_bearer_token),
):
    global processed_docs, vectorstore

    message = req.message.strip()
    weather_context = ""
    weather_raw = None
    current_datetime = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    # session_token contains verified token string
    session_id = session_token or (EXPECTED_TOKEN if EXPECTED_TOKEN != "sample_token_123" else "default_user")
    history_str = build_conversation_context(session_id)

    # Weather
    if any(kw in message.lower() for kw in WEATHER_KEYWORDS):
        if not location:
            return {
                "response": "🌱 **KrishiBandhu Advice** 🌱\n\nPlease provide your location or postal code to get accurate weather details.",
                "weather_raw": None
            }
        weather_raw = await get_weather_forecast(location)
        weather_context = f"Weather forecast for {location} as of {current_datetime}:\n{weather_raw}"

    is_agri_query = any(kw in message.lower() for kw in AGRICULTURE_KEYWORDS)

    try:
        if is_agri_query and processed_docs and vectorstore:
            # build retriever and run retrieval + QA chain
            docs = processed_docs.copy()
            if weather_context:
                docs.append(Document(page_content=weather_context, metadata={"source": "weather_api"}))
            retriever = create_hybrid_retriever(docs, vectorstore)
            llm = model_router.route_model(message, sum(len(d.page_content) for d in docs))

            # Create prompt template
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

            # RetrievalQA expects a retriever and llm; passing prompt via chain_type_kwargs if supported
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},  # library variations exist: try this first
                return_source_documents=True
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, lambda: qa_chain.invoke({"query": message}))
            # result may be dict with 'result' key or string-like - handle both
            if isinstance(result, dict) and "result" in result:
                final_response_text = result.get("result", "")
            elif isinstance(result, str):
                final_response_text = result
            elif hasattr(result, "content"):
                final_response_text = getattr(result, "content")
            else:
                final_response_text = str(result)
        else:
            # Non-agri or no vectorstore available: direct LLM response
            llm = model_router.route_model(message)
            prompt_text = f"[Current Date & Time: {current_datetime}]\n{history_str}\nUser: {message}\n{weather_context}"
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(executor, lambda: llm.invoke(prompt_text))
            # llm.invoke may return object or string
            if isinstance(res, dict) and "content" in res:
                final_response_text = res.get("content", "")
            elif isinstance(res, str):
                final_response_text = res
            elif hasattr(res, "content"):
                final_response_text = getattr(res, "content")
            else:
                final_response_text = str(res)
    except Exception as e:
        logger.exception("Error during LLM / retrieval processing")
        final_response_text = f"Sorry — an internal error occurred while preparing the response: {e}"

    # Save to conversation history
    conversation_history[session_id].append({"role": "user", "content": message})
    conversation_history[session_id].append({"role": "assistant", "content": final_response_text})

    if end_chat:
        conversation_history.pop(session_id, None)

    return {"response": final_response_text, "weather_raw": weather_raw}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(router, prefix="/api/v1/farmer")

# -------------------- CLI (Local Ingestion) --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farmer Assistant App")
    parser.add_argument("--ingest", type=str, default=None,
                        help="Path to a folder containing PDFs. Produces processed_text.json + vectorstore.* and exits.")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI app with Uvicorn.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if args.ingest:
        # Local preprocessing step (do this once on your machine)
        os.makedirs(args.ingest, exist_ok=True)
        asyncio.run(ingest_pdfs_to_artifacts(args.ingest))
        if not args.serve:
            # Just build artifacts and exit; you will git add + push these files (not PDFs).
            exit(0)

    if args.serve:
        import uvicorn
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
