# main.py
import os
import re
import io
import uuid
import shutil
import tempfile
import asyncio
import pdfplumber
import logging
import requests
import json
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict, deque

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Language detection & translation
from langdetect import detect
from deep_translator import GoogleTranslator

# Speech / TTS
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TTS_DIR = os.path.join(STATIC_DIR, "tts")
os.makedirs(TTS_DIR, exist_ok=True)
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

DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
VECTORSTORE_PATH = os.path.join(CACHE_DIR, "vectorstore.faiss")
DOCS_METADATA_PATH = os.path.join(CACHE_DIR, "docs_metadata.json")
processed_docs: List[Document] = []
vectorstore = None

# Conversation memory (rolling)
MAX_HISTORY = 6
conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))

def build_conversation_context(session_id: str) -> str:
    history = conversation_history.get(session_id, [])
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in history)

# Authentication
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
    output_language: Optional[str] = 'en'
    return_audio: Optional[bool] = False

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
    try:
        with pdfplumber.open(filename) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                except Exception as e:
                    logger.error(f"❌ Error reading page {i} in '{os.path.basename(filename)}': {e}")
    except Exception as e:
        logger.error(f"❌ Failed to open or process PDF '{os.path.basename(filename)}': {e}")
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
    vs = await loop.run_in_executor(
        executor,
        lambda: FAISS.from_documents(documents, embeddings)
    )
    return vs, documents

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
            temperature=0.2, max_tokens=1024
        )
        self.powerful_model = ChatGroq(
            model=os.environ.get("GROQ_POWER_MODEL", "llama3-70b-8192"),
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.2, max_tokens=2048
        )

    def route_model(self, question: str, context_length: int = 0):
        if len(question.split()) > 15 or context_length > 2000:
            return self.powerful_model
        return self.fast_model

model_router = ModelRouter()

# Weather & soil helper
async def get_weather_forecast(location: str) -> Optional[str]:
    if not OPENWEATHER_API_KEY:
        return "Weather API key not set."
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location},IN&limit=1&appid={OPENWEATHER_API_KEY}"
    try:
        geo_resp = await asyncio.get_event_loop().run_in_executor(executor, lambda: requests.get(geo_url))
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
        if not geo_data:
            return f"Could not find location '{location}'."
        lat, lon = geo_data[0]['lat'], geo_data[0]['lon']
    except requests.RequestException as e:
        logger.error(f"Geo API request failed: {e}")
        return "Could not fetch location data."
    except (KeyError, IndexError):
        return "Invalid location data received."
        
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        forecast_resp = await asyncio.get_event_loop().run_in_executor(executor, lambda: requests.get(forecast_url))
        forecast_resp.raise_for_status()
        forecast_data = forecast_resp.json()
    except requests.RequestException as e:
        logger.error(f"Forecast API request failed: {e}")
        return "Could not fetch forecast data."

    forecasts = []
    for entry in forecast_data.get('list', [])[: 5 * 8]: # 5 days, 8 entries per day
        date = entry.get('dt_txt', 'unknown')
        temp = entry.get('main', {}).get('temp')
        humidity = entry.get('main', {}).get('humidity')
        wind = entry.get('wind', {}).get('speed')
        rain = entry.get('rain', {}).get('3h', 0)
        desc = entry.get('weather', [{}])[0].get('description', '')
        soil_tip = _soil_advice(humidity, rain)
        forecasts.append(
            f"{date}: {temp:.1f}°C, {desc}, Humidity: {humidity}%, Wind: {wind} m/s, Rain(3h): {rain} mm | {soil_tip}"
        )
    return "\n".join(forecasts)

def _soil_advice(humidity: Optional[float], rain_3h: Optional[float]) -> str:
    if rain_3h and float(rain_3h) > 5:
        return "Heavy short-term rain expected—consider drainage and avoid immediate pesticide spraying."
    if humidity is None:
        return "No soil moisture estimate available."
    h = float(humidity)
    if h < 40: return "Low humidity—soil may dry quickly; irrigation recommended."
    if h < 70: return "Moderate humidity—monitor soil; irrigation optional based on crop."
    return "High humidity—watch for fungal diseases; delay irrigation if soil is wet."

# Simple crop calendar & Keywords
CROP_CALENDAR = {
    1: ["wheat (rabi)", "potato", "mustard"], 2: ["wheat (maturing)", "mustard"],
    3: ["sugarcane planting", "cotton sowing"], 4: ["maize sowing", "groundnut sowing"],
    5: ["paddy transplanting in irrigated areas"], 6: ["paddy (kharif) active growth"],
    7: ["paddy (kharif) - high water demand"], 8: ["paddy (kharif) - monitor for pests"],
    9: ["paddy harvesting", "rabi sowing prep"], 10: ["sowing wheat (rabi)"],
    11: ["wheat establishment"], 12: ["wheat growth; prepare for irrigation"]
}
WEATHER_KEYWORDS = ["weather", "rain", "temperature", "forecast", "irrigate", "frost", "humidity"]
AGRICULTURE_KEYWORDS = ["crop", "farming", "harvest", "fertilizer", "pesticide", "soil", "yield", "agriculture", "irrigation"]
FERTILIZER_KEYWORDS = ["fertilizer", "urea", "dap", "npk", "manure"]
PESTICIDE_KEYWORDS = ["pesticide", "spray", "insect", "bollworm", "aphid", "fungus", "blight"]

# Translation helpers
def detect_language(text: str) -> str:
    try: return detect(text)
    except Exception: return "en"

def translate_to_english(text: str, src_lang: str) -> str:
    if not text or src_lang == 'en': return text
    try: return GoogleTranslator(source=src_lang, target='en').translate(text)
    except Exception as e:
        logger.warning(f"Translation to English failed: {e}"); return text

def translate_from_english_if_requested(text_en: str, target_lang: Optional[str]) -> str:
    if not text_en or not target_lang or target_lang == 'en': return text_en
    try: return GoogleTranslator(source='en', target=target_lang).translate(text_en)
    except Exception as e:
        logger.warning(f"Translation from English failed: {e}"); return text_en

# Fertilizer / pesticide quick advice
def fertilizer_advice(crop: Optional[str] = None) -> str:
    if not crop: return "General advice: test soil, follow recommended NPK doses for your crop and growth stage."
    crop = crop.lower()
    if 'wheat' in crop: return "Wheat: Apply basal N at sowing, top-dress N at tillering. Follow soil test recommendations."
    if 'paddy' in crop or 'rice' in crop: return "Paddy: Use split applications of nitrogen; maintain puddled fields; avoid overuse."
    if 'maize' in crop or 'corn' in crop: return "Maize: Balanced NPK; consider starter fertilizer at planting."
    return "Use crop-specific guidelines from local agricultural extension."

def pesticide_advice(issue: Optional[str] = None) -> str:
    if not issue: return "If pests are visible, identify pest and use targeted pesticide. Avoid spraying during flowering and windy conditions."
    issue = issue.lower()
    if 'bollworm' in issue: return "Bollworm: Use pheromone traps and recommended insecticides; consider biocontrol agents."
    if 'aphid' in issue: return "Aphid: Encourage beneficial insects; use selective insecticides if infestation is high."
    if 'fungus' in issue or 'blight' in issue: return "Fungal disease: Use timely fungicide sprays and avoid prolonged leaf wetness."
    return "Consult local agri-extension for pesticide selection and dose."

# Speech & TTS Utilities
def convert_audio_to_wav_bytes(upload_file: UploadFile) -> Optional[bytes]:
    try:
        raw = upload_file.file.read()
        upload_file.file.seek(0)
        audio = AudioSegment.from_file(io.BytesIO(raw))
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Audio conversion failed: {e}"); return None

def transcribe_audio_bytes_to_text(wav_bytes: bytes) -> str:
    r = sr.Recognizer()
    audio_data = sr.AudioData(wav_bytes, sample_rate=16000, sample_width=2)
    try:
        return r.recognize_google(audio_data)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        logger.warning(f"Speech recognition request failed: {e}")
        return ""


def generate_tts_file(text: str, lang_code: str = "en") -> str:
    if not text: return ""
    try:
        filename = f"{uuid.uuid4().hex}.mp3"
        out_path = os.path.join(TTS_DIR, filename)
        tts = gTTS(text=text, lang=lang_code)
        tts.save(out_path)
        return f"/static/tts/{filename}"
    except Exception as e:
        logger.warning(f"TTS generation failed for lang '{lang_code}': {e}"); return ""

# Startup: load/process documents
@app.on_event("startup")
async def load_or_process_documents():
    global processed_docs, vectorstore
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    pdf_files = sorted([os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")])
    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCS_METADATA_PATH):
        try:
            with open(DOCS_METADATA_PATH, "r") as f: metadata = json.load(f)
            if metadata.get("files", []) == pdf_files:
                logger.info("✅ Using cached vectorstore.")
                vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
                processed_docs = [Document(page_content=d["content"], metadata=d.get("metadata", {})) for d in metadata.get("docs", [])]
                return
        except Exception as e:
            logger.warning(f"Failed to load cached metadata: {e}")

    logger.info("📄 Processing documents...")
    all_docs, all_vs = [], []
    for p in pdf_files:
        vs, docs = await get_or_create_document_pipeline(p)
        if docs:
            for d in docs: d.metadata["source"] = p
            all_docs.extend(docs)
            all_vs.append(vs)
    if not all_vs:
        logger.info("❌ No valid PDFs found to process.")
        return
    
    combined_vs = all_vs[0]
    for vs in all_vs[1:]: combined_vs.merge_from(vs)
    combined_vs.save_local(VECTORSTORE_PATH)
    processed_docs = all_docs
    vectorstore = combined_vs
    with open(DOCS_METADATA_PATH, "w") as f:
        json.dump({
            "files": pdf_files,
            "docs": [{"content": d.page_content, "metadata": d.metadata} for d in all_docs]
        }, f)
    logger.info("✅ Documents processed and cached.")

# --------- REFACTORED: Core Chat Processing Logic ----------
async def _process_chat_logic(
    message_en: str,
    session_id: str,
    location: Optional[str],
    return_sources: bool = False
) -> Dict[str, Any]:
    """Central logic for handling chat, shared by text and voice endpoints."""
    current_datetime = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    history_str = build_conversation_context(session_id)
    
    weather_context, weather_raw = "", None
    if any(re.search(rf"\b{kw}\b", message_en.lower()) for kw in WEATHER_KEYWORDS):
        if not location:
            return {"final_response_en": "Please provide your location (village / city / district).", "weather_raw": None, "considered_docs": []}
        weather_raw = await get_weather_forecast(location)
        weather_context = f"Weather forecast for {location} as of {current_datetime}:\n{weather_raw}"

    is_agriculture_query = any(re.search(rf"\b{kw}\b", message_en.lower()) for kw in AGRICULTURE_KEYWORDS)
    considered_docs, final_response_text_en = [], ""

    if is_agriculture_query and processed_docs and vectorstore:
        docs = processed_docs.copy()
        if weather_context: docs.append(Document(page_content=weather_context, metadata={"source": "weather_api"}))
        retriever = create_hybrid_retriever(docs, vectorstore)
        llm = model_router.route_model(message_en, sum(len(d.page_content) for d in docs))
        prompt = PromptTemplate(
            template=(
                "You are **KrishiBandhu**, an AI assistant for Indian farmers.\n"
                "Today is {date_time}.\nConversation so far:\n{history}\n"
                "**Context:**\n{context}\n\n**Question:** {question}\n\n**Answer:**"
            ),
            input_variables=["context", "question", "date_time", "history"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever,
            chain_type_kwargs={"prompt": prompt.partial(date_time=current_datetime, history=history_str)},
            return_source_documents=return_sources
        )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, lambda: qa_chain.invoke({"query": message_en}))
        final_response_text_en = result.get("result", "")
        if return_sources:
            considered_docs = [{"content": doc.page_content, "metadata": doc.metadata} for doc in result.get("source_documents", [])]
    else:
        llm = model_router.route_model(message_en)
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(executor, lambda: llm.invoke(f"[{current_datetime}]\n{history_str}\nUser: {message_en}\n{weather_context}"))
        final_response_text_en = getattr(res, 'content', str(res))

    # Structured heuristics
    if any(re.search(rf"\b{kw}\b", message_en.lower()) for kw in FERTILIZER_KEYWORDS):
        crop_guess = next((c for c in ['wheat', 'paddy', 'rice', 'maize'] if c in message_en.lower()), None)
        final_response_text_en += "\n\n" + fertilizer_advice(crop_guess)
    
    if any(re.search(rf"\b{kw}\b", message_en.lower()) for kw in PESTICIDE_KEYWORDS):
        issue_guess = next((i for i in ['bollworm', 'aphid', 'blight', 'fungus'] if i in message_en.lower()), None)
        final_response_text_en += "\n\n" + pesticide_advice(issue_guess)

    if re.search(r"\b(calendar|what to sow|plant|sowing|harvest)\b", message_en.lower()):
        month = datetime.now().month
        cal = CROP_CALENDAR.get(month, [])
        if cal: final_response_text_en += "\n\n" + "Crop suggestions for this month: " + ", ".join(cal)
        
    return {
        "final_response_en": final_response_text_en.strip(),
        "weather_raw": weather_raw,
        "considered_docs": considered_docs
    }

# --------- Core Chat Endpoint (text) ----------
@router.post("/chat")
async def chat_endpoint(
    req: ChatRequest,
    location: Optional[str] = Query(None),
    end_chat: Optional[bool] = Query(False),
    session_token: str = Depends(verify_bearer_token),
):
    raw_message = (req.message or "").strip()
    if not raw_message: raise HTTPException(status_code=400, detail="Empty message")

    input_lang = detect_language(raw_message)
    message_en = translate_to_english(raw_message, input_lang)
    session_id = session_token or "default_user"

    result = await _process_chat_logic(message_en, session_id, location, return_sources=True)
    
    response_for_user = translate_from_english_if_requested(result["final_response_en"], req.output_language)
    
    tts_url = ""
    if req.return_audio:
        tts_lang = req.output_language or "en"
        tts_url = generate_tts_file(response_for_user, lang_code=tts_lang)

    conversation_history[session_id].append({"role": "user", "content": raw_message})
    conversation_history[session_id].append({"role": "assistant", "content": response_for_user})
    if end_chat: conversation_history.pop(session_id, None)

    return {
        "response": response_for_user,
        "response_language": req.output_language or "en",
        "weather_raw": result["weather_raw"],
        "tts_url": tts_url,
        "considered_documents": [
            os.path.basename(doc["metadata"].get("source", ""))
            for doc in result["considered_docs"] if "metadata" in doc and doc["metadata"].get("source")
        ]
    }

# --------- Voice endpoint ----------
@router.post("/voice")
async def voice_endpoint(
    file: UploadFile = File(...),
    output_language: Optional[str] = Query('en'),
    return_audio: Optional[bool] = Query(True),
    location: Optional[str] = Query(None),
    session_token: str = Depends(verify_bearer_token),
):
    wav_bytes = convert_audio_to_wav_bytes(file)
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Unsupported audio format or conversion failed. Ensure ffmpeg is installed.")
    
    user_text = transcribe_audio_bytes_to_text(wav_bytes)
    if not user_text:
        raise HTTPException(status_code=400, detail="Could not transcribe audio or speech not understood.")

    input_lang = detect_language(user_text)
    message_en = translate_to_english(user_text, input_lang)
    session_id = session_token or "default_user"

    result = await _process_chat_logic(message_en, session_id, location, return_sources=False)
    
    response_for_user = translate_from_english_if_requested(result["final_response_en"], output_language)
    
    tts_url = ""
    if return_audio:
        tts_lang = output_language or "en"
        tts_url = generate_tts_file(response_for_user, lang_code=tts_lang)
    
    conversation_history[session_id].append({"role": "user", "content": user_text})
    conversation_history[session_id].append({"role": "assistant", "content": response_for_user})

    return {
        "transcript": user_text,
        "input_language": input_lang,
        "response": response_for_user,
        "response_language": output_language or "en",
        "tts_url": tts_url,
        "weather_raw": result["weather_raw"]
    }

# Simple upload endpoint
@router.post("/upload-soil-report")
async def upload_soil_report(file: UploadFile = File(...), session_token: str = Depends(verify_bearer_token)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    safe_filename = os.path.basename(file.filename)
    dest = os.path.join(DOCUMENTS_DIR, safe_filename)
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # In a real app, you might trigger a background reprocessing task here.
    # For now, we clear the cache so it will be reprocessed on next startup.
    if os.path.exists(VECTORSTORE_PATH): os.remove(VECTORSTORE_PATH)
    if os.path.exists(DOCS_METADATA_PATH): os.remove(DOCS_METADATA_PATH)
    
    return {"status": "uploaded", "filename": safe_filename, "note": "File saved. Vector store cache cleared. Please restart the server to re-index documents."}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(router, prefix="/api/v1/farmer")

if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)