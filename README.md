# 🌾 KrishiBandhu – Farmer's Friend

**KrishiBandhu** is a **multi-lingual, multi-modal AI assistant** designed to support Indian farmers with **real-time weather forecasts, soil/crop report analysis, fertilizer & pesticide recommendations, and conversational Q&A**.  

It combines a **FastAPI backend (AI + retrieval pipeline)** with a **responsive frontend (HTML/JS + Chart.js)**, enabling farmers to interact via **text, voice, or document uploads** in their preferred language.  

---

## 🚀 Features

### 👨‍🌾 Farmer-Facing
- Real-time **weather forecasts** with irrigation advice.  
- Ask farming-related queries in **local languages** (Hindi, Bengali, Marathi, Tamil, etc.).  
- **Upload soil/crop reports** and get instant analysis.  
- **Voice-enabled chat** – speak with the bot, hear responses back.  
- Simple, mobile-friendly dashboard with charts and quick action buttons.  

### ⚙️ Technical Highlights
- **Backend:**  
  - FastAPI framework with secure Bearer Token authentication.  
  - Groq **LLaMA-3 (8B/70B)** LLMs with heuristic routing.  
  - **Hybrid Retrieval:** FAISS (semantic vectors) + BM25 (keyword).  
  - PDF ingestion with `pdfplumber`, embeddings via SentenceTransformers.  
  - **Weather integration** via OpenWeather API.  
  - **Multi-lingual support** with Google Translator + gTTS (TTS) + SpeechRecognition (STT).  
  - Conversation memory: maintains last 6 turns per user.  

- **Frontend:**  
  - HTML/JS single-page app with **Chart.js** for weather visualization.  
  - Chat UI with timestamps, typing indicators, and markdown support.  
  - **Voice recording** with MediaRecorder API.  
  - File upload support for soil/crop reports.  
  - Responsive design with farm-themed styling + toast notifications.  

- **Performance & Infra:**  
  - Vectorstore caching in `/tmp/farmer_cache`.  
  - Concurrent task execution via `ThreadPoolExecutor`.  
  - Structured project layout with `/documents`, `/static/tts`, `/frontend`.  

---
## 📂 Project Structure

```bash
farmer-assistant/
├── main.py               # FastAPI backend (core API + logic)
├── documents/            # Uploaded soil/crop reports (PDFs)
├── static/
│   └── tts/              # Generated audio responses (TTS files)
│   └── index.html        # Farmer-facing web app (chat + charts + voice)
├── tmp/
│   └── farmer_cache/     # FAISS + BM25 vectorstore cache
    
```
---

## 🔑 API Endpoints

- **Health Check:**  
  `GET /api/v1/farmer/health`

- **Chat:**  
  `POST /api/v1/farmer/chat`  
  Body: `{ message, output_language?, return_audio? }`

- **Voice:**  
  `POST /api/v1/farmer/voice`  
  Body: audio file  

- **Upload Soil Report:**  
  `POST /api/v1/farmer/upload-soil-report`  
  Body: PDF file  

---

## 🧑‍💻 Tech Stack

- **Backend:** FastAPI, LangChain, Groq LLMs, FAISS, BM25, pdfplumber, SentenceTransformers  
- **APIs/Services:** OpenWeather, Google Translator, gTTS, SpeechRecognition  
- **Frontend:** HTML, JavaScript, Chart.js, MediaRecorder API, CSS  
- **Infra/Tools:** ThreadPoolExecutor, caching, modular API design  

---

## ⚙️ How It Works (Flow)

1. Farmer sends a query (text/voice/upload).  
2. Backend detects language, translates to English.  
3. Hybrid retriever pulls context from soil reports + weather.  
4. Groq LLM generates an answer (LLaMA-3, routed by complexity).  
5. Response translated back to farmer’s preferred language.  
6. Bot reply sent as **text + optional TTS audio**.  
7. Frontend displays response with charts/sources/audio.  

---

## 🌐 Future Enhancements
- Offline mobile app version.  
- Crop disease image recognition.  
- SMS/IVR-based farmer support.  
- Integration with government agriculture databases.  

---

## 📜 License
MIT License – free to use and modify.  

---

## 🤝 Contributing
Pull requests and suggestions are welcome!  

---

## 🙌 Acknowledgements
- Farmers & agri-community for real-world insights.  
- OpenWeather API for weather forecasts.  
- Google services (STT, TTS, Translation) for multi-language support.  
- Groq for powering fast LLaMA-3 inference.  
