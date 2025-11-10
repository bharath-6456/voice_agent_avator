import argparse,os,re,logging,base64,io
import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime
import uvicorn
import uuid
from agent_tools import run_bot2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-base")
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request,WebSocket, UploadFile, File, HTTPException
from fastapi.responses import FileResponse,JSONResponse,Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import pymupdf as fitz  # PyMuPDF
from PIL import Image

# RAG dependencies (mirroring RAG_UI.py, no Streamlit)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

# Load environment variables
load_dotenv(override=True)

# -------- RAG (FAISS/LangChain) globals & helpers (no Streamlit) --------
# Stores last processed PDF, vector index and chain per session
rag_states: Dict[str, dict] = {}

# Lock to ensure only one RAG query is processed at a time (global)
rag_lock = asyncio.Lock()

# Last RAG result cache for UI/voice assistant to retrieve per session
last_rag_results: Dict[str, dict] = {}

def split_into_factual_points(answer_text: str):
    sents = re.split(r'(?<=[\.?\!])\s+', (answer_text or "").strip())
    return [s.strip() for s in sents if s.strip()]

def keyword_overlap(a: str, b: str):
    wa = set(re.findall(r'\w+', (a or "").lower()))
    wb = set(re.findall(r'\w+', (b or "").lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)

def semantic_sim_score(point: str, doc_text: str, embedder):
    import math
    v_point = embedder.embed_query(point)
    v_doc = embedder.embed_documents([doc_text])[0]
    dot = sum(a*b for a,b in zip(v_point, v_doc))
    norm_p = math.sqrt(sum(a*a for a in v_point))
    norm_d = math.sqrt(sum(a*a for a in v_doc))
    if norm_p == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_p * norm_d)

def citefix_correct_answer(raw_answer: str, candidate_docs, embedder, lam=0.8):
    points = split_into_factual_points(raw_answer)
    corrected_points = []
    pages_used = []
    for pt in points:
        found = re.findall(r'page\s*(\d+)', pt, flags=re.IGNORECASE)
        Ci = len(found) if found else 1
        scores = []
        for doc in candidate_docs or []:
            kw = keyword_overlap(pt, doc.page_content)
            sem = semantic_sim_score(pt, doc.page_content, embedder)
            score = lam * kw + (1 - lam) * sem
            scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        chosen = [d for s,d in scores[:Ci]]
        if chosen:
            primary = chosen[0]
            page_meta = primary.metadata.get("page", 0)
            page_one_based = page_meta + 1
            pages_used.append(page_one_based)
            corrected_pt = f"{pt} (Source: Page {page_one_based})"
        else:
            corrected_pt = pt
        corrected_points.append(corrected_pt)
    final = " ".join(corrected_points)
    return final, sorted(set(pages_used))

def build_rag_index(pdf_path: str, chunk_size=1200, chunk_overlap=250):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for i, p in enumerate(pages):
        p.metadata["page"] = p.metadata.get("page", i)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    # embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./models")
    vectorstore = FAISS.from_documents(chunks, embedder)
    # Use MMR retriever to improve diversity and relevance, fetch more then select top-k
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 60, "lambda_mult": 0.5})
    # Lower temperature for deterministic answers grounded in context, stronger model for better grounding
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    prompt_template = """
You are a concise, friendly assistant that helps answer questions based on provided document excerpts. Use the context to provide accurate answers.

Context: {context}

Question: {question}

Then, answer the user's question using ONLY the provided context, unless the information is missing.

Instructions:
- If the answer IS found in the context, write a clear, complete answer directly addressing the question.
- Prefer specifics over general statements.

"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return vectorstore, embedder, rag_chain, len(pages), len(chunks)

def render_page_image(pdf_path: str, page_number: int):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}



# Configure ICE servers using your Xirsys credentials
ice_servers = [
    IceServer(
        urls="stun:34.14.202.215:3479"
    ),
    IceServer(
        urls="turn:34.14.202.215:3479?transport=udp",
        username="test",
        credential="test123"
    ),
    IceServer(
        urls="turn:34.14.202.215:3479?transport=tcp",
        username="test",
        credential="test123"
    )
]

@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")
    session_id = request.get("session_id")
    if not session_id:
        logger.warning(f"No session_id provided for pc_id: {pc_id}. RAG will not work.")
        # You might want to raise an HTTPException here if RAG is required
        # raise HTTPException(status_code=400, detail="Missing session_id")

    logger.info(f"Linking pc_id {pc_id} to session_id {session_id}")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(sdp=request["sdp"], type=request["type"])
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot2, pipecat_connection, session_id)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer

@app.post("/pdf/upload")
async def pdf_upload(file: UploadFile = File(...)):
    try:
        # Generate a unique session ID for this upload
        session_id = str(uuid.uuid4())
        os.makedirs("temp_pdf_files", exist_ok=True)
        pdf_path = os.path.join("temp_pdf_files", f"{session_id}_{file.filename}")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        # Initialize RAG state for this session
        rag_states[session_id] = {
            "pdf_path": pdf_path,
            "vectorstore": None,
            "embedder": None,
            "rag_chain": None,
            "num_pages": 0,
            "num_chunks": 0,
        }
        logger.success(f"📄 [PDF] Uploaded and saved to {pdf_path} for session {session_id}")
        return {"pdf_path": pdf_path, "filename": file.filename, "session_id": session_id}
    except Exception:
        logger.exception("[PDF] Upload failed")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.post("/pdf/process")
async def pdf_process(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    if not session_id or session_id not in rag_states:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id")
    pdf_path = body.get("pdf_path") or rag_states[session_id].get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail="PDF path not set or file missing")
    try:
        vectorstore, embedder, rag_chain, num_pages, num_chunks = build_rag_index(pdf_path)
        rag_states[session_id].update({
            "vectorstore": vectorstore,
            "embedder": embedder,
            "rag_chain": rag_chain,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
        })
        logger.success(f"🧭 [RAG] Index built for session {session_id}: pages={num_pages}, chunks={num_chunks}")
        return {"pages": num_pages, "chunks": num_chunks}
    except Exception:
        logger.exception("[RAG] Failed to build index")
        raise HTTPException(status_code=500, detail="Failed to process PDF")

@app.post("/rag/query")
async def rag_query(request: Request):
    async with rag_lock:
        body = await request.json()
        session_id = body.get("session_id")
        if not session_id or session_id not in rag_states:
            raise HTTPException(status_code=400, detail="Invalid or missing session_id")
        if not rag_states[session_id].get("rag_chain") or not rag_states[session_id].get("embedder"):
            raise HTTPException(status_code=400, detail="RAG index not built. Upload and process a PDF first.")
        query = body.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Missing 'query'")
        try:
            rag_chain = rag_states[session_id]["rag_chain"]
            response = rag_chain.invoke({"query": query})
            raw_answer = response.get("result") or response.get("answer") or ""
            candidate_docs = response.get("source_documents")
            if not candidate_docs and rag_states[session_id].get("vectorstore"):
                try:
                    fallback = rag_states[session_id]["vectorstore"].as_retriever(search_kwargs={"k": 3})
                    candidate_docs = fallback.get_relevant_documents(query)
                except Exception:
                    candidate_docs = []
            corrected_answer, pages_used = citefix_correct_answer(raw_answer, candidate_docs, rag_states[session_id]["embedder"], lam=0.5)
            # Relevance guard: if nothing cited and retrieved docs are weak vs query, report not available
            try:
                sims = []
                for d in (candidate_docs or [])[:6]:
                    sims.append(semantic_sim_score(query, d.page_content or "", rag_states[session_id]["embedder"]))
                max_sim = max(sims) if sims else 0.0
                if (not pages_used) and max_sim < 0.35:
                    corrected_answer = "The information is not available in the document."
            except Exception:
                pass
            # pack candidate snippets
            candidates = []
            for d in (candidate_docs or [])[:6]:
                p = (d.metadata.get("page", 0) + 1)
                candidates.append({"page": p, "snippet": (d.page_content or "")[:200]})
            # save last result for UI/voice display per session
            try:
                from datetime import datetime as _dt
                last_rag_results[session_id] = {
                    "answer": corrected_answer,
                    "pages": pages_used,
                    "timestamp": _dt.utcnow().isoformat() + "Z",
                }
            except Exception:
                pass
            return {"answer": corrected_answer, "pages": pages_used, "candidates": candidates}
        except Exception:
            logger.exception("[RAG] Query failed")
            raise HTTPException(status_code=500, detail="Query failed")

@app.get("/rag/last/{session_id}")
async def get_last_rag(session_id: str):
    return last_rag_results.get(session_id, {})

@app.get("/pdf/page/{session_id}/{page}")
async def pdf_page(session_id: str, page: int):
    if session_id not in rag_states:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    pdf_path = rag_states[session_id].get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail="No PDF loaded")
    if page < 1 or (rag_states[session_id].get("num_pages") and page > rag_states[session_id]["num_pages"]):
        raise HTTPException(status_code=400, detail="Page out of range")
    try:
        buf = render_page_image(pdf_path, page)
        return StreamingResponse(buf, media_type="image/png")
    except Exception:
        logger.exception("[PDF] Render failed")
        raise HTTPException(status_code=500, detail="Render failed")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../index.html"))

@app.get("/login")
async def serve_login():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../login.html"))

@app.get("/options")
async def serve_options():
    return FileResponse(os.path.join(os.path.dirname(__file__), "../options.html"))




def get_embedding(text: str):
    logger.debug(f"🔢 [EMBEDDING] Generating embedding for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    embedding = model.encode(text).tolist()
    logger.debug(f"✅ [EMBEDDING] Generated embedding with {len(embedding)} dimensions")
    return embedding

def clean_answer_for_voice(answer: str) -> str:
    """
    Clean FAQ answer for voice output by removing special characters
    and making it more conversational
    """
    # Remove common symbols and replace with voice-friendly alternatives
    replacements = {
        '&': ' and ',
        '%': ' percent ',
        '#': ' hashtag ',
        '@': ' at the rate ',
        '*': '',
        '•': '',
        '₹': ' rupees ',
        '$': ' dollars ',
        '€': ' euros ',
        '£': ' pounds ',
    }
    
    cleaned = answer
    for symbol, replacement in replacements.items():
        cleaned = cleaned.replace(symbol, replacement)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    # Limit length for voice output (roughly 15-20 seconds of speech)
    words = cleaned.split()
    if len(words) > 50:  # Roughly 15-20 seconds at normal speaking pace
        cleaned = ' '.join(words[:50]) + '...'
    
    return cleaned





    
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP server (default: 8000)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)