#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse,os,re,logging,sqlite3,base64
import asyncio
import sys
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict
from datetime import datetime
import uvicorn
from agent_tools import run_bot2
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request,WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sentence_transformers import SentenceTransformer


from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
#from shared_state import live_transcriptions
# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
LOG_DIR = "logs"
#api_key = os.getenv("PINECONE_API_KEY")
#environment = "us-east-1"
#index_name = "travelport-smartpoint"

#pc = Pinecone(api_key=api_key)
#index = pc.Index(index_name)

# Embedding model
#model = SentenceTransformer("intfloat/e5-base")

#FLIGHT_DATA = pd.read_csv("flight_status_data.csv")

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}



# Configure ICE servers using your Xirsys credentials
#ice_servers = [
    # STUN server from Xirsys
#    IceServer(
#        urls="stun:bn-turn1.xirsys.com",
#    ),
    # TURN servers from Xirsys (UDP, TCP, TLS over TCP)
    # The pipecat IceServer class expects a single URL string for 'urls'.
    # So, we'll create separate IceServer objects for each distinct URL you want to use.
    # It's good to include UDP, TCP, and TLS options for maximum compatibility.

    # UDP TURN servers
#    IceServer(
#        urls="turn:bn-turn1.xirsys.com:80?transport=udp",
#        username=XIRSYS_USERNAME,
#        credential=XIRSYS_CREDENTIAL,
#    ),
#    IceServer(
#        urls="turn:bn-turn1.xirsys.com:3478?transport=udp",
#        username=XIRSYS_USERNAME,
#        credential=XIRSYS_CREDENTIAL,
#    ),
    # TCP TURN servers
#    IceServer(
#        urls="turn:bn-turn1.xirsys.com:80?transport=tcp",
#        username=XIRSYS_USERNAME,
#        credential=XIRSYS_CREDENTIAL,
#    ),
#    IceServer(
#        urls="turn:bn-turn1.xirsys.com:3478?transport=tcp",
#        username=XIRSYS_USERNAME,
#        credential=XIRSYS_CREDENTIAL,
#    ),
#    # TLS over TCP TURN servers (often the most reliable for strict firewalls)
#    IceServer(
#        urls="turns:bn-turn1.xirsys.com:443?transport=tcp",
#        username=XIRSYS_USERNAME,
#        credential=XIRSYS_CREDENTIAL,
#    ),
#    IceServer(
#        urls="turns:bn-turn1.xirsys.com:5349?transport=tcp",
#        username=XIRSYS_USERNAME,
#        credential=XIRSYS_CREDENTIAL,
#    ),
#]

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
'''
@app.websocket("/ws/transcriptions")
async def transcriptions_ws(websocket: WebSocket):
    await websocket.accept()
    live_transcriptions.add(websocket)
    logger.info("New live transcription client connected.") # Added log
    try:
        await asyncio.Future()
    except WebSocketDisconnect:
        # Client disconnected (e.g., closed the tab)
        live_transcriptions.remove(websocket)
        logger.info("Live transcription client disconnected gracefully.") # Added log
    except Exception as e:
        # Handle unexpected errors gracefully
        logger.error(f"Unexpected WebSocket error: {e}")
        if websocket in live_transcriptions:
            live_transcriptions.remove(websocket)
'''
@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

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

        background_tasks.add_task(run_bot2, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/login")
async def serve_login():
    return FileResponse(os.path.join(os.path.dirname(__file__), "login.html"))

@app.get("/options")
async def serve_options():
    return FileResponse(os.path.join(os.path.dirname(__file__), "options.html"))
'''
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

@app.post("/rag")
async def rag_search(request: Request):

    logger.info("🔍 [RAG_ENDPOINT] RAG search request received")
    try:
        body = await request.json()
        user_query = body.get("query")

        logger.info(f"📝 [RAG_ENDPOINT] Query extracted: '{user_query}'")

        if not user_query:
            logger.warning("⚠️ [RAG_ENDPOINT] Missing 'query' in request body")
            return {"error": "Missing 'query' in request body"}

        logger.info(f"Processing RAG for query: {user_query}")

        query_vector = get_embedding(user_query)
        logger.debug(f"🔢 [RAG_ENDPOINT] Generated query vector with {len(query_vector)} dimensions")
        logger.info("🔍 [RAG_ENDPOINT] Querying Pinecone index")
        #results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=query_vector,
                top_k=3,
                include_metadata=True
            )
        )
        logger.debug(f"📊 [RAG_ENDPOINT] Pinecone query completed")

        matches = results.get("matches", [])
        logger.info(f"🎯 [RAG_ENDPOINT] Found {len(matches)} matches")

        if matches:
            # Log all matches with scores
            #for i, match in enumerate(matches):
            #    score = match.get("score", 0)
            #    question = match.get("metadata", {}).get("question", "N/A")
            #    logger.debug(f"📋 [RAG_ENDPOINT] Match {i+1}: Score={score:.4f}, Question='{question[:100]}{'...' if len(question) > 100 else ''}'")

            best_match = matches[0]
            best_score = best_match.get("score", 0)
            logger.info(f"🏆 [RAG_ENDPOINT] Best match score: {best_score:.4f}")

            if best_score > 0.80:
                answer = best_match.get("metadata", {}).get("text", "")
                logger.success(f"✅ [RAG_ENDPOINT] High confidence match found. Raw answer: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")
                
                if answer:
                    cleaned = clean_answer_for_voice(answer)
                    return {"answer": cleaned}
                else:
                    return {"answer": "I couldn’t find a proper answer."}
            else:
                logger.warning(f"⚠️ [RAG_ENDPOINT] Best match score {best_score:.4f} below threshold (0.80)")
        else:
            logger.warning("⚠️ [RAG_ENDPOINT] No matches found in Pinecone")
            logger.info("🤷 [RAG_ENDPOINT] Returning default response")
            return {"answer": "Let me check that for you."}
    except Exception as e:
        logger.exception("Error during RAG search")
        return {"error": "Internal server error"}

@app.post("/search-flights")
async def search_flights(request: Request):
    body = await request.json()
    source = body.get("source","").strip().lower()
    destination = body.get("destination","").strip().lower()
    date_str = body.get("date", "").strip()  # format expected: DD-MM-YYYY
    logger.info(f"✈️ [FLIGHT_SEARCH] Searching flights from '{source}' to '{destination}'")

    if not source or not destination or not date_str:
        return {"error":"Missing source or destination or date"}
    
    # Convert date to match CSV format
    try:
        date_obj = datetime.strptime(date_str, "%d-%m-%Y")
        formatted_date = date_obj.strftime("%Y-%m-%d")  # Matches CSV
    except ValueError:
        return {"error": "Invalid date format. Use DD-MM-YYYY."}
    
    logger.debug(f"🧾 [DEBUG] CSV Sources: {FLIGHT_DATA['Source'].tolist()}")
    logger.debug(f"🧾 [DEBUG] CSV Destinations: {FLIGHT_DATA['Destination'].tolist()}")
    logger.debug(f"🧾 [DEBUG] CSV Dates: {FLIGHT_DATA['Date'].tolist()}")
    
    logger.debug(f"🔁 [DEBUG] Matching against: source='{source}', destination='{destination}', date='{formatted_date}'")

    #filter flights
    filtered = FLIGHT_DATA[
        (FLIGHT_DATA["Source"].astype(str).str.strip().str.lower() == source) &
        (FLIGHT_DATA["Destination"].astype(str).str.strip().str.lower() == destination) &
        (FLIGHT_DATA["Date"].astype(str).str.strip() == formatted_date)
    ]
    logger.info(f"🔎 [FLIGHT_SEARCH] Matched {len(filtered)} flights")


    if not filtered.empty:
        results = []
        for _,row in filtered.iterrows():
            result = {
                "flight_type":row["Flight Type"],
                "flight_number": row["Flight Number"],
                "departure_time": row["Departure Time"],
                "arrival_time": row["Arrival Time"],
                "status": row["Status"],
                "economy_price": row["Price (Economy)"],
                "business_price": row["Price (Business)"]
            }
            results.append(result)

        logger.success(f"✅ [FLIGHT_SEARCH] Found {len(results)} flight(s)")
        return {"flights": results}
    else:
        logger.warning("❌ [FLIGHT_SEARCH] No flights found")
        return {"flights": []}

@app.post("/pnr-status")
async def pnr_status(request:Request):
    body = await request.json()
    pnr = body.get("pnr","").strip().upper()

    logger.info(f"📦 [PNR] Searching PNR: {pnr}")

    if not pnr:
        return {"error": "PNR number is required."}
    
    try:
        conn = sqlite3.connect("pnr_database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pnr_status WHERE pnr_number = ?", (pnr,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            logger.warning(f"❌ [PNR] No booking found for PNR: {pnr}")
            return {"status": "not_found"}

        logger.success(f"✅ [PNR] Booking found: {row}")
        return {
            "status": "found",
            "pnr_number": row[0],
            "first_name": row[1],
            "last_name": row[2],
            "flight_number": row[3],
            "source": row[4],
            "departure_time": row[5],
            "destination": row[6],
            "arrival_time": row[7],
            "date": row[8],
            "booking_status": row[9]
        }

    except Exception as e:
        logger.exception("[PNR] Error while checking PNR")
        return {"error": "Internal server error"}
    
@app.get("/sessions")
async def list_sessions():
    logs = []
    for fname in sorted(os.listdir(LOG_DIR),reverse=True):
        if fname.endswith(".log"):
            parts=fname.split("_")
            ts = " ".join(parts[:2])
            session_id = parts[2].replace(".log", "")
            logs.append({"filename": fname, "timestamp": ts, "session_id": session_id})
    return logs

@app.get("/download-log/{filename}")
async def download_log(filename: str):
    safe_path = os.path.join(LOG_DIR,filename)
    if not (os.path.isfile(safe_path) and filename.endswith(".log")):
        return JSONResponse({"error": "File not found"},status_code=404)
    return FileResponse(safe_path,media_type="text/plain", filename=filename)
''' 
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
        "--port", type=int, default=8001, help="Port for HTTP server (default: 8001)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)