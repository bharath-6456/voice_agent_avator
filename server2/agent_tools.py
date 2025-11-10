import os
import sys, aiohttp, asyncio, sqlite3
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime, timezone
import time, uuid, logging
from typing import Dict, Optional, List
import re

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    TextFrame,
    EndFrame,
    StartFrame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
    MetricsFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.frames.frames import TranscriptionMessage
from textblob import TextBlob
# from shared_state import live_transcriptions
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# RAG-specific implementation imports (like LangChain, FAISS) are removed 
# as this logic now lives exclusively on the server.

import math
from pipecat.services.llm_service import FunctionCallParams

from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.services.heygen.api import AvatarQuality, NewSessionRequest
# from pipecat.services.tavus.video import TavusVideoService

load_dotenv(override=True)


SYSTEM_INSTRUCTION = f"""
Your name is Anish, a friendly and helpful voice assistant. Begin talking with "hello, myself Anish, How can I help you today?"
Your job is to answer user questions in a clear and concise way.

You have access to one RAG DB. 
---
Your responses should always:
Be short and to the point (one or two sentences).
Stay friendly, polite, and easy to understand.
Only use the information returned from the tools.

If no useful answer is found, say: "I’m sorry, I don’t have that information right now. Please try again later."

Do not ask for personal details or complicate the conversation.
Your goal is to clearly answer questions based on the knowledge base.
"""

# -----------------------------------------------------------------
# START: MODIFIED RAG HANDLER
# -----------------------------------------------------------------

# Add this global cache near the top of agent_tools.py (outside the function)
rag_cache = {}

async def rag_search_implementation(params: FunctionCallParams, session_id: str):
    """
    Handler for RAG search function calls.
    It forwards the query and session_id to the main server's /rag/query endpoint.
    Includes caching to prevent redundant RAG calls for the same query/session.
    """
    logger.info(f"[TOOL] rag_search called for session {session_id} with query: {params.arguments}")
    try:
        query = params.arguments.get("query", "").strip()

        if not query:
            logger.warning("⚠️ [RAG_HANDLER] Empty query received")
            await params.result_callback("I need a search query to help you")
            return

        # --- NEW: CACHE KEY & DEDUPLICATION ---
        key = f"{session_id}:{query.lower()}"
        if key in rag_cache:
            logger.info(f"⚡ [RAG_HANDLER] Cached result used for {key}")
            cached_answer = rag_cache[key]
            await params.result_callback(cached_answer)
            return
        # -------------------------------------

        logger.info(f"🔎 [RAG_HANDLER] Processing query for session {session_id}: '{query}'")

        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    json_payload = {"query": query, "session_id": session_id}
                    async with session.post("http://localhost:8000/rag/query", json=json_payload) as resp:

                        if resp.status == 200:
                            data = await resp.json()
                            answer = data.get("answer") or "The information is not available in the document."
                            logger.info(f"✅ [RAG_HANDLER] Got answer for session {session_id}")

                            # --- NEW: SAVE TO CACHE ---
                            rag_cache[key] = answer
                            # ---------------------------

                            await params.result_callback(answer)
                            return
                        
                        elif resp.status == 400:
                            data = await resp.json()
                            logger.error(f"[RAG_HANDLER] 400 error for session {session_id}: {data.get('detail')}")
                            msg = "I'm sorry, I can't access the document. Please make sure it's uploaded correctly."
                            rag_cache[key] = msg  # cache failure response too
                            await params.result_callback(msg)
                            return

                        elif resp.status == 503:
                            logger.warning(f"[RAG_HANDLER] 503 error on attempt {attempt + 1}, retrying...")
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                await asyncio.sleep(delay)
                                continue
                            else:
                                msg = "The service is temporarily unavailable. Please try again later."
                                rag_cache[key] = msg
                                await params.result_callback(msg)
                                return
                        else:
                            msg = "I'm having trouble finding that information. Please try again."
                            rag_cache[key] = msg
                            logger.error(f"[RAG_HANDLER] Backend returned status {resp.status}")
                            await params.result_callback(msg)
                            return
                            
            except Exception as e:
                logger.exception(f"[TOOL] Error in rag_search on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    msg = "I'm having trouble finding that information. Please try again."
                    rag_cache[key] = msg
                    await params.result_callback(msg)
                    return

    except Exception as e:
        logger.exception(f"[TOOL] Unexpected error in rag_search: {e}")
        msg = "I'm having trouble finding that information. Please try again."
        rag_cache[f"{session_id}:{query.lower()}"] = msg
        await params.result_callback(msg)

# -----------------------------------------------------------------
# END: MODIFIED RAG HANDLER
# -----------------------------------------------------------------


async def flight_search_handler(params: FunctionCallParams):
    logger.info(f"[TOOL] flight_search called with args: {params.arguments}")
    try:
        source = params.arguments.get("source", "")
        destination = params.arguments.get("destination", "")
        date = params.arguments.get("date", "")  # expected in DD-MM-YYYY

        if not source or not destination or not date:
            await params.result_callback("Please specify both source and destination cities and date.")
            return
        async with aiohttp.ClientSession() as session:
            payload = {"source": source, "destination": destination, "date": date}
            async with session.post("http://localhost:8000/search-flights", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    flights = data.get("flights", [])

                    if flights:
                        flight = flights[0]  # You can also loop through multiple
                        msg = (
                            f"Flight {flight['flight_number']} is a {flight['flight_type']} departing at {flight['departure_time']} "
                            f"and arriving at {flight['arrival_time']}. Current status is {flight['status']}. "
                            f"Economy seats are available for {int(flight['economy_price'])} dollars."
                        )
                        await params.result_callback(msg)
                    else:
                        await params.result_callback("Sorry, no flights found for your route.")
                else:
                    await params.result_callback("Flight search failed. Please try again.")
    except Exception as e:
        logger.exception(f"[TOOL] Error in flight_search: {e}")
        await params.result_callback("Something went wrong while searching for flights.")


async def pnr_search_handler(params: FunctionCallParams):
    logger.info(f"[TOOL] pnr_search called with args: {params.arguments}")
    try:
        pnr = params.arguments.get("pnr", "").strip().upper()

        if not pnr:
            await params.result_callback("Please provide your PNR number to continue.")
            return

        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8000/pnr-status", json={"pnr": pnr}) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "found":
                        msg = (
                            f"PNR {data['pnr_number']} belongs to {data['first_name']} {data['last_name']}. "
                            f"Flight {data['flight_number']} from {data['source']} to {data['destination']} on {data['date']}, "
                            f"departs at {data['departure_time']} and arrives at {data['arrival_time']}. "
                            f"Status: {data['booking_status']}."
                        )
                        await params.result_callback(msg)
                    else:
                        await params.result_callback(f"No booking found for PNR {pnr}. Please check and try again.")
                else:
                    await params.result_callback("Unable to check PNR right now. Please try again later.")
    except Exception as e:
        logger.exception("[TOOL] Error in pnr_search")
        await params.result_callback("Something went wrong while checking your PNR. Please try again.")


# -----------------------------------------------------------------
# START: MODIFIED run_bot2
# -----------------------------------------------------------------

async def run_bot2(webrtc_connection, session_id: str):
    """
    This function now accepts a session_id to create a session-aware RAG handler.
    """
    logger.info(f"🚀 [BOT] Initializing bot for session_id: {session_id}")
    
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=1280,
            video_out_height=720,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=10,
        ),
    )
    logger.debug("🔌 [BOT] Transport initialized")

    async with aiohttp.ClientSession() as session:
        stt = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"), model="whisper-large-v3-turbo")
        tts = SarvamTTSService(api_key=os.getenv("SARVAM_API_KEY"), voice_id="hitesh", model="bulbul:v2", aiohttp_session=session, params=SarvamTTSService.InputParams(language=Language.EN))
        llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-maverick-17b-128e-instruct")
        logger.debug("🤖 [BOT] LLM service initialized")
        
        heyGen = HeyGenVideoService(
            api_key=os.getenv("HEYGEN_API_KEY"),
            session=session,
            session_request=NewSessionRequest(
                avatar_id="Shawn_Therapist_public",
                version="v2",
                quality=AvatarQuality.high
            ),
        )

        # --- THIS IS THE KEY CHANGE ---
        # 1. Create a "session-aware" wrapper function (a closure)
        #    This function has access to the 'session_id' from the parent 'run_bot2' scope
        async def session_aware_rag_handler(params: FunctionCallParams):
            logger.debug(f"Passing call to RAG implementation with session_id: {session_id}")
            # This automatically calls the implementation with the correct session_id
            await rag_search_implementation(params, session_id)

        # 2. Register the NEW wrapper function with the LLM
        llm.register_function("rag_search", session_aware_rag_handler)
        # ------------------------------

        # Register the other handlers normally
        llm.register_function("flight_search", flight_search_handler)
        llm.register_function("pnr_search", pnr_search_handler)

        logger.info(f" [BOT] All tool functions registered for session {session_id}")

        rag_function = FunctionSchema(
            name="rag_search",
            description="Search the knowledge base for information about the attached PDF document.",
            properties={
                "query": {
                    "type": "string",
                    "description": "The user's question or search query related to the document",
                }
            },
            required=["query"]
        )

        flight_function = FunctionSchema(
            name="flight_search",
            description="Search for flights between two cities on a specific date",
            properties={
                "source": {
                    "type": "string",
                    "description": "Departure city"
                },
                "destination": {
                    "type": "string",
                    "description": "Arrival city"
                },
                "date": {
                    "type": "string",
                    "description": "Date of travel in DD-MM-YYYY format"
                }
            },
            required=["source", "destination", "date"]
        )

        pnr_function = FunctionSchema(
            name="pnr_search",
            description="Check the booking status using a PNR number",
            properties={
                "pnr": {
                    "type": "string",
                    "description": "The user's 6-8 character PNR number"
                }
            },
            required=["pnr"]
        )

        # --- THIS IS THE KEY CHANGE (FIX) ---
        # Register ALL tools with the ToolsSchema
        tools = ToolsSchema(standard_tools=[rag_function, flight_function, pnr_function])
        # ------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION,
            },
        ]

        context = OpenAILLMContext(messages=messages, tools=tools)
        logger.debug("📝 [BOT] Context initialized with tools")
        context_aggregator = llm.create_context_aggregator(context)
        logger.debug("📊 [BOT] Context aggregator created")

        pipeline = Pipeline(
            [
                pipecat_transport.input(),
                stt,
                context_aggregator.user(),
                llm,  # LLM
                tts,
                heyGen,
                # simli_ai, # Other video services commented out
                # tavus,
                pipecat_transport.output(),
                context_aggregator.assistant(),
                # latency_tracker
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
                allow_interruptions=False,
            ),
        )

        @pipecat_transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Pipecat Client connected for session {session_id}")
            # Kick off the conversation.
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @pipecat_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Pipecat Client disconnected for session {session_id}")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)