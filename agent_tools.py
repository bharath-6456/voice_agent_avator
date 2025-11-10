        #
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os
import sys,aiohttp,asyncio,sqlite3,logging
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime,timezone
import time,uuid,logging
from typing import Dict, Optional, List
import re

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.processors.frame_processor import FrameProcessor,FrameDirection
from pipecat.observers.base_observer import BaseObserver,FramePushed
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
   BotStoppedSpeakingFrame
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
#from shared_state import live_transcriptions

from sentence_transformers import SentenceTransformer
from pipecat.services.heygen.video import HeyGenVideoService
from pipecat.services.heygen.api import AvatarQuality, NewSessionRequest
from pinecone import Pinecone

#from pipecat.audio.filters.noisereduce_filter import NoisereduceFilter
load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
You are Sama, a friendly and helpful voice assistant for Super Airlines.

Your job is to answer user questions in a clear and concise way.
 
Always follow these rules:
 
Keep responses short (1–2 sentences).
 
Stay friendly, polite, and easy to understand.
 
If no useful answer is found, say:
 
“I’m sorry, I don’t have that information right now. Please try again later.”
 
Example Interactions
 
User: How can I book a ticket?

Sama: Sure! Please share your departure city, destination, travel date, and number of passengers so I can help you with the booking.
 
User: I want to cancel my flight ticket.

Sama: I can help with that. Please provide your booking reference or PNR number, and I’ll check your cancellation options.
 
User: Can you check my PNR status?

Sama: Of course! Please tell me your PNR number so I can look up your booking details.
 
User: What’s the baggage limit for domestic flights?

Sama: You can carry up to 15 kg of check-in baggage and 7 kg of cabin baggage on domestic flights.
 
User: What meals do you serve on international flights?

Sama: We serve both vegetarian and non-vegetarian meals on all international routes. You can choose your preference while booking or check-in.
 
User: What’s the status of flight SA203 to Delhi?

Sama: I’m sorry, I don’t have that information right now. Please try again later.
 
"""

api_key = os.getenv("PINECONE_API_KEY")
index_name = "travelport-smartpoint"
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

model = SentenceTransformer("intfloat/e5-base")


def start_session_log(session_id=None):
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR,exist_ok=True)
    session_id = str(uuid.uuid4())
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fname = f"{start_time}_{session_id}.log"
    log_path = os.path.join(LOG_DIR, log_fname)
    logger = logging.getLogger(session_id)
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    print(f"Session log created: {log_path}")
    return session_id, log_fname, logger

def get_embedding(text: str):
    return model.encode(text).tolist()

async def rag_search_handler(params:FunctionCallParams):
    """
    Handler for RAG search function calls
    """
    logger.info(f"[TOOL] rag_search called with query: {params.arguments}")
    try:
        query = params.arguments.get("query","")

        if not query:
            logger.warning("⚠️ [RAG_HANDLER] Empty query received")
            await params.result_callback("I need a search query to help you")
            return

        logger.info(f"🔎 [RAG_HANDLER] Processing query: '{query}'")

        query_vector = get_embedding(query)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: index.query(vector=query_vector, top_k=1, include_metadata=True)
        )

        matches = results.get("matches", [])
        if matches:
            best_match = matches[0]
            score = best_match.get("score", 0)
            text_answer = best_match.get("metadata", {}).get("text", "")

            if score > 0.80 and text_answer:
                await params.result_callback(text_answer)
            else:
                await params.result_callback("Let me check that for you.")
        else:
            await params.result_callback("I’m sorry, I don’t have that information right now. Please try again later.")
    except Exception as e:
        logger.exception(f"[TOOL] Error in rag_search: {e}")
        await params.result_callback("I'm having trouble finding that information. Please try again.")


   # await params.result_callback("This is a test answer from the RAG tool.")



async def run_bot2(webrtc_connection):

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
            audio_out_10ms_chunks=2,
            #audio_in_filter=NoisereduceFilter(),
        ),
    )
    logger.debug("🔌 [BOT] Transport initialized")
    
    async with aiohttp.ClientSession() as session:
        stt = GroqSTTService(api_key=os.getenv("GROQ_API_KEY"),model="whisper-large-v3-turbo")
        tts = SarvamTTSService(api_key=os.getenv("SARVAM_API_KEY"),voice_id="manisha",model="bulbul:v2",aiohttp_session=session,params=SarvamTTSService.InputParams(language=Language.EN))
        llm = GroqLLMService(api_key=os.getenv("GROQ_API_KEY"),model="meta-llama/llama-4-scout-17b-16e-instruct", params=GroqLLMService.InputParams(streaming=True))
        logger.debug("🤖 [BOT] LLM service initialized")
        heyGen = HeyGenVideoService(
            api_key=os.getenv("HEYGEN_API_KEY"),
            session=session,
            session_request=NewSessionRequest(
                # You can change the avatar_id to your desired Heygen avatar
                #avatar_id="Shawn_Therapist_public",
                avatar_id="Katya_Chair_Sitting_public",
                version="v2",
                quality=AvatarQuality.high
            ),
        )
        llm.register_function("rag_search",rag_search_handler)
        # llm.register_function("flight_search", flight_search_handler)
        # llm.register_function("pnr_search", pnr_search_handler)


        logger.info("🔧 [BOT] RAG search function registered with LLM")

        rag_function = FunctionSchema(
            name="rag_search",
            description= "Search the knowledge base for information about Indigo Voice services, account issues,faq questions and general company information",
            properties={
                "query":{
                    "type":"string",
                    "description":"The user's question or search query",
                }
            },
            required=["query"]
        )





        tools = ToolsSchema(standard_tools=[rag_function])
        messages = [
            {
                "role":"system",
                "content":SYSTEM_INSTRUCTION,
            },
        ]

        context = OpenAILLMContext(messages=messages, tools=tools)
        logger.debug("📝 [BOT] Context initialized with tools")
        context_aggregator = llm.create_context_aggregator(context)
        logger.debug("📊 [BOT] Context aggregator created")

        #session_id = str(uuid.uuid4())
        #latency_tracker = LatencyTracker(session_id)
        transcript = TranscriptProcessor()

        pipeline = Pipeline(
            [
                pipecat_transport.input(),
                stt,
                transcript.user(),
                context_aggregator.user(),
                llm,  # LLM
                tts,
                heyGen,
                pipecat_transport.output(),
                transcript.assistant(),
                context_aggregator.assistant(),
                #latency_tracker
            ]
        )


        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
                allow_interruptions=True,
            ),
            enable_tracing=True,
            enable_turn_tracking=True,
        )

        @pipecat_transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Pipecat Client connected")
            # Kick off the conversation.
            #start_session_log()
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @pipecat_transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipecat Client disconnected")
            await task.cancel()

        '''
        @transcript.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            logger.info(">>> on_transcript_update CALLED <<<")
            for msg in frame.messages:
                if isinstance(msg, TranscriptionMessage):
                    # timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                    # line = f"{timestamp}{msg.role}: {msg.content}"
                    # logger.info(f"Transcript: {line}")
                    role = msg.role
                    content = msg.content
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    # Send to all connected websockets
                    for ws in live_transcriptions.copy():
                        try:
                            logger.info(f"Pushing transcription to websockets: role={role}, text={content}, ts={timestamp} ... conns={live_transcriptions}")
                            await ws.send_json({"role": role, "text": content,"timestamp":timestamp})
                        except Exception as e:
                            # Log the error and remove the disconnected WebSocket
                            logger.warning(f"Could not send transcription to client. Removing from list: {e}")
                            if ws in live_transcriptions:
                                live_transcriptions.remove(ws)
          '''
        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)