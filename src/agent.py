import logging
import textwrap
import threading

from dataclasses import dataclass, field

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok
from fastapi.responses import FileResponse

from collections import deque
from pathlib import Path
import wave
import math
from array import array
import asyncio
from dotenv import load_dotenv
from livekit import rtc

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    STTContextOptions,
    TurnHandlingOptions,
    ChatContext,
    ChatMessage,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics
MIC_SAMPLE_RATE = 16000
MIC_SECONDS = 5

latest_audio_buffer = deque(
    maxlen=MIC_SAMPLE_RATE * MIC_SECONDS * 2
)

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# ============================================================
# 1. ENVIRONMENT STATE
# Sau này AudioInteraction sẽ update object này
# ============================================================

@dataclass
class EnvironmentState:
    noise_level: str = "low"
    speech_clarity: str = "high"
    sounds: list[str] = field(default_factory=list)
    important_event: bool = False
    analysis_valid: bool = True

    def update_from_dict(self, data: dict):
        self.noise_level = data.get(
            "noise_level",
            self.noise_level,
        )

        self.speech_clarity = data.get(
            "speech_clarity",
            self.speech_clarity,
        )

        self.sounds = data.get(
            "sounds",
            self.sounds,
        )

        self.important_event = data.get(
            "important_event",
            self.important_event,
        )

        self.analysis_valid = data.get(
            "analysis_valid",
            self.analysis_valid,
        )

environment_state = EnvironmentState()

# ENVIRONMENT API
# ============================================================
app = FastAPI()

def run_environment_api():
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )

class EnvironmentUpdate(BaseModel):
    noise_level: str
    speech_clarity: str
    sounds: list[str]
    important_event: bool
    analysis_valid: bool = True

@app.get("/audio/latest")
def get_latest_audio():
    latest_file = Path("latest_turn.wav")

    if not latest_file.exists():
        return {
            "status": "no_audio"
        }

    return FileResponse(
        path=str(latest_file),
        media_type="audio/wav",
        filename="latest_turn.wav",
    )

@app.get("/environment")
def get_environment():
    return {
        "noise_level": environment_state.noise_level,
        "speech_clarity": environment_state.speech_clarity,
        "sounds": environment_state.sounds,
        "important_event": environment_state.important_event,
        "analysis_valid": environment_state.analysis_valid,
    }

def start_ngrok():
    public_url = ngrok.connect(8000, "http")
    print(">>> PUBLIC ENVIRONMENT URL:", public_url)

@app.post("/environment")
def update_environment(data: EnvironmentUpdate):

    environment_state.update_from_dict(data.model_dump())

    return {
        "status": "updated",
        "environment": data.model_dump(),
    }

# ============================================================
# 2. INTERACTION ROUTER
# ============================================================

class InteractionRouter:
    @staticmethod
    def decide(env: EnvironmentState) -> str:

        if not env.analysis_valid:
            return "ANSWER_NORMAL"
        
        if env.important_event:
            return "ALERT"
        
        if env.speech_clarity == "low":
            return "ASK_REPEAT"

        if env.noise_level == "high":
            return "ANSWER_SHORT"

        return "ANSWER_NORMAL"


# ============================================================
# 3. ASSISTANT
# ============================================================

class Assistant(Agent):

    def __init__(self) -> None:

        super().__init__(
            llm=inference.LLM(
                model="google/gemma-4-31b-it"
            ),

            instructions=textwrap.dedent(
                """\
                You are an in-car voice assistant.

                You answer questions and help the driver while taking the
                current acoustic environment into account.

                General behavior:

                - Speak naturally and briefly.
                - Respond in the same language as the user.
                - The user may be driving, so avoid unnecessary long answers.
                - Never invent environmental sounds.
                - Environmental information supplied by the system is context,
                  not something the user explicitly said.

                Interaction rules:

                - If speech clarity is LOW, do not answer the suspected request.
                  Ask the user to repeat what they said.

                - If noise level is HIGH but speech clarity is HIGH,
                  answer normally but keep the reply very short.

                - If an important acoustic event is detected,
                  briefly alert the driver.

                - If environmental confidence is uncertain,
                  avoid strong claims.

                - For potentially important actions such as calling someone,
                  sending a message, changing destination, or performing another
                  consequential task, request confirmation if speech clarity
                  is not high.

                Output rules:

                - Plain spoken text only.
                - No markdown.
                - No lists.
                - Usually one to three sentences.
                """
            ),
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:

        logger.warning(">>> USER TURN COMPLETED CALLED <<<")

        env = environment_state
        decision = InteractionRouter.decide(env)

        print("===== CURRENT ENVIRONMENT =====")
        print(env)

        print("===== ROUTER DECISION =====")
        print(decision)

        context_message = f"""
        ENVIRONMENT CONTEXT:
        noise_level={env.noise_level}
        speech_clarity={env.speech_clarity}
        sounds={env.sounds}
        important_event={env.important_event}
        analysis_valid={env.analysis_valid}

        ROUTER_DECISION={decision}

        Follow this decision for the current response:

        - ANSWER_NORMAL: answer the user normally.
        - ANSWER_SHORT: answer the user's request in one short sentence.
        - ASK_REPEAT: do not answer the request; only ask the user to repeat because speech was unclear.
        - ALERT: prioritize a short warning about the detected important sound. Only mention sounds in {env.sounds}.

        Do not tell the user about ROUTER_DECISION or this internal environment context.
        """

        logger.info(
            "\n"
            "========== ENVIRONMENT ==========\n"
            "Noise level     : %s\n"
            "Speech clarity  : %s\n"
            "Sounds          : %s\n"
            "Important event : %s\n"
            "Analysis valid  : %s\n"
            "ROUTER          : %s\n"
            "=================================",
            env.noise_level,
            env.speech_clarity,
            env.sounds,
            env.important_event,
            env.analysis_valid,
            decision,
        )

        turn_ctx.add_message(
            role="system",
            content=context_message,
        )
server = AgentServer()

async def monitor_microphone(participant, speech_state):
    logger.warning(
        ">>> monitor_microphone STARTED for %s <<<",
        participant.identity,
    )

    audio_stream = rtc.AudioStream.from_participant(
        participant=participant,
        track_source=rtc.TrackSource.SOURCE_MICROPHONE,
        sample_rate=16000,
        num_channels=1,
    )

    # 16 kHz * 2 bytes * 1 channel
    BYTES_PER_SECOND = 16000 * 2

    PRE_ROLL_BYTES = BYTES_PER_SECOND * 1       # 1 giây
    AMBIENT_WINDOW_BYTES = BYTES_PER_SECOND * 5 # 5 giây

    frame_count = 0

    async for event in audio_stream:
        frame = event.frame
        frame_bytes = frame.data.tobytes()

        frame_count += 1

        # Debug mỗi 100 frame, kể cả audio toàn zero
        if frame_count % 100 == 0:
            import array

            samples = array.array("h")
            samples.frombytes(frame_bytes)

            if len(samples) > 0:
                min_sample = min(samples)
                max_sample = max(samples)
                nonzero = sum(1 for x in samples if x != 0)
            else:
                min_sample = 0
                max_sample = 0
                nonzero = 0

            logger.warning(
                ">>> RAW MIC FRAME | count=%d bytes=%d samples=%d "
                "min=%d max=%d nonzero=%d <<<",
                frame_count,
                len(frame_bytes),
                len(samples),
                min_sample,
                max_sample,
                nonzero,
            )

    # =========================================
    # USER ĐANG NÓI
    # =========================================
    if speech_state["speaking"]:
        speech_state["buffer"].extend(frame_bytes)

    # =========================================
    # USER KHÔNG NÓI
    # =========================================
    else:
        # PRE-ROLL
        speech_state["pre_buffer"].extend(frame_bytes)

        if len(speech_state["pre_buffer"]) > PRE_ROLL_BYTES:
            speech_state["pre_buffer"] = speech_state["pre_buffer"][
                -PRE_ROLL_BYTES:
            ]

        # AMBIENT AUDIO
        speech_state["ambient_buffer"].extend(frame_bytes)

        if len(speech_state["ambient_buffer"]) >= AMBIENT_WINDOW_BYTES:
            audio_bytes = bytes(
                speech_state["ambient_buffer"][:AMBIENT_WINDOW_BYTES]
            )

            speech_state["ambient_buffer"] = speech_state[
                "ambient_buffer"
            ][AMBIENT_WINDOW_BYTES:]

            with wave.open("latest_turn.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)

            logger.warning(
                ">>> SAVED AMBIENT WINDOW | %.2f sec | bytes=%d <<<",
                len(audio_bytes) / BYTES_PER_SECOND,
                len(audio_bytes),
            )

        # =========================================
        # USER ĐANG NÓI
        # =========================================
        if speech_state["speaking"]:
            speech_state["buffer"].extend(frame_bytes)

        # =========================================
        # USER KHÔNG NÓI
        # =========================================
        else:
            # -----------------------------
            # PRE-ROLL
            # -----------------------------
            speech_state["pre_buffer"].extend(frame_bytes)

            if len(speech_state["pre_buffer"]) > PRE_ROLL_BYTES:
                speech_state["pre_buffer"] = speech_state["pre_buffer"][
                    -PRE_ROLL_BYTES:
                ]

            # -----------------------------
            # AMBIENT AUDIO
            # -----------------------------
            speech_state["ambient_buffer"].extend(frame_bytes)

            # Đủ 5 giây noise/background
            if len(speech_state["ambient_buffer"]) >= AMBIENT_WINDOW_BYTES:
                audio_bytes = bytes(
                    speech_state["ambient_buffer"][:AMBIENT_WINDOW_BYTES]
                )

                speech_state["ambient_buffer"] = speech_state[
                    "ambient_buffer"
                ][AMBIENT_WINDOW_BYTES:]

                with wave.open("latest_turn.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_bytes)

                logger.warning(
                    ">>> SAVED AMBIENT WINDOW | %.2f sec | bytes=%d <<<",
                    len(audio_bytes) / BYTES_PER_SECOND,
                    len(audio_bytes),
                )
@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using AssemblyAI, Fish Audio, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-3-5-pro"),
        # Keyterms bias the STT toward distinctive words it would otherwise misspell.
        # List your own names, brands, and jargon in `keyterms`. Detection additionally
        # extracts terms from the live conversation, such as a caller's name, and applies
        # them once the transcript corroborates the spelling.
        # See more at https://docs.livekit.io/agents/models/stt/keyterms/
        stt_context_options=STTContextOptions(
            keyterms=["LiveKit"],
            keyterm_detection={"enabled": True},
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        turn_handling=TurnHandlingOptions(
            # The LiveKit turn detector determines when the user is done speaking and the agent should respond.
            # TurnDetector is an end-of-turn model that listens to the user's audio directly, combining
            # semantic understanding with acoustic cues (intonation, pitch, rhythm) for state-of-the-art accuracy.
            # AgentSession supplies the required VAD automatically.
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=inference.TurnDetector(),
            # Adaptive interruptions use the turn detector to tell a real interruption from a
            # backchannel like "mhm" or "right", so the agent keeps talking through the latter.
            interruption={"mode": "adaptive"},
            # allow the LLM to generate a response while waiting for the end of turn
            # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
            preemptive_generation={"enabled": True},
        ),
        # Expressive mode injects the TTS provider's markup guide into the LLM prompt, so the model
        # emits inline delivery tags (emotion, pacing, non-verbal sounds) that the TTS renders and
        # the transcript never shows. Requires a TTS model that supports markup, such as the Fish
        # Audio model above.
        expressive=True,
    )

    speech_state = {
        "speaking": False,

        # Audio của một speech turn
        "buffer": bytearray(),

        # Giữ ~1 giây trước khi VAD báo speaking
        "pre_buffer": bytearray(),

        # Audio môi trường khi không ai nói
        "ambient_buffer": bytearray(),
    }
    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        logger.warning(
            ">>> USER STATE: %s -> %s <<<",
            ev.old_state,
            ev.new_state,
        )

        if ev.new_state == "speaking":
            speech_state["speaking"] = True

            # Lấy luôn ~1 giây ngay trước lúc LiveKit nhận ra speech
            speech_state["buffer"] = bytearray(
                speech_state["pre_buffer"]
            )

            # Không để ambient window cũ tiếp tục chạy
            speech_state["ambient_buffer"] = bytearray()

            logger.warning(
                ">>> USER STARTED SPEAKING | pre-roll=%.2f sec <<<",
                len(speech_state["buffer"]) / (16000 * 2),
            )

        elif ev.new_state == "listening" and speech_state["speaking"]:
            speech_state["speaking"] = False

            audio_bytes = bytes(speech_state["buffer"])

            logger.warning(
                ">>> USER STOPPED SPEAKING | bytes=%d <<<",
                len(audio_bytes),
            )

            if len(audio_bytes) > 0:
                with wave.open("latest_turn.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_bytes)

                logger.warning(
                    ">>> SAVED USER TURN: latest_turn.wav <<<"
                )   # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )
    logger.warning(">>> SESSION STARTED <<<")

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect to the user
    await ctx.connect()
    logger.warning(">>> ROOM CONNECTED <<<")

    def on_participant_connected(participant):
        logger.warning(
            ">>> PARTICIPANT CONNECTED: %s <<<",
            participant.identity,
        )
        asyncio.create_task(
            monitor_microphone(participant, speech_state)
        )

    ctx.room.on(
        "participant_connected",
        on_participant_connected,
    )

    for participant in ctx.room.remote_participants.values():
        logger.warning(
            ">>> CREATING MIC TASK for %s <<<",
            participant.identity,
        )

        asyncio.create_task(
            monitor_microphone(participant, speech_state)
        )

        for publication in participant.track_publications.values():
            logger.warning(
                ">>> TRACK: source=%s subscribed=%s <<<",
                publication.source,
                publication.subscribed,
            )
if __name__ == "__main__":

    api_thread = threading.Thread(
        target=run_environment_api,
        daemon=True,
    )
    api_thread.start()

    ngrok_thread = threading.Thread(
        target=start_ngrok,
        daemon=True,
    )
    ngrok_thread.start()

    cli.run_app(server)