import os
import subprocess
import time
import typing
from openai import OpenAI

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from elevenlabs.core.request_options import RequestOptions

# Inspiration:
# https://elevenlabs.io/docs/api-reference/websockets#example-of-voice-streaming-using-elevenlabs-and-openai


# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
#aclient = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
#client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')

# Set API keys and voice ID
# tts_client = ElevenLabs(
#   api_key=os.getenv('ELEVENLABS_API_KEY'),
# )

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'
OPTIMIZE_STREAMING_LATENCY = 4
#Possible values: 
# 0 - default mode (no latency optimizations) 
# 1 - normal latency optimizations (about 50% of possible latency improvement of option 3) 
# 2 - strong latency optimizations (about 75% of possible latency improvement of option 3) 
# 3 - max latency optimizations 
# 4 - max latency optimizations, but also with text normalizer turned off for even more latency savings (best latency, but can mispronounce eg numbers and dates).



def stream(audio_stream):
    """Stream audio data using mpv player."""
    mpv_process = subprocess.Popen(
         ["mpv", "--no-cache", "--no-terminal", "--idle=no", "--loop=no", "--", "fd://0"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    for chunk in audio_stream:
        if chunk:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()


def openai_to_elevenlabs(openai_client, elevenlabs_client, thread_id, assistant_id, voice_id=VOICE_ID, text=None):
    """Fetch text responses from OpenAI and stream it to Elevenlabs for speech synthesis."""

    # see: https://github.com/elevenlabs/elevenlabs-python/blob/f8c44d52910c8dab4c8e3cfc6493c3969ba8db6a/src/elevenlabs/realtime_tts.py#L65

    if text and openai_client is not None:
        send_message(openai_client, thread_id, text)


    def text_iterator() -> typing.Iterator[str]:
        if openai_client is None:
            yield text
            return
        
        with openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant_id,
        ) as stream:
            for event in stream:
                # delete instructions attribute from event
                # if hasattr(event.data, "instructions"):
                #     del event.data.instructions
                # print(event)
                if event.event == "thread.message.delta" and event.data.delta.content:
                    print(event.data.delta.content[0].text.value, end="", flush=True)
                    yield event.data.delta.content[0].text.value
                # end of stream
                elif event.event == "thread.message.completed":
                    break
                elif event.event == "thread.run.failed":
                    print("Thread run failed:")
                    print(event.data.last_error)
                    break

    if elevenlabs_client is None:
        # don't speak the text
        for t in text_iterator():
            text += t
        return

    audio_stream = elevenlabs_client.text_to_speech.convert_realtime(
        voice_id=voice_id,
        text=text_iterator(),
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0, # anything > 0 causes latency
            use_speaker_boost=False,), #: true causes latency
        request_options=RequestOptions(
            additional_query_parameters={
                "optimize_streaming_latency": OPTIMIZE_STREAMING_LATENCY
            }),
    )

    stream(audio_stream)


def send_message(client, thread_id, text):
    """Send a message to the thread."""
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=text
    )