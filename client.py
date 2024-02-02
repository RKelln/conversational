import argparse
import asyncio
import threading

from conversation import Conversation
from api.openai_api_async import (create_assistant, create_thread, add_message_to_thread, get_answer)

from TTSManager import TTSManager

from whisper_live.client import TranscriptionClient
from elevenlabs import generate, stream


async def send_text_to_LLM(text, assistant, thread):
    await add_message_to_thread(thread.id, text)
    reply = await get_answer(assistant.id, thread)
    return reply


async def process_transcriptions(queue, assistant, thread):
    tts_manager = TTSManager()
    while True:
        print("Waiting for transcription...")
        text = await queue.get()
        print("Human: ", text)
        reply = await send_text_to_LLM(text, assistant, thread)
        print("Spirit: ", reply)
        await tts_manager.speak(reply)


def transcription_callback(queue, loop, text):
    # Instead of directly invoking asyncio operations, we now schedule them
    loop.call_soon_threadsafe(lambda: queue.put_nowait(text))


def start_client(queue, loop, host="localhost", port=9090, model_size="small"):
    transcription_client = TranscriptionClient(
        host=host,
        port=port,
        model_size=model_size,
        is_multilingual=False,
        lang="en",
        translate=False,
        callback=lambda text: transcription_callback(queue, loop, text),
    )
    print(f"Client created, connecting to server {host}:{port}")
    # The client starts its own thread for WebSocket communication
    transcription_client()


async def main(args):
    # Start the async task to process transcription texts
    # asyncio.create_task(process_transcriptions())
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Start the transcription client
    # Since it runs in its own thread, it won't block the asyncio event loop
    client_thread = threading.Thread(target=start_client, args=(queue, loop), 
                                     kwargs=vars(args)) 
    client_thread.start()

    assistant = await create_assistant()
    thread = await create_thread()

    await process_transcriptions(queue, assistant, thread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sohkepayin Live Client")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--model_size", type=str, default="small")

    args = parser.parse_args()

    asyncio.run(main(args))
