import argparse
import asyncio
import string
import threading

from api.openai_api_async import (create_assistant, create_thread, add_message_to_thread, get_answer)

from TTSManager import TTSManager, DummyTTSManager


TESTING = False


async def send_text_to_LLM(text, assistant, thread):
    if TESTING:
        await asyncio.sleep(1)
        return "I am a test response"
    await add_message_to_thread(thread.id, text)
    reply = await get_answer(assistant.id, thread)
    return reply


async def process_transcriptions(queue, assistant, thread, tts_service=DummyTTSManager()):

    while True:
        print("Waiting for transcription...", queue.qsize())
        text = await queue.get()
        print("Human: ", text)
        reply = await send_text_to_LLM(text, assistant, thread)
        print("LLM: ", reply)
        await tts_service.speak(reply)


def transcription_callback(queue, loop, text):
    # Instead of directly invoking asyncio operations, we now schedule them
    loop.call_soon_threadsafe(lambda: queue.put_nowait(text))


def get_stt(queue, loop, args):
    if args.stt == "assembly":
        from transcription.assemblyai_transcription import AssemblyAITranscription

        return AssemblyAITranscription(
            transcript_callback=lambda text: transcription_callback(queue, loop, text),
        )
    elif args.stt == "deepgram":
        print("Deepgram TTS not implemented yet")
    elif args.stt == "whisper":
        print("Whisper TTS not implemented yet")
        from transcription.local_whisper_transcription import LocalWhisperTranscription
        return LocalWhisperTranscription(
            host=args.host,
            port=args.port,
            model_size=args.model_size,
            is_multilingual=False,
            lang="en",
            translate=False,
            callback=lambda text: transcription_callback(queue, loop, text),
        )
    print("Invalid TTS service parameter, defaulting to text input")
    from transcription.text_input_transcription import TextInputTranscription
    return TextInputTranscription(
        transcript_callback=lambda text: transcription_callback(queue, loop, text),
    )

def get_tts(args):
    if args.testing or args.no_voice:
        return DummyTTSManager()  
    elif args.tts == "elevenlabs":
        return TTSManager()

    return DummyTTSManager()


async def main(args):

    # Start the async task to process transcription texts
    # asyncio.create_task(process_transcriptions())
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Parse the command line arguments for online services
    stt_service = get_stt(queue, loop, args)
    tts_service = get_tts(args)

    # Start the transcription client in its own thread
    stt_thread = threading.Thread(target=stt_service.start)
    stt_thread.start()
        
    assistant = await create_assistant()
    thread = await create_thread()
    
    await process_transcriptions(queue, assistant, thread, tts_service=tts_service)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversation client")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", "-p", type=int, default=9090)
    parser.add_argument("--model-size", type=str, default="medium", choices=["small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--stt", type=str, default="assembly", choices=["assembly", "deepgram", "whisper", "text"])
    parser.add_argument("--llm", type=str, default="openai", choices=["openai"])
    parser.add_argument("--tts", type=str, default="elevenlabs", choices=["elevenlabs"])
    args = parser.parse_args()

    if args.testing:
        TESTING = True

    asyncio.run(main(args))
