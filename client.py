import argparse
import asyncio
import string
import threading

from conversation import Conversation
from api.openai_api_async import (create_assistant, create_thread, add_message_to_thread, get_answer)

from TTSManager import TTSManager, DummyTTSManager

from whisper_live.client import TranscriptionClient
from elevenlabs import generate, stream
from wtpsplit import WtP

TESTING = True

wtp = WtP("wtp-canine-s-1l-no-adapters")
WTP_SENTENCE_THRESHOLD = 0.0002
SHORT_WORDS = ["yes", "no", "ok", "okay", "sure", "what", "yeah", "hi", "hello", "bye", "thanks"]
THINKING_WORDS = ["hmm", "uh", "um", "huh", "hmm", "hmmm", "uhh", "uhm", "uhhh", "uhmmm", "ahem"]
NO_PUNCTUATION = str.maketrans('', '', string.punctuation)
SILENCE_TIMEOUT = 2 # seconds

async def send_text_to_LLM(text, assistant, thread):
    if TESTING:
        return "I am a test response"
    await add_message_to_thread(thread.id, text)
    reply = await get_answer(assistant.id, thread)
    return reply


async def process_transcriptions(queue, assistant, thread, tts_manager=DummyTTSManager()):

    waiting_for_full_sentence = False
    last_text = ""
    silence_timeout_task = None

    while True:
        print("Waiting for transcription...", queue.qsize(), waiting_for_full_sentence)
        text = await queue.get()
        print("Received transcription: ", text)
        if text == "" and waiting_for_full_sentence:
            print("Silence timeout while waiting for sentence to complete")
            # don't process the text if it is just an ignored word
            if is_ignored_words(simplify_text(text)):
                continue
            text = last_text
        else: 
            waiting_for_full_sentence = False
            if silence_timeout_task:
                silence_timeout_task.cancel()
                silence_timeout_task = None
        
            last_text = text

            # check for full sentence
            if not waiting_for_full_sentence and not is_full_sentence(text):
                waiting_for_full_sentence = True
                # in 2 seconds, if we don't get a full sentence, we'll send an empty string to the queue
                asyncio.create_task(asyncio.sleep(SILENCE_TIMEOUT)).add_done_callback(lambda _: queue.put_nowait(""))
                continue
        
        # clean up silence timeout
        if silence_timeout_task:
                silence_timeout_task.cancel()
                silence_timeout_task = None
        waiting_for_full_sentence = False

        print("Human: ", text)
        reply = await send_text_to_LLM(text, assistant, thread)
        print("Spirit: ", reply)
        await tts_manager.speak(reply)


def remove_punctuation(text):
    return text.translate(NO_PUNCTUATION)


def is_ignored_words(text):
    if text in THINKING_WORDS:
        print(f"Thinking word: {simple_text}")
        return False


def simplify_text(text):
    return remove_punctuation(text.strip().lower())


def is_full_sentence(text):
    text = text.strip()
    simple_text = simplify_text(text)
    word_count = len(simple_text.split())

    # too short
    if len(text) <= 1:
        return False   
    # doesn't end in ellipse
    if text.endswith("..."):
        return False
    # thinking words only
    if is_ignored_words(simple_text):
        return False
    
    threshold = WTP_SENTENCE_THRESHOLD
    if simple_text in SHORT_WORDS:
        print(f"Short word: {simple_text}")
        return True
    # short sentence
    if simple_text not in SHORT_WORDS and (len(simple_text) < 10 or word_count < 3):
        threshold *= 1.2
    # ends in punctuation
    if text[-1] in [".", "!", "?"]: 
        threshold *= 0.75
        if word_count > 4: # ends and is long
            threshold *= 0.5
    # check for full sentence using WtP
    # it doesn't handle single senences well, but we can check that last character to see if the probably
    # of being the end of a sentence is high
    prob = wtp.predict_proba(text)
    print(max(prob), prob[-3:])
    print(f"Sentence prob: {prob[-1]} ( > {threshold}) : {prob[-1] > threshold}")
    return max(prob) > threshold


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
                                     kwargs={"host": args.host, "port": args.port, "model_size": args.model_size}) 
    client_thread.start()

    assistant = await create_assistant()
    thread = await create_thread()
    
    if args.no_voice:
        tts_manager = DummyTTSManager()
    else:
        tts_manager = TTSManager()
    
    await process_transcriptions(queue, assistant, thread, tts_manager=tts_manager)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sohkepayin Live Client")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "medium", "large-v2", "large-v3"])
    parser.add_argument("--no-voice", action="store_true")

    args = parser.parse_args()

    asyncio.run(main(args))
