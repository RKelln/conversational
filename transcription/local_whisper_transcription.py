import string

from wtpsplit import WtP
from whisper_live.client import TranscriptionClient

# TODO: Implement this module


class LocalWhisperTranscription(TranscriptionClient):
    def __init__(self,
                 transcript_callback=None, # func to call with final transcript
                 callbacks: dict = None):  # callbacks for various events in transcriber):
        super().__init__(transcript_callback=transcript_callback, callbacks=callbacks)
        
        self.sentence = "" # for the current sentence (finalized at end of sentence)
        self.current_transcript = "" # for the current utterance
        self.full_transcript = "" # all utterances
    
    def start(self):
        while True:
            text = input("> ")
            if text != "":
                self.sentence = text
                self.current_transcript += " " + text
                self.full_transcript += " " + text
                self.transcript_callback(text)


"""
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
"""