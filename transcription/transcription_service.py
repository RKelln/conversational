import logging

class TranscriptionService:
    def __init__(self, 
                transcript_callback=None, # func to call with final transcript
                callbacks: dict = None):  # callbacks for various events in transcriber
        self.logger = logging.getLogger(__name__)
        self.transcript_callback = transcript_callback if transcript_callback is not None else print
        
        self.sentence = "" # for the current sentence (finalized at end of sentence)
        self.current_transcript = "" # for the current utterance
        self.full_transcript = "" # all utterances

        self.callbacks = {
            "on_open": None,
            "on_message": None,
            "on_metadata": None,
            "on_speech_started": None,
            "on_utterance_end": None,
            "on_error": None,
            "on_close": None
        }

        for callback in self.callbacks:
            # if the passed in callback is True use existing callback
            if callbacks is None or callback not in callbacks or callbacks[callback] is True:
                self.callbacks[callback] = getattr(self, callback)
            # if the callback is a function then use the function
            elif hasattr(callbacks[callback], '__call__'):
                self.callbacks[callback] = callbacks[callback]
                setattr(self, callback, callbacks[callback])

    def start(self):
        self.logger.info("Starting transcription service")

    def stop(self):
        self.logger.info("Stopping transcription service")

    # current sentence, may be incomplete
    def current_sentence(self):
        return ""
    
    # current transcript (multiple sentences in single utterance), may be incomplete
    @staticmethod
    def _join_text(text : str, new_text : str, separator=" ") -> str:
        if text == "":
            return new_text
        return (text + separator + new_text).strip()

    def _message_processing(self, text):
        if text == "":
            return
        self.sentence += text
        if len(self.sentence) > 0:
            self.current_transcript = self._join_text(self.current_transcript, self.sentence)
            self.sentence = ""
            if self.on_utterance_end(self.current_transcript): # if utterance is complete
                self.full_transcript = self._join_text(self.full_transcript, self.current_transcript, separator="\n\n")
                self.transcript_callback(self.current_transcript)
                self.current_transcript = ""


    # overridable callbacks
    def on_open(self, open, **kwargs):
        if open is None:
            return
        self.logger.debug(f"open: {open}")

    def on_message(self, result, **kwargs):
        self.logger.debug(f"message: {result}")

    def on_metadata(self, metadata, **kwargs):
        self.logger.debug(f"metadata: {metadata}")

    def on_speech_started(self, **kwargs):
        self.logger.debug(f"speech started")

    # return true if the utterance is complete
    def on_utterance_end(self, utterance_end, **kwargs) -> bool:
        self.logger.debug(f"utternace_end: {utterance_end}")
        return True

    def on_error(self, error, **kwargs):
        self.logger.debug(f"error: {error}")

    def on_close(self, close, **kwargs):
        self.logger.debug(f"close: {close}")
