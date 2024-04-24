import logging

class TranscriptionService:
    def __init__(self, 
                transcript_callback=None, # func to call with final transcript
                callbacks: dict = None):  # callbacks for various events in transcriber
        self.logger = logging.getLogger(__name__)
        self.transcript_callback = transcript_callback if transcript_callback is not None else print
        
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
    def current_transcript(self):
        return ""
    
    # full transcript (multiple utterances), may be incomplete 
    def full_transcript(self):
        return ""

    # overridable callbacks
    def on_open(self, open, **kwargs):
        if open is None:
            return
        self.logger.debug(f"open: {open}")

    def on_message(self, result, **kwargs):
        self.logger.debug(f"message: {result}")

    def on_metadata(self, metadata, **kwargs):
        self.logger.debug(f"metadata: {metadata}")

    def on_speech_started(self, speech_started, **kwargs):
        self.logger.debug(f"speech started: {speech_started}")

    def on_utterance_end(self, utterance_end, **kwargs):
        self.logger.debug(f"utternace_end: {utterance_end}")

    def on_error(self, error, **kwargs):
        self.logger.debug(f"error: {error}")

    def on_close(self, close, **kwargs):
        self.logger.debug(f"close: {close}")
