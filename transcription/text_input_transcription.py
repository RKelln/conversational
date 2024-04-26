from .transcription_service import TranscriptionService

class TextInputTranscription(TranscriptionService):
    def __init__(self,
                 transcript_callback=None, # func to call with final transcript
                 callbacks: dict = None):  # callbacks for various events in transcriber):
        super().__init__(transcript_callback=transcript_callback, callbacks=callbacks)

        self.sentence = "" # for the current sentence (finalized at end of sentence)
        self.current_transcript = "" # for the current utterance
        self.full_transcript = "" # all utterances

    
    def start(self):
        self.logger.info("Starting text input service")
        while True:
            text = input()
            if text != "":
                self._message_processing(text)


    # current sentence, may be incomplete
    def current_sentence(self):
        return self.current_sentence
    
    # current transcript (multiple sentences in single utterance), may be incomplete
    def current_transcript(self):
        return self.current_transcript
    
    # full transcript (multiple utterances), may be incomplete 
    def full_transcript(self):
        return self.full_transcript