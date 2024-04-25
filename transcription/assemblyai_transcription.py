# Start by making sure the `assemblyai` package is installed.
# If not, you can install it by running the following command:
# pip install -U assemblyai
#
# Then, make sure you have PyAudio installed: https://pypi.org/project/PyAudio/
#
# Note: Some macOS users might need to use `pip3` instead of `pip`.
import os
import assemblyai as aai

from .transcription_service import TranscriptionService

# env vars
from dotenv import load_dotenv
load_dotenv()
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')


class AssemblyAITranscription(TranscriptionService):

    valid_options = [
        # Assembly ai options: https://www.assemblyai.com/docs/speech-to-text/streaming
        "sample_rate", "end_utterance_silence_threshold", "word_boost", "encoding", "disable_partial_transcripts"
    ]

    def __init__(self, 
                 transcript_callback=None,
                 callbacks: dict = {"on_message": True}, 
                 options: dict = None):
        super().__init__(transcript_callback, callbacks)

        self.transcriber = None
        self.sentence = "" # for the current sentence (finalized at end of sentence)
        self.current_transcript = "" # for the current utterance
        self.full_transcript = "" # all utterances

        # extract acceptable options, keep only the keys that are valid
        if options is not None:
            print("Options: ", options)
            options = {k: v for k, v in options.items() if k in AssemblyAITranscription.valid_options}

        # options: https://www.assemblyai.com/docs/speech-to-text/streaming
        self.options = dict(    
            sample_rate=44_100,
            end_utterance_silence_threshold=500,
            disable_partial_transcripts=True,
        )
        if options is not None:
            self.options.update(options)


    def start(self):
        self.logger.info("Starting Deepgram transcription service")

        try:
            on_data = None
            on_error = None
            on_open = None
            on_close = None

            if "on_open" in self.callbacks and self.callbacks["on_open"] is not None:
                on_open = self.on_open
            if "on_message" in self.callbacks and self.callbacks["on_message"] is not None:
                on_data = self.on_message
            if "on_metadata" in self.callbacks and self.callbacks["on_metadata"] is not None:
                pass # TODO
            if "on_speech_started" in self.callbacks and self.callbacks["on_speech_started"] is not None:
                pass # TODO
            if "on_utterance_end" in self.callbacks and self.callbacks["on_utterance_end"] is not None:
                pass # TODO
            if "on_error" in self.callbacks and self.callbacks["on_error"] is not None:
                on_error = self.on_error
            if "on_close" in self.callbacks and self.callbacks["on_close"] is not None:
                on_close = self.on_close

            options = self.options.copy()
            options.update(dict(
              on_data=on_data,
              on_error=on_error,
              on_open=on_open, # optional
              on_close=on_close, # optional
            ))

            self.transcriber = aai.RealtimeTranscriber(**options)

            # Start the connection
            self.transcriber.connect()

            # Open a microphone stream
            microphone_stream = aai.extras.MicrophoneStream()

            # Press CTRL+C to abort
            self.transcriber.stream(microphone_stream)  # waits

        except Exception as e:
            print(f"Error starting Assembly transcription service: {e}")
        finally:
            microphone_stream.close()
            self.stop()


    def stop(self):
        self.logger.info("Stopping Assembly transcription service")
        if self.transcriber:
            self.transcriber.close()


    def on_message(self, result, **kwargs):
        #print(self, result, kwargs)
        if not result.text:
            return

        if isinstance(result, aai.RealtimeFinalTranscript):
            self.sentence += result.text
            self.current_transcript += self.sentence
            self.full_transcript += self.sentence
            if len(self.sentence) > 0:
                self.transcript_callback(self.sentence)
                self.sentence = ''
                self.on_utterance_end(self.current_transcript)
        else:
            print(result.text, end="\r")


    def current_sentence(self):
        return self.sentence
    
    def current_transcript(self):
        return self.current_transcript
    
    def full_transcript(self):
        return self.full_transcript


if __name__ == "__main__":
    # Create a Assembly transcription
    test = AssemblyAITranscriptionService()
    test.start()
    test.stop()