import argparse
import os
import time

from api.openai_elevenlabs_api import openai_to_elevenlabs

from TTSManager import TTSManager, DummyTTSManager

from openai import OpenAI
from elevenlabs.client import ElevenLabs
import assemblyai as aai
import pyaudio

TESTING = False


class MicrophoneStream:
    def __init__(
        self,
        sample_rate: int = 44_100,
        idle_threshold: float = 0.0, # seconds of idling before idle callback
        idle_callback=None,
    ):
        """
        Creates a stream of audio from the microphone.

        Args:
            chunk_size: The size of each chunk of audio to read from the microphone. TODO
            channels: The number of channels to record audio from. TODO 
            sample_rate: The sample rate to record audio at.
        """
        self._pyaudio = pyaudio.PyAudio()
        self.sample_rate = sample_rate

        self._chunk_size = int(self.sample_rate * 0.1)
        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
        )

        self._open = True
        self._last_start_time = time.time()
        self.idle_threshold = idle_threshold
        self.idle_callback = idle_callback
        

    def __iter__(self):
        return self

    def __next__(self):
        """
        Reads a chunk of audio from the microphone.
        """
        #print(".", end="", flush=True)

        if not self._open:
            if self._stream.is_active():
                self._stream.stop_stream()
            raise StopIteration

        if self.idle_threshold > 0 and self._last_start_time is not None:
            if time.time() - self._last_start_time > self.idle_threshold:
                self._last_start_time = None
                if self.idle_callback:
                    self.idle_callback()

        try:
            return self._stream.read(self._chunk_size)
        except KeyboardInterrupt:
            raise StopIteration

    def stop(self):
        self._open = False
        self._last_start_time = None
        # FIXME: don't stop the stream as we need to wait for the _next_() instead
        # if self._stream.is_active():
        #     print("stop mic")
        #     self._stream.stop_stream()

    def start(self):
        self._open = True
        self._last_start_time = time.time()
        if self._stream.is_stopped():
            self._stream.start_stream()

    def close(self):
        """
        Closes the stream.
        """

        self._open = False

        if self._stream.is_active():
            self._stream.stop_stream()

        self._stream.close()
        self._pyaudio.terminate()


def process(args, on_voice_detected=None, on_idle=None):

    if args.testing:
        openai_client = None
        thread = None
        assistant = None
    else:
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        thread = openai_client.beta.threads.create()
        assistant = openai_client.beta.assistants.retrieve(os.getenv('OPENAI_ASSISTANT_ID'))

    if args.no_voice or args.testing:
        elevenlabs_client = None
    else:
        elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY')) 

    sample_rate = 44_100 # for usb mic
    speech_started = False
    text_input = ""
    microphone_stream = MicrophoneStream(sample_rate=sample_rate, 
                                         idle_threshold=8.0, 
                                         idle_callback=on_idle)
    
    if args.full_sentence:
        from sentence import is_full_sentence

    def on_data(result: aai.RealtimeTranscript):
        nonlocal speech_started, text_input, microphone_stream
        #print(result)
        if not result.text:
            return

        if speech_started is False:
            speech_started = True
            #print("Voice detected")
            if on_voice_detected:
                on_voice_detected()

        if isinstance(result, aai.RealtimeFinalTranscript):
            #print("Final transcript:", result.text)
            text_input += result.text
            speech_started = False
            microphone_stream.stop()
        else:
            print(result.text, end="\r", flush=True)

    def on_error(error: aai.RealtimeError):
        print("An error occured:", error)
        microphone_stream.stop()

    stt_client = aai.RealtimeTranscriber(
        sample_rate=sample_rate,
        end_utterance_silence_threshold=700,
        disable_partial_transcripts=False,
        on_data=on_data,
        on_error=on_error
    )
    stt_client.connect()

    aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

    active = True
    while active:
        print()
        try:
            microphone_stream.start()
            stt_client.stream(microphone_stream)
        except Exception as e:
            print(f"Error with transcription: {e}")
            active = False
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            active = False

        if not active:
            break

        if text_input == "":
            continue

        if args.full_sentence:
            if not is_full_sentence(text_input):
                print("Incomplete sentence: ", text_input)
                openai_to_elevenlabs(openai_client=None, elevenlabs_client=elevenlabs_client, 
                                     thread_id=None, assistant_id=None, text="Hmm?")
                continue
                
        print("Human: ", text_input)
        print("LLM: ", end="")
        # send text to LLM and speak response
        openai_to_elevenlabs(
            openai_client=openai_client,
            elevenlabs_client=elevenlabs_client,
            thread_id=thread.id, 
            assistant_id=assistant.id, 
            text=text_input)
        text_input = ""
    
    microphone_stream.close()
    stt_client.close()
    openai_client.beta.threads.delete(thread.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversation client")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--full-sentence", action="store_true", help="Only respond to full sentences.")
    parser.add_argument("--video", type=str, help="Video file to play")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.testing:
        TESTING = True

    if args.video:
        from python_mpv_jsonipc import MPV
        mpv = MPV()
        mpv.loop = True
        mpv.play(args.video)
        def duck_sound():
            print("Ducking sound")
            mpv.volume = 75
        def unduck_sound():
            print("UNducking sound")
            mpv.volume = 100

    process(args, on_voice_detected=duck_sound, on_idle=unduck_sound)
