import argparse
import os
import time
import textwrap
import pyaudio

from api.openai_elevenlabs_api import openai_to_elevenlabs
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from identifydevices import list_audio_devices
import assemblyai as aai
import numpy as np

# from webcam import detect_face_in_image, capture_image_from_webcam

TESTING = False

class MicrophoneStream:
    MIN_FREQ = 100  # Minimum frequency to consider (Hz)
    MAX_FREQ = 3000  # Maximum frequency to consider (Hz)

    def __init__(
        self,
        input_index: int = 0,
        output_index: int = 0,
        sample_rate: int = 44_100,
        chunk_size: int = 2048,
        gain: float = 1.0,
        channels: int = 1,
        idle_threshold: float = 0.0,  # seconds of idling before idle callback
        idle_callback=None,
        energy_threshold: float = 2e6,
        energy_callback=None,
    ):
        self._pyaudio = pyaudio.PyAudio()

        # Get input device info
        input_device_info = self._pyaudio.get_device_info_by_host_api_device_index(0, input_index)
        max_input_channels = input_device_info.get('maxInputChannels')

        if channels > max_input_channels:
            raise ValueError(f"Invalid number of channels: {channels}. Device supports up to {max_input_channels} channels.")

        # Get output device info
        output_device_info = self._pyaudio.get_device_info_by_host_api_device_index(0, output_index)
        max_output_channels = output_device_info.get('maxOutputChannels')

        self.sample_rate = sample_rate
        self.gain = gain
        self._chunk_size = chunk_size
        self._channels = channels

        self._stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            input_device_index=input_index,
        )

        self._output_stream = self._pyaudio.open(
            format=pyaudio.paInt16,
            channels=max_output_channels,
            rate=sample_rate,
            output=True,
            output_device_index=output_index,
        )

        self._open = True
        self._last_idle_at = time.monotonic()
        self.idle_threshold = idle_threshold
        self.idle_callback = idle_callback

        self.energy_threshold = energy_threshold
        self.energy_callback = energy_callback

        print("Microphone stream created")
        print("  Input index:      ", input_index)
        print("  Output index:     ", output_index)
        print("  Sample rate:      ", self.sample_rate)
        print("  Chunk size:       ", self._chunk_size)
        print("  Energy threshold: ", self.energy_threshold)
        print("  Energy callback:  ", self.energy_callback is not None)
        print("  Idle threshold:   ", self.idle_threshold)
        print("  Idle callback:    ", self.idle_callback is not None)
        print("  Channels:         ", self._channels)
        print("  Gain:             ", self.gain)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._open:
            if self._stream.is_active():
                self._stream.stop_stream()
            raise StopIteration

        if self.idle_threshold > 0 and self._last_idle_at is not None:
            now = time.monotonic()
            if now - self._last_idle_at > self.idle_threshold:
                print("Idle")
                self._last_idle_at = now
                if self.idle_callback:
                    self.idle_callback()
                    raise StopIteration

        if not self._stream.is_active():
            self.stop()
            raise StopIteration

        try:
            data = self._stream.read(self._chunk_size * self._channels, exception_on_overflow=False)
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                print("Input overflow, inserting silence")
            data = ('\x00' * self._chunk_size).encode()
        except KeyboardInterrupt:
            print("Keyboard interrupt in mic stream")
            self.close()
            raise StopIteration
        except Exception as e:
            print("Error in mic stream: ", e)
            self.close()
            raise StopIteration

        if self._channels > 1:
            audio_data = np.frombuffer(data, dtype=np.int16)[0::self._channels]
        else:
            audio_data = np.frombuffer(data, dtype=np.int16)

        if self.gain != 1.0:
            audio_data = np.clip(audio_data * self.gain, -32768, 32767)  # Ensure values stay within int16 range

        if self.energy_threshold > 0 and self.energy_callback:
            fft_data = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft_data), 1.0 / self.sample_rate)
            idx_min = np.where(freqs >= self.MIN_FREQ)[0][0]
            idx_max = np.where(freqs <= self.MAX_FREQ)[0][-1]

            energy = np.sum(np.abs(fft_data[idx_min:idx_max]))

            energy_meter = 20 * (energy - self.energy_threshold) / (10 * self.energy_threshold - self.energy_threshold)
            energy_meter = int(max(0, min(20, energy_meter)))

            if energy > self.energy_threshold and self.energy_callback:
                self._last_idle_at = time.monotonic()
                self.energy_callback(energy)

        self._output_stream.write(data)
        return audio_data.tobytes()

    def stop(self):
        if not self._open: return

        self._open = False
        self._last_idle_at = None

    def start(self):
        if self._open: return

        self._open = True
        self._last_idle_at = time.monotonic()
        if self._stream.is_stopped():
            self._stream.start_stream()

    def close(self):
        self._open = False

        if self._stream.is_active():
            self._stream.stop_stream()

        if self._output_stream.is_active():
            self._output_stream.stop_stream()

        self._stream.close()
        self._output_stream.close()
        self._pyaudio.terminate()


def process(args, mpv=None, on_voice_detected=None, on_idle=None):
    if args.testing:
        openai_client = None
        thread = lambda: None
        thread.id = None
        assistant = lambda: None
        assistant.id = None
    else:
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        thread = openai_client.beta.threads.create()
        assistant = openai_client.beta.assistants.retrieve(os.getenv('OPENAI_ASSISTANT_ID'))

    if args.no_voice or args.testing:
        elevenlabs_client = None
    else:
        elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

    speech_started = False
    text_input = ""
    last_presence_at = time.monotonic()
    has_presence = False
    prev_incomplete_sentence = None

    def start_conversation():
        nonlocal openai_client, thread
        if openai_client:
            print("Starting new conversation")
            thread = openai_client.beta.threads.create()

    def no_presence_detected():
        nonlocal has_presence, stt_client, openai_client, mpv, args, thread, stt_client_connected
        print(f"No presence detected for {args.presence_idle}s, restarting conversation")
        has_presence = False
        if stt_client:
            stt_client.close()
            stt_client_connected = False
        start_conversation()
        if mpv:
            mpv.speed = 0.2

    def presence_detected():
        nonlocal has_presence, last_presence_at, stt_client, stt_client_connected, mpv
        last_presence_at = time.monotonic()
        if has_presence: return
        print("Presence detected")
        has_presence = True
        if mpv:
            mpv.speed = 1.0
        if stt_client and not stt_client_connected:
            print("Reconnecting to STT")
            stt_client = create_stt_client()

    # def on_idle_wrapper():
    #     nonlocal on_idle, image_presence, has_presence, last_presence_at, args
    #     if on_idle:
    #         on_idle()
    #     if image_presence:
    #         print("Checking visual presence: ", end="")
    #         image = capture_image_from_webcam(webcam_index=args.webcam, num_frames=3) # type: ignore
    #         if image is None: return
    #         faces = detect_face_in_image(image, debug=True)
    #         print("Faces detected:", faces)
    #         if faces > 0:
    #             presence_detected()
    #         elif has_presence and args.presence_idle > 0 and time.monotonic() - last_presence_at > args.presence_idle:
    #             print("No face detected, assuming no presence")
    #             no_presence_detected()

    def audio_presence_detected(energy):
        nonlocal last_presence_at, speech_started, has_presence
        presence_detected()

        if energy > args.presence_threshold * 2.0 and speech_started is False:
            speech_started = True
            if on_voice_detected:
                on_voice_detected()

    if args.full_sentence:
        from sentence import is_full_sentence

    microphone_stream = MicrophoneStream(
        input_index=args.input_index,
        output_index=args.output_index,
        sample_rate=args.sample_rate,
        channels=args.input_channels,
        idle_threshold=args.idle_threshold,
        # idle_callback=on_idle_wrapper,
        energy_callback=audio_presence_detected,
        energy_threshold=args.presence_threshold,
    )

    def on_data(result: aai.RealtimeTranscript):
        nonlocal speech_started, text_input, microphone_stream
        if not result.text:
            return

        if speech_started is False:
            speech_started = True
            if on_voice_detected:
                on_voice_detected()

        if isinstance(result, aai.RealtimeFinalTranscript):
            text_input += result.text
            speech_started = False
            microphone_stream.stop()
        else:
            print(result.text, end="\r", flush=True)

    def on_error(error: aai.RealtimeError):
        print("A transcription error occurred:", error)
        microphone_stream.stop()

    stt_client_connected = False

    def create_stt_client():
        nonlocal stt_client_connected
        if stt_client_connected:
            stt_client.close()
        stt_client = aai.RealtimeTranscriber(
            sample_rate=args.sample_rate,
            end_utterance_silence_threshold=700,
            disable_partial_transcripts=(args.presence_threshold <= 0),
            on_data=on_data,
            on_error=on_error
        )
        stt_client.connect()
        stt_client_connected = True
        return stt_client

    stt_client = create_stt_client()
    aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

    active = True
    end_conversation = False
    while active:
        try:
            now = time.monotonic()

            if has_presence and args.presence_idle > 0 and now - last_presence_at > args.presence_idle:
                print(f"No presence detected for {args.presence_idle}s, restarting conversation")
                no_presence_detected()

            if not has_presence:
                print("No presence detected, waiting for presence")
                microphone_stream.start()
                for _ in microphone_stream:
                    if has_presence or speech_started:
                        presence_detected()
                        break
                print("End no presence loop")
                continue

            print(f"\ntranscribing: {(now - last_presence_at):.1}", flush=True)
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
                if prev_incomplete_sentence == text_input:
                    text_input = ""
                    prev_incomplete_sentence = None
                else:
                    prev_incomplete_sentence = text_input
                    openai_to_elevenlabs(openai_client=None, elevenlabs_client=elevenlabs_client,
                                         thread_id=None, assistant_id=None, text="Hmm?")
                continue

        print("Human: ", text_input)
#        if mpv and args.onscreen_display:
#            mpv_overlay_id = add_text_overlay(mpv, text_input, wrap=40)
        print("LLM: ", end="")

        try:
            output = openai_to_elevenlabs(
                openai_client=openai_client,
                elevenlabs_client=elevenlabs_client,
                thread_id=thread.id,
                assistant_id=assistant.id,
                text=text_input)
        except Exception as e:
            print("Error with LLM: ", e)

#        if mpv and args.onscreen_display:
#            remove_text_overlay(mpv, mpv_overlay_id)

        if output and output.endswith("<end>"):
            print("Ending conversation")
            end_conversation = True

        last_presence_at = time.monotonic()

        if end_conversation:
            end_conversation = False
            start_conversation()
        text_input = ""
        speech_started = False

    print("Closing")
    microphone_stream.stop()
    microphone_stream.close()
    if stt_client and stt_client_connected:
        stt_client.close()
    stt_client_connected = False
    if openai_client:
        openai_client.beta.threads.delete(thread.id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversation client")
    #list_audio_devices()
    #input_device = int(input("Enter the id of your input device: "))
    #output_device = int(input("Enter the id of your output device: "))
    
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--input-index", type=int, default=0, help="Input index for microphone")
    parser.add_argument("--output-index", type=int, default=0, help="Output index for speaker")
    parser.add_argument("--sample-rate", type=int, default=44_100, help="Sample rate for input audio")
    parser.add_argument("--input-gain", type=float, default=1.0, help="Input gain for microphone")
    parser.add_argument("--input-channels", type=int, default=1, help="Number of input channels for microphone")
    parser.add_argument("--full-sentence", action="store_true", help="Only respond to full sentences.")
    parser.add_argument("--video", type=str, help="Video file to play")
    parser.add_argument("--idle-threshold", type=float, default=8.0, help="Seconds of no audio before calling idle callback")
    parser.add_argument("--presence-threshold", type=float, default=4e6, help="Energy threshold for audio presence detection, 0 to disable")
    parser.add_argument("--presence-idle", type=float, default=0.0, help="Seconds of no presence detected before restarting the conversation")
    parser.add_argument("--webcam", type=int, default=-1, help="Use webcam for presence detection, supply index of device (generally 0)")
    parser.add_argument("--video-volume", type=float, default=1.0, help="Scales the video volume")
    parser.add_argument("--onscreen-display", action="store_true", help="Show onscreen display of audio transcript")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    #args.input_index = input_device
    #args.output_index = output_device

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.testing:
        TESTING = True

    if args.list_devices:
        list_audio_devices()
        exit()
