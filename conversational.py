import argparse
import os
import time
import textwrap

from api.openai_elevenlabs_api import openai_to_elevenlabs

from openai import OpenAI
from elevenlabs.client import ElevenLabs
import assemblyai as aai
import pyaudio
import numpy as np

from webcam import detect_face_in_image, capture_image_from_webcam

TESTING = False


class MicrophoneStream:
    MIN_FREQ = 100  # Minimum frequency to consider (Hz)
    MAX_FREQ = 3000  # Maximum frequency to consider (Hz)

    # for respeaker details see:
    # https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/#extract-voice

    def __init__(
        self,
        input_index : int = 0,
        sample_rate: int = 44_100,
        chunk_size: int = 2048,
        gain: float = 1.0,
        channels: int = 1,
        idle_threshold: float = 0.0, # seconds of idling before idle callback
        idle_callback=None,
        energy_threshold: float = 2e6,
        energy_callback=None,
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
        self.gain = gain

        #self._chunk_size = int(self.sample_rate * 0.1)
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

        self._open = True
        self._last_idle_at = time.monotonic()
        self.idle_threshold = idle_threshold
        self.idle_callback = idle_callback
        
        self.energy_threshold = energy_threshold
        self.energy_callback = energy_callback

        print("Microphone stream created")
        print("  Input index:      ", input_index)
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
        """
        Reads a chunk of audio from the microphone.
        """
        #print(".", end="", flush=True)

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
            data = self._stream.read(self._chunk_size * self._channels, exception_on_overflow = False)
            #data = self._stream.read(self._chunk_size * self._channels)
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
            # for 6 channel respeaker, extract channel 0, the processed audio
            audio_data = np.frombuffer(data, dtype=np.int16)[0::self._channels]
        else:
            audio_data = np.frombuffer(data, dtype=np.int16)

        if self.gain != 1.0:
            # Apply gain
            audio_data = np.clip(audio_data * self.gain, -32768, 32767)  # Ensure values stay within int16 range

        if self.energy_threshold > 0 and self.energy_callback:
            # Apply FFT to get frequency spectrum
            fft_data = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft_data), 1.0/self.sample_rate)
            # Find indices corresponding to frequency range of interest
            idx_min = np.where(freqs >= self.MIN_FREQ)[0][0]
            idx_max = np.where(freqs <= self.MAX_FREQ)[0][-1]

            # Calculate energy within the frequency range
            energy = np.sum(np.abs(fft_data[idx_min:idx_max]))

            # remap energy to a meter 0-20
            energy_meter = 20 * (energy - self.energy_threshold) / (10 * self.energy_threshold - self.energy_threshold)
            energy_meter = int(max(0, min(20, energy_meter)))# ensure the value is between 0 and 20
            #print("█" * energy_meter, " " * (20 - energy_meter), end="\r", flush=True)

            if energy > self.energy_threshold and self.energy_callback:
                self._last_idle_at = time.monotonic()
                self.energy_callback(energy)
        
        return audio_data.tobytes()
            
    def stop(self):
        if not self._open: return

        self._open = False
        self._last_idle_at = None
        # FIXME: don't stop the stream as we need to wait for the _next_() instead
        # if self._stream.is_active():
        #     print("stop mic")
        #     self._stream.stop_stream()

    def start(self):
        if self._open: return

        self._open = True
        self._last_idle_at = time.monotonic()
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


def add_text_overlay(mpv, text, wrap=50):
    """ Adds a text overlay to the video for a specified duration. """
    # Create an overlay ID (can be any arbitrary unique number)
    overlay_id = 1

    # Wrap the text at the specified width
    wrapped_text = "\n".join(textwrap.wrap(text, width=wrap))
    
    # Format the wrapped text for ASS
    formatted_text = "\\N".join(wrapped_text.split('\n'))  # Use ASS newline

    # Set text attributes: 
    # font size (e.g., \\fs40)
    # Font Boldness: Adjust the boldness using \\b1 for bold and \\b0 for normal.
    # Font Color: Change the color using \\1c&HRRGGBB& where RR, GG, BB are the red, green, and blue color components in hexadecimal format.
    # See: https://fileformats.fandom.com/wiki/SubStation_Alpha
    data = f"{{\\an5\\fs28\\bord2\\shad0\\1c&HFFFFFF&\\3c&H000000&}}{formatted_text}"
    
    # Command to add the text overlay
    try:
        mpv.osd_overlay(overlay_id, "ass-events", data)
    except Exception as e:
        print(f"Failed to add overlay with ID {overlay_id}: {e}")
    return overlay_id


def remove_text_overlay(mpv, overlay_id):
    """ Removes a text overlay from the video. """
    # Command to remove the text overlay
    try:
        # Command to remove the text overlay
        mpv.osd_overlay(overlay_id, "ass-events", "")
    except Exception as e:
        print(f"Failed to remove overlay with ID {overlay_id}: {e}")


def process(args, mpv = None, on_voice_detected=None, on_idle=None):

    if args.testing:
        openai_client = None
        # create mocks for thread and assistant
        thread = lambda:None
        thread.id = None
        assistant = lambda:None
        assistant.id = None
    else:
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        thread = openai_client.beta.threads.create()
        assistant = openai_client.beta.assistants.retrieve(os.getenv('OPENAI_ASSISTANT_ID'))

    if args.no_voice or args.testing:
        elevenlabs_client = None
    else:
        elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY')) 

    #sample_rate = 44_100 # for usb mic
    speech_started = False
    text_input = ""
    last_presence_at = time.monotonic()
    has_presence = False
    image_presence = args.webcam >= 0
    prev_incomplete_sentence = None

    def start_conversation():
        nonlocal openai_client, thread
        if openai_client:
            print("Starting new conversation")
            #openai_client.beta.threads.delete(thread.id)
            thread = openai_client.beta.threads.create()

    def no_presence_detected():
        nonlocal has_presence, stt_client, openai_client, mpv, args, thread, stt_client_connected
        print(f"No presence detected for {args.presence_idle}s, restarting conversation")
        has_presence = False
        # close connection to STT
        if stt_client:
            stt_client.close()
            stt_client_connected = False
        # end the conversation and start a new one
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

    def on_idle_wrapper():
        nonlocal on_idle, image_presence, has_presence, last_presence_at, args
        if on_idle:
            on_idle()
        if image_presence:
            print("Checking visual presence: ", end="")
            # capture image from webcam
            image = capture_image_from_webcam(webcam_index=args.webcam, num_frames=3)
            if image is None: return
            faces = detect_face_in_image(image, debug=True)
            print("Faces detected:", faces)
            if faces > 0:
                presence_detected()
            elif has_presence and args.presence_idle > 0 and time.monotonic() - last_presence_at > args.presence_idle:
                print("No face detected, assuming no presence")
                no_presence_detected()

    def audio_presence_detected(energy):
        nonlocal last_presence_at, speech_started, has_presence
        #print(f"Audio presence: {energy:.0}", end="\r", flush=True)
        presence_detected()

        if energy > args.presence_threshold * 2.0 and speech_started is False:
            speech_started = True
            if on_voice_detected:
                on_voice_detected()

    if args.full_sentence:
        # this import takes awhile, ensure it is done before mic stream starts
        from sentence import is_full_sentence

    microphone_stream = MicrophoneStream(input_index=args.input_index,
                                         sample_rate=args.sample_rate, 
                                         channels=args.input_channels,
                                         idle_threshold=args.idle_threshold, 
                                         idle_callback=on_idle_wrapper,
                                         energy_callback=audio_presence_detected,
                                         energy_threshold=args.presence_threshold,
                                         )

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
        print("An transcription error occured:", error)
        microphone_stream.stop()

    stt_client_connected = False

    def create_stt_client():
        nonlocal stt_client_connected
        if stt_client_connected:
            stt_client.close()
        stt_client = aai.RealtimeTranscriber(
            sample_rate=args.sample_rate,
            end_utterance_silence_threshold=700,
            disable_partial_transcripts=(args.presence_threshold <= 0), # unncessary if presence detection is used
            on_data=on_data,
            on_error=on_error
        )
        stt_client.connect()
        stt_client_connected = True
        return stt_client
    
    stt_client = create_stt_client()

    aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

    active = True
    end_conversation = False # whether to end the conversation at the end of the while loop
    while active:
        try:
            now = time.monotonic()

            # check presence, if it has been a long time assume a new conversation
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
            stt_client.stream(microphone_stream) # blocks until utterance is complete, set text_input
        except Exception as e:
            print(f"Error with transcription: {e}")
            active = False
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            active = False

        if not active:
            break

        # text_input = text_input.strip()

        # # handle LLM deciding no repsonse is necessary
        # if text_input.find("<no response>") >= 0:
        #     print("no response")
        #     text_input = ""

        if text_input == "":
            #print("No text input")
            continue

        # handle full sentence mode
        if args.full_sentence:
            if not is_full_sentence(text_input):
                print("Incomplete sentence: ", text_input)
                # only do one round of incomplete sentence handling
                if prev_incomplete_sentence == text_input:
                    text_input = ""
                    prev_incomplete_sentence = None
                else:
                    prev_incomplete_sentence = text_input
                    openai_to_elevenlabs(openai_client=None, elevenlabs_client=elevenlabs_client, 
                                        thread_id=None, assistant_id=None, text="Hmm?")
                continue

        print("Human: ", text_input)
        if mpv and args.onscreen_display:
            mpv_overlay_id = add_text_overlay(mpv, text_input, wrap=40)
        print("LLM: ", end="")

        # send text to LLM and speak response (blocking)
        try:
            output = openai_to_elevenlabs(
                openai_client=openai_client,
                elevenlabs_client=elevenlabs_client,
                thread_id=thread.id, 
                assistant_id=assistant.id, 
                text=text_input)
        except Exception as e:
            print("Error with LLM: ", e)
            
        if mpv and args.onscreen_display:
            remove_text_overlay(mpv, mpv_overlay_id)

        # handle and remove <end> tags:
        if output and output.endswith("<end>"):
            print("Ending conversation")
            end_conversation = True

        # HACK: update presence timer to avoid LLM carrying on
        last_presence_at = time.monotonic()
        
        # reset for next loop
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
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--input-index", type=int, default=0, help="Input index for microphone")
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

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.testing:
        TESTING = True

    if args.list_devices:
        # list webcames
        import subprocess
        result = subprocess.Popen(["v4l2-ctl", "--list-devices"], stdout=subprocess.PIPE)
        print(result.stdout.read().decode())

    if args.input_index < 0 or args.list_devices:
        # print input devices
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        exit()

    if args.video:
        from python_mpv_jsonipc import MPV
        mpv = MPV()
        mpv.loop = True
        mpv.fs_screen = 2
        mpv.fullscreen = True
        mpv.play(args.video)
        low_volume = int(75 * args.video_volume)
        normal_volume = int(100 * args.video_volume)
        def duck_sound():
            #print("Ducking sound")
            mpv.volume = low_volume
        def unduck_sound():
            #print("UNducking sound")
            mpv.volume = normal_volume

    process(args, mpv, on_voice_detected=duck_sound, on_idle=unduck_sound)

    if args.video:
        if mpv and mpv.mpv_process:
            mpv.quit()

