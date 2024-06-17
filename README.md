# Abyss of Time
## Conversational AI for robotic vacuum cleaner of the future

Original conversational code by @RKelln
Adapted for performance by @mfbergmann

This code is very specific to its application for an installation piece, and is very messy at the moment.

---

# Install

1. Set up virtual environment
   1. `python -m venv .venv`
   2. `source .venv/bin/activate`
2. Install packages:
   1. For Assembly AI:
      1.Make sure to install `apt install portaudio19-dev` (Debian/Ubuntu) or
        `brew install portaudio` (MacOS) before installing
   2. For online services only: (assembly, openai, elevenlabs)
      1.  `pip -r requirements.txt`
  1.  For whisper local:
      1.  `pip -r requirements_whisper.txt`
3. Make a copy of `.env.sample` and rename in `.env`
4. Add your API keys to `.env`


# Run

## IMPORTANT

Without very good (hardware) echo cancellation on your mic it will hear the text-to-speech output, making `client.py` pretty useless (for now). I've made a `conversational.py` that uses Assembly, OpenAI, and Elevenlabs and only listens when it is not thinking/speaking.

This supports various options for input and presence detection (webcam). Some of these may require passing the index of the microphone or webcam if you have more than one. You can see the enumerations using `--list-devices`
```
python conversational.py --list-devices
```

There is also support for playing a video (using mpv) during the conversation and displaying the transcription text on the video.

For example the full arguments I used with the Seeed USB speaker and a webcam at a recent prototype installation was:
```
python3 conversational.py --video experimance.mp4 --presence-idle 20 --presence-threshold 1e9 --input-index 1 --input-channels 6 --sample-rate 16000 --webcam 2 --video-volume 0.9 --onscreen-display
```

# Old and broken:

The `client.py` file is an older version that using async but handles turn taking poorly. Don't use it.

## With online services
Run client with online services:
(defaults: assembly.ai for speech-to-text (STT), OpenAI for LLM, elevenlabs for text-to-speech (TTS))
```
python client.py --stt assembly --tts elevenlabs
```

## With local services:
Run local whisper server (with NVIDIA card):
```
python server.py
```

Run client with local whisper running on another machine:
```
python client.py --host 192.168.2.196 --model "large-v3"
```

# TODO

- detect voice while LLM speaking to stop and listen
- litellm / langchain for LLM management

# Tools:

## Deepgram
https://github.com/deepgram/deepgram-python-sdk
```
pip install deepgram-sdk==3.*
```

## Assembly.ai
https://github.com/AssemblyAI/assemblyai-python-sdk
```
pip install -U assemblyai "assemblyai[extras]"
```

## OpenAI 
https://github.com/openai/openai-python
```
pip install openai
```

## Elevenlabs
https://github.com/elevenlabs/elevenlabs-python
```
pip install elevenlabs
```

# WhisperLive

- applied pull:
  -  https://github.com/collabora/WhisperLive/pull/101
  
- for custom WhisperLive (that updates when changing files):
  ```
  cd WhisperLife
  python -m pip install --editable .
  ```


# wtpsplit
For sentence detection.
https://github.com/bminixhofer/wtpsplit
```
wtpsplit
```

# Research

- prevent output from interfering with speech-2-text
  - Acoustic Echo Cancellation
    - https://github.com/topics/acoustic-echo-cancellation 
  - https://github.com/fjiang9/NKF-AEC
- https://github.com/gkamradt/QuickAgent
  - Deepgram + groq
- https://github.com/deepgram-devs/deepgram-conversational-demo
- 

