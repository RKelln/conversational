# Conversational AI

Very early experiments with conversational AI in python. Pipeline: Speech-to-text -> text sent to LLM -> LLM text response sent to text-to-speech.

The code is ugly and barely functional and subject to large breaking changes. I've only started making it generic enough to make it easier to plugin new services.


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

- streaming text from LLM straight to TTS streaming to reduce lag
- flag to stop listening wen speaking
- detect voice while LLM speaking to stop and listen
- litellm for LLM management

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


# Research

- prevent output from interfering with speech-2-text
  - Acoustic Echo Cancellation
    - https://github.com/topics/acoustic-echo-cancellation 
  - https://github.com/fjiang9/NKF-AEC


