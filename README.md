# Voice Assistant using Whisper, ChatGPT, and gTTS

## Task
Develop a Python program that:
1. **Records audio** from the user.
2. **Converts speech to text** using Whisper.
3. **Sends the text to ChatGPT API** and receives a response.
4. **Converts the response to speech** using gTTS.
5. **Plays the generated audio response.**

## Required Libraries
Install the following dependencies before running the script:
```sh
pip install sounddevice numpy whisper scipy openai gtts playsound requests
```

## Code Implementation
```python
import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
import os
import openai
from gtts import gTTS
from playsound import playsound
import requests

samplerate = 16000  
channels = 1        
duration = 5        
filename = "recorded_audio.wav"

model = whisper.load_model("base")

def record_audio(filename, duration, samplerate):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()  
    write(filename, samplerate, audio)
    print("Recording saved")

record_audio(filename, duration, samplerate)

if os.path.exists(filename):
    print("File found")
else:
    print("File not found")

result = model.transcribe(filename)
text = result["text"]
print("Extracted text:", text)

API_KEY = "your_openai_api_key_here"
url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": text}]
}

try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    chatbot_response = response.json()["choices"][0]["message"]["content"]
    print("ChatGPT Response:", chatbot_response)
except requests.exceptions.RequestException as e:
    print("Error:", e)

tts = gTTS(text=chatbot_response, lang="ar")
response_audio = "response_audio.mp3"
tts.save(response_audio)

print("Playing response...")
playsound(response_audio)
```

