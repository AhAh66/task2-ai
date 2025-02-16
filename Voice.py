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
