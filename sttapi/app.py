import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
from convert import audio_transcription_mp3


app = FastAPI(title='Speech to text API',
              description='Transcribing your speech to text')


@app.get("/", response_class=PlainTextResponse)
def home():
    return "Welcome! API is working perfectly well. Use /docs to proceed to transcribe your speech to text."


@app.post("/Speech_to_text(mp3/wav files)")
async def speech_to_text(file: UploadFile = File(...)) -> str:
    try:
        files = await file.read()
        filename = "podcast.wav"
        with open(filename, "wb+") as f:
            f.write(files)
        output = audio_transcription_mp3(filename)
        if os.path.exists(filename):
            os.remove(filename)
        return output
    except Exception as e:
        return "Please input mp3/wav file only"

