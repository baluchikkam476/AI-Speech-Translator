# app1.py
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
from googletrans import Translator
import whisper
from yt_dlp import YoutubeDL

app = Flask(__name__)
CORS(app)

# Load models / clients once
print("Loading Whisper model (this may take a moment)...")
model = whisper.load_model("small")   # "small" is a good balance for CPU
translator = Translator()


def convert_to_wav(input_path):
    """
    Convert any audio file to a wav file using pydub (ffmpeg required).
    Returns path to wav file.
    """
    wav_path = tempfile.mktemp(suffix=".wav")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # ensure mono 16k
    audio.export(wav_path, format="wav")
    return wav_path


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI Speech Translator backend running"}), 200


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided (form key 'audio')"}), 400

    audio_file = request.files["audio"]
    # Save uploaded file to a temp path
    tmp_input = tempfile.mktemp(suffix=os.path.splitext(audio_file.filename)[1] or ".tmp")
    audio_file.save(tmp_input)
    print("Received file:", audio_file.filename, "->", tmp_input)

    # Convert to wav for Whisper
    try:
        tmp_wav = convert_to_wav(tmp_input)
    except Exception as e:
        # attempt to remove tmp_input and return error
        if os.path.exists(tmp_input):
            try: os.remove(tmp_input)
            except: pass
        return jsonify({"error": f"Audio conversion failed: {str(e)}"}), 500

    # Transcribe
    try:
        result = model.transcribe(tmp_wav)
        text = result.get("text", "").strip()
        print("Transcription:", text)
    except Exception as e:
        # cleanup
        for p in (tmp_input, tmp_wav):
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
        return jsonify({"error": f"Whisper failed: {str(e)}"}), 500

    # cleanup
    for p in (tmp_input, tmp_wav):
        if os.path.exists(p):
            try: os.remove(p)
            except: pass

    return jsonify({"transcription": text}), 200


@app.route("/translate_text", methods=["POST"])
def translate_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    target_lang = data.get("target_lang")

    if not text or not target_lang:
        return jsonify({"error": "Missing 'text' or 'target_lang' in JSON body"}), 400

    try:
        translated = translator.translate(text, dest=target_lang)
        return jsonify({"translated_text": translated.text}), 200
    except Exception as e:
        return jsonify({"error": f"Translation failed: {str(e)}"}), 500


@app.route("/youtube_to_text", methods=["POST"])
def youtube_to_text():
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "No YouTube URL provided"}), 400

    # create a temp filename for download
    tmp_audio = tempfile.mktemp(suffix=".mp3")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_audio,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Downloaded audio to:", tmp_audio)
    except Exception as e:
        if os.path.exists(tmp_audio):
            try: os.remove(tmp_audio)
            except: pass
        return jsonify({"error": f"yt-dlp download failed: {str(e)}"}), 500

    # Convert to wav and transcribe
    try:
        tmp_wav = convert_to_wav(tmp_audio)
        result = model.transcribe(tmp_wav)
        text = result.get("text", "").strip()
        print("YouTube transcription:", text)
    except Exception as e:
        # cleanup and return error
        for p in (tmp_audio, tmp_wav if 'tmp_wav' in locals() else None):
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    # cleanup
    for p in (tmp_audio, tmp_wav):
        if p and os.path.exists(p):
            try: os.remove(p)
            except: pass

    return jsonify({"transcription": text}), 200


if __name__ == "__main__":
    # Make Flask accessible from other devices on local network for testing
    app.run(host="0.0.0.0", port=5000, debug=True)
