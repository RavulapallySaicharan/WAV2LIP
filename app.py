def runcmd(cmd, verbose = False, *args, **kwargs):
  process = subprocess.Popen(
      cmd,
      stdout = subprocess.PIPE,
      stderr = subprocess.PIPE,
      text = True,
      shell = True
  )
  std_out, std_err = process.communicate()
  if verbose:
      print(std_out.strip(), std_err)
  pass

def generate_audio(text, audio_file):
  audio_file = time.strftime("%Y%m%d-%H%M%S")+'.wav'
  #tts.tts_to_file(text=text, speaker=tts.speakers[4], language=tts.languages[0], file_path=audio_file)
  tts.tts_to_file(text=text, speaker_wav=audio_file, language='en', file_path=audio_file)
  
  return('/content/'+audio_file)

def generate_video(audio_path, video_path):
  runcmd(f"cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {video_path} --audio {audio_path} --resize_factor 2", verbose=True)
  return('/content/Wav2Lip/results/result_voice.mp4')

def clean_text(text):
  # Remove website links
  text = re.sub(r'http\S+', '', text)
  # Remove unnecessary punctuations
  text = re.sub(r'[^\w\s,\.]', '\n', text)
  # Remove extra whitespaces
  text = re.sub(r'\s+', ' ', text)
  # Remove periods not preceded by any character
  text = re.sub(r'(?<!\w)\.', '', text)
  # Replace multiple white spaces with a single white space
  text = re.sub(r'\s+', ' ', text)
  return text.strip()



from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from TTS.api import TTS
import time
import subprocess
import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import re


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

model_name = TTS.list_models()[0]

#tts = TTS(model_name)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)


app = Flask(__name__)
CORS(app)
run_with_ngrok(app)


@app.route('/')
def home():
    return '<h1>Hello, World!</h1>'

@app.route('/generate_audio', methods=['POST'])
def generate_audio_route():
    text = request.form['text']
    audio_path = generate_audio(text)
    return send_file(
         audio_path, 
         mimetype="audio/wav",
         as_attachment=True)


@app.route('/generate_video', methods=['POST'])
def generate_video_route():
    text = request.form['text']
    clone_audio = request.file['clone_audio']
    clone_audio.save('clone_input_file.wav')
    input_video = request.file['input_video']
    input_video.save('input_video.mp4')
    
    audio_path = generate_audio(text, 'clone_input_file.wav')
    #video_path = '/content/amit_vaid_input_video.mp4'
    output_path = generate_video(audio_path, 'input_video.mp4')
    return send_file(
         output_path, 
         mimetype="video/mp4", 
         as_attachment=True)


app.run()
