# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import sys
import itertools
import torch
import time
import threading
import whisper
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import sounddevice as sd

# Use GPU device is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models and tokenizer from the saved file
checkpoint = torch.load('models.pt')

model_stt = checkpoint['model_stt']
model_stt.to(device)  # Ensure model is on GPU if not moved already

model_llm = checkpoint['model_llm']

model_tts = checkpoint['model_tts']
model_tts.to(device)  # Ensure model is on GPU if not moved already

tokenizer_tts = checkpoint['tokenizer_tts']

# print('Models and tokenizer loaded successfully\n')

"""### Step-1: Perform Speech-to-text using Whisper"""

### code for real-time recording

# Set parameters for recording audio (input audio)
sample_rate = 16000  # Sample rate in Hz
duration = 6  # using this to manage memory resources as I am hosting this app locally
channels = 1

# function to display countdown to tell user about time remaining for recording
def countdown(duration):
    for remaining in range(duration, 0, -1):
        print(f"\r Ask me anything. I am listening... | {remaining} seconds", end='')
        time.sleep(1)
    # print('\n \n Generating response. Please wait ... \n')

# function to record audio
def record_audio(duration, sample_rate):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    return audio_data

# Start countdown in a separate thread
print('\n')
countdown_thread = threading.Thread(target=countdown, args=(duration,))
countdown_thread.start()

# Start recording
audio_data = record_audio(duration, sample_rate)

# Wait for the countdown to finish
countdown_thread.join()

# Save the recorded audio to a WAV file
file_path = "input_audio.wav"
sf.write(file_path, audio_data, sample_rate)

# Performing trascription using Whisper

result = model_stt.transcribe(file_path, language="en")

transcribed_text = result["text"]

"""### Step-2: Use LLM for response generation"""

# Generating response from LLM

prompt_llm = f"Please provide a response in not more than two short and crisp sentences, without using any abbreviations or brackets: {transcribed_text}"

# Generate response
response_text = model_llm.invoke(prompt_llm)

"""### Step-3: Convert LLM-generated response into speech using Parler-TTS"""

# Function to display the loading spinner
def show_spinner(message):
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    sys.stdout.write('\n\n')  
    while not done:
        sys.stdout.write(f"\r{message} {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.2)
    sys.stdout.write('\n') 

pitch = 200  # pitch adjustment in Hertz
speed = 180  # speed in words per minute

# parler-tts is trained on certain voices from which we can choose
speaker = {'male': ['Jon', 'Gary', 'Mike'],
          'female': ['Laura', 'Jenna', 'Lea']}

# Prompt text to be converted to speech
prompt_tts = response_text

# Describe what kind of voice you need using a prompt
description = f"{speaker['male'][0]} speaks in a conversational tone with a pitch of {pitch} Hertz. \
Speaker speaks in a clear voice with a speed of {speed} words per minutes."

# Start the spinner in a separate thread just before TTS generation
done = False
spinner_thread = threading.Thread(target=show_spinner, args=("Generating response. Please wait...",))
spinner_thread.start()

# Tokenize input text and description text
description_encoding = tokenizer_tts(description, return_tensors="pt", padding=True)
prompt_encoding = tokenizer_tts(response_text, return_tensors="pt", padding=True)

# Extract input_ids and attention_mask and move them to GPU device for further computations
description_input_ids = description_encoding['input_ids'].to(device)
description_attention_mask = description_encoding['attention_mask'].to(device)
prompt_input_ids = prompt_encoding['input_ids'].to(device)
prompt_attention_mask = prompt_encoding['attention_mask'].to(device)

# Generate response
generation = model_tts.generate(
    input_ids=description_input_ids,
    attention_mask=description_attention_mask,
    prompt_input_ids=prompt_input_ids,
    prompt_attention_mask=prompt_attention_mask
)

# Stop the spinner once processing is done
done = True
spinner_thread.join()
print()

# Convert the generated tensor to a NumPy array
audio_arr = generation.cpu().numpy().squeeze()

# Stream the audio using sounddevice
sd.play(audio_arr, samplerate=model_tts.config.sampling_rate)
sd.wait()  # Wait until the sound has finished playing

