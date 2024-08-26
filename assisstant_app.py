# suppress warnings
import warnings

warnings.filterwarnings("ignore")


import torch
import time
import threading
import whisper
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import sounddevice as sd

# Check if GPU device is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models and tokenizer from the saved file
checkpoint = torch.load('models.pt')

model_stt = checkpoint['model_stt']
model_stt.to(device)  # Ensure model is on GPU if needed

model_llm = checkpoint['model_llm']

model_tts = checkpoint['model_tts']
model_tts.to(device)  # Ensure model is on GPU if needed

tokenizer_tts = checkpoint['tokenizer_tts']

# print('Models and tokenizer loaded successfully\n')

"""### Step-1: Perform Speech-to-text using Whisper"""

### code for real-time recording

# Set parameters for recording audio (input audio)
sample_rate = 16000  # Sample rate in Hz
duration = 6  # Duration in seconds
channels = 1

# Function to perform the countdown and display messages
def countdown(duration):
    for remaining in range(duration, 0, -1):
        print(f"\r Ask me anything. I am listening... | {remaining} seconds", end='')
        time.sleep(1)
    # print("\nStopped recording")
    print('\n \n Generating response. Please wait ...')



# Function to record audio
def record_audio(duration, sample_rate):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    return audio_data

# Start countdown in a separate thread
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

start_time = time.time()

# please make sure to change the path of audio file as per your system
result = model_stt.transcribe(file_path, language="en")

transcribed_text = result["text"]

end_time = time.time()

# print('\n', 'You:', transcribed_text)

# print('\n\nTime taken for transcription:', end_time - start_time)

"""### Step-2: Use LLM for response generation"""

# Generating response from LLM
start_time = time.time()

prompt_llm = f"Please provide a response in not more than two short and crisp sentences, without using any abbreviations or brackets (such as '()'. \
    You should not use any abbreviations.). {transcribed_text}"

# Generate response
response_text = model_llm.invoke(prompt_llm)
# print('\n', ':', response_text)

end_time = time.time()

# print('Time taken to generate response:', end_time-start_time)

"""### Step-3: Convert LLM-generated response into speech using Parler-TTS"""

pitch = 90  # pitch adjustment in Hertz
# gender = 'male'  
speed = 120  # speed in words per minute

speaker = {'male': ['Jon', 'Gary', 'Mike'],
          'female': ['Laura', 'Jenna', 'Lea']}

# Prompt text to be converted to speech
prompt_tts = response_text

# Describe what kind of voice you need using a prompt
description = f" {speaker['female'][1]} speaks in a conversational tone with a pitch of {pitch} Hertz. \
Speaker speaks in a clear human voice with a speed of {speed} words per minutes."

start_time = time.time()

# Tokenize inputs with padding and truncation
description_encoding = tokenizer_tts(description, return_tensors="pt", padding=True, truncation=True)
prompt_encoding = tokenizer_tts(response_text, return_tensors="pt", padding=True, truncation=True)

# Extract input_ids and attention_mask
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


# Convert the generated tensor to a NumPy array
audio_arr = generation.cpu().numpy().squeeze()

end_time = time.time()

print("\n\nTime taken for Text-to-speech:", end_time - start_time)

# Stream the audio using sounddevice
sd.play(audio_arr, samplerate=model_tts.config.sampling_rate)
sd.wait()  # Wait until the sound has finished playing

