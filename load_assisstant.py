# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# importing necessary packages

import time
import torch
import whisper
from langchain_ollama import OllamaLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from parler_tts import ParlerTTSForConditionalGeneration

# check if GPU device is available

if torch.cuda.is_available():
    print("GPU is available and being used.")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is not available, using CPU.")


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""### Load the Speech-to-Text model: Whisper"""

# load whisper model into a variable
model_stt = whisper.load_model("tiny.en")

# Move the model to the GPU device if available
model_stt.to(device)

print('Whisper model loaded successfully')

# load llama2 model using Ollama
model_llm = OllamaLLM(model='gemma2:2b')

print('LLM model (Gemma 2) from Ollama is ready to use')

"""### Load Text-to-speech model: Parler-tts"""

# Load the Parler-TTS model and Tokenizer

model_tts_id = "parler-tts/parler-tts-mini-v1"

# Load the tokenizer
tokenizer_tts = AutoTokenizer.from_pretrained(model_tts_id)

# Load the model
model_tts = ParlerTTSForConditionalGeneration.from_pretrained(model_tts_id)

# Move the model to the GPU device if available
model_tts.to(device)

print('TTS model loaded successfully')

# Save entire models and tokenizer to a file
torch.save({
    'model_stt': model_stt,
    'model_llm': model_llm,
    'model_tts': model_tts,
    'tokenizer_tts': tokenizer_tts
}, 'models.pt')

print('Models and tokenizer saved successfully')