from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPT2Tokenizer, GPT2LMHeadModel
import torch
# New Addition: Checking for Apple Silicon (M-Series) CPU if available: (EEP 2025 - Week 2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_path = '/Users/ariankharazmi/PycharmProjects/Curiosity16-run1/PhaseI-checkpoint/checkpoint-12375'
#tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Setup pipeline with a reasonable max length
text_generator = pipeline(
    'text-generation', model=model, tokenizer=tokenizer
)

# Set up pipeline for text generation (relating to user prompt)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, truncation=True)

# Interactive Prompt for user, generate text based on user's entered prompt
while True:
    text = input("Enter a prompt: ")
    if text.lower() == 'exit!':
        break
    result = text_generator(text, max_length=312, num_return_sequences=1)
    print(result[0]['generated_text'])

