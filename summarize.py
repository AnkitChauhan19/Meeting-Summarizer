import regex as re
import torch
import json
from transformers import LEDForConditionalGeneration, LEDTokenizer

def load_trained_model(model_path="./led_summarization_model"):
    """
    Function to load the trained model and tokenizer

    --- Parameters ---
    model_path: path to the folder where the trained model is stored
    """

    tokenizer = LEDTokenizer.from_pretrained(model_path)
    model = LEDForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def summarize_transcript(transcript, model, tokenizer, max_new_tokens=512):
    """
    Function to summarize a new transcript using the trained model
    
    --- Parameters ---
    transcript: transcripted meeting text
    model: model to be used for summarization
    tokenizer: tokenizer to be used by the model
    max_new_tokens: maximum number of new tokens in the generated summary
    """
    
    inputs = tokenizer(
        transcript,
        max_length=5120,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    global_attention_mask = torch.zeros_like(inputs["attention_mask"])
    global_attention_mask[:, 0] = 1
    
    seq_len = inputs["input_ids"].shape[1]
    for i in range(512, seq_len, 512):
        if i < seq_len:
            global_attention_mask[:, i] = 1
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=global_attention_mask,
            max_new_tokens=max_new_tokens,
            min_length=100,
            num_beams=4,
            early_stopping=False,
            no_repeat_ngram_size=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def extract_sections(text):
    """
    Function to separate the generated summary into different sections

    --- Parameters ---
    text: summary text generated by the model
    """
    
    sections = ["ABSTRACT", "ACTIONS", "DECISIONS", "PROBLEMS"]
    result = {e.upper(): "" for e in sections}

    for section in sections:
        # Replace *SECTION* with [SECTION]
        text = re.sub(rf'\*{section}\*', rf'[{section}]', text, flags=re.IGNORECASE)
        # Replace [SECTION* with [SECTION]
        text = re.sub(rf'\[{section}\*', rf'[{section}]', text, flags=re.IGNORECASE)
        # Replace *SECTION] with [SECTION]
        text = re.sub(rf'\*{section}\]', rf'[{section}]', text, flags=re.IGNORECASE)

    text = re.sub(rf'\*NA\*', rf'', text, flags=re.IGNORECASE)
    text = re.sub(rf'\NA*', rf'', text, flags=re.IGNORECASE)

    tag_pattern = re.compile(r'\[(ABSTRACT|ACTIONS|DECISIONS|PROBLEMS)\]', re.IGNORECASE)
    matches = list(tag_pattern.finditer(text))

    for i, match in enumerate(matches):
        tag = match.group(1).upper()
        tag_lower = tag.upper()
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        result[tag_lower] = content

    return result