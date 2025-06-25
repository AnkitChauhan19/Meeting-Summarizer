import json
import torch
from datasets import Dataset
from transformers import LEDForConditionalGeneration, LEDTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import logging

"""
Loading the LED (Longformer-Encoder-Decoder) model of 
the Hugging Face transformer library, fine-tuning it
on the ami_aligned_sections.json dataset for summarizing 
meeting transcripts into different sections and finally 
storing the fine-tuned model
"""

logging.basicConfig(level=logging.INFO)

device = "cpu"
torch.set_num_threads(4)

with open('ami_aligned_sections.json') as f:
    data = json.load(f)

model_name = "allenai/led-large-16384"

tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)

texts = []
summaries = []

for item in data:
    src = item['transcript']
    
    abstract = " ".join(item['abstract']) if isinstance(item['abstract'], list) else item['abstract']
    actions = " ".join(item['actions']) if isinstance(item['actions'], list) else item['actions']
    decisions = " ".join(item['decisions']) if isinstance(item['decisions'], list) else item['decisions']
    problems = " ".join(item['problems']) if isinstance(item['problems'], list) else item['problems']
    
    tgt = f"[ABSTRACT] {abstract} [ACTIONS] {actions} [DECISIONS] {decisions} [PROBLEMS] {problems}"
    
    texts.append(src)
    summaries.append(tgt)

print(f"Loaded {len(texts)} examples")

hf_dataset = Dataset.from_dict({"input": texts, "target": summaries}).train_test_split(test_size=0.1)

def preprocess(examples):
    max_input_length = 5120
    max_target_length = 256
    
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
    
    global_attention_mask = []
    for input_ids in model_inputs["input_ids"]:
        attention_mask = [0] * len(input_ids)
        attention_mask[0] = 1
        for i in range(512, len(input_ids), 512):
            if i < len(input_ids):
                attention_mask[i] = 1
        global_attention_mask.append(attention_mask)
    
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["global_attention_mask"] = global_attention_mask
    
    return model_inputs

print("Preprocessing dataset...")
encoded = hf_dataset.map(
    preprocess, 
    batched=True, 
    remove_columns=hf_dataset["train"].column_names,
    desc="Preprocessing"
)

print("Setting format...")
encoded.set_format(type="torch")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=None
)

training_args = TrainingArguments(
    output_dir="./led_summarization_model",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=20,
    max_steps=40,
    use_cpu=True,
    eval_strategy="steps",
    eval_steps=20,
    save_total_limit=2,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to=None,
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

print(f"Train dataset size: {len(encoded['train'])}")
print(f"Test dataset size: {len(encoded['test'])}")

try:
    print("Starting training on CPU...")
    trainer.train()
    print("Training finished successfully!")
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()

try:
    trainer.save_model()
    print("Model saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")