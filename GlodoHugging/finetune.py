import sys
import importlib.util
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk

def import_config(filename):
    if not os.path.exists(filename):
        print(f"Config file '{filename}' not found.")
        return

    module_name = os.path.splitext(os.path.basename(filename))[0]
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config = import_config(sys.argv[1] if len(sys.argv) > 1 else "configs/default.py")

dataset = load_dataset("text", data_dir="datasets", sample_by="document")

# Path to save the tokenized dataset
tokenized_dataset_path = "out/tokens-glodomorek-medium"
tokenizer = AutoTokenizer.from_pretrained(config.model_data)
tokenizer.pad_token = tokenizer.eos_token

# Check if the tokenized dataset already exists
if os.path.exists(tokenized_dataset_path):
    print("Loading tokenized dataset from disk...")
    tokenized_datasets = load_from_disk(tokenized_dataset_path)
else:
    print("Tokenized dataset not found. Tokenizing now...")

    def tokenize_function(examples):
        # The max length is not set to max context window size here, as the DataCollator during training
        # will do the padding.
        # Padding here doesn't matter much as there's a single big file as an input, so it will be mostly
        # truncated instead of padded.
        outputs = tokenizer(examples["text"], padding=False, truncation=True, max_length=512, stride=20,
                            return_overflowing_tokens=True, add_special_tokens=False)
        outputs["labels"] = outputs["input_ids"]
        return outputs

    # Remove 'text', as the dataset will be 'exploded' due to truncating to the block_size. In that case
    # the 'text' column cannot be handled and there will be error like this:
    # 'Column 1 named input_ids expected length 1 but got length 62'
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # By flattening or otherwise reformatting the output so that each chunk becomes an independent example,
    # you can avoid the mismatch error and ensure all parts of your long document are preserved for training.
    # tokenized_datasets_flattened = tokenized_datasets.flatten()

    # Save the tokenized dataset for future use
    tokenized_datasets.save_to_disk(tokenized_dataset_path)
    print("Tokenized dataset saved to disk.")

train_dataset = tokenized_datasets["train"]

# The dataset is so small that taking the validation data from it makes not much sense.
# However, can play-around with at least some kind of validation in the future.
#eval_dataset = tokenized_datasets["test"]

training_args = TrainingArguments(
    report_to="none",
    run_name="Glodotuning-" + str(time.time()),
    output_dir="./out/glodomorek-ft-medium-checkpoints",      # Katalog zapisu wyników
    #eval_strategy="epoch", # Ocena modelu po każdej epoce
    do_eval=False,
    eval_strategy='no',
    #save_strategy="best",       # Zapisywanie checkpointów co epokę
    save_strategy="steps",
    save_steps=30,
    learning_rate=3e-5,          # Współczynnik uczenia
    per_device_train_batch_size=1,  # Batch size na GPU/CPU
    per_device_eval_batch_size=1,   # Batch size dla walidacji
    #num_train_epochs=1,         # Liczba epok treningowych
    max_steps = 30,
    weight_decay=1e-1,          # Regularizacja L2
    logging_dir="./logs",       # Katalog logów
    logging_steps=10,           # Częstotliwość logowania
    push_to_hub=False,           # Jeśli chcesz zapisać model na HF Hub, zmień na True
    gradient_accumulation_steps=32,
    bf16=True,
    adam_beta2=0.95,
    warmup_steps=2000,
    lr_scheduler_type='constant',
    seed=42,
    data_seed=42,
)

model = AutoModelForCausalLM.from_pretrained(config.model_data).to(config.device)
model.compile()


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    processing_class=tokenizer,
)
print("Start SFT")
#trainer.train(resume_from_checkpoint="out/glodomorek-ft-small-checkpoints/checkpoint-10")
trainer.train()

print("Saving models")
trainer.save_model("out/fine_tuned_model-medium")
