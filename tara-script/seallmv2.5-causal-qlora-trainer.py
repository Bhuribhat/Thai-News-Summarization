import torch
from datasets import  load_from_disk
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, TaskType

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print(f"Completed importing frameworks")

# Tara working directory
working_dir = f"/tarafs/data/project/proj0183-ATS/finetune/lanta-finetune"
project_dir = f"{working_dir}/pattern-proj"
outputs_dir = f"{project_dir}/seallm2.5-thaisum"

# Download tokenized dataset
tokenized_datasets = load_from_disk(f'{project_dir}/causallm_preprocessed_thaisum.hf')
print(f"Completed loading dataset")

# Download tokenizer and model
model_checkpoint = f"{working_dir}/SeaLLM-7B-v2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    quantization_config=bnb_config
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
print(f"Completed loading tokenizer and model")

# QLoRA Adapter
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=32, 
    lora_alpha=16, 
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
print(f"Completed loading data collator")

# Finetuning via Trainer
args = TrainingArguments(
    output_dir=outputs_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=2e-5,
    save_steps=5_000,
    fp16=True,
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
)
print(f"Started finetuning..\n")
trainer.train()
trainer.save_model()
print(f"Finished finetuning..")
print(f"The fine-tuned weight is saved at {outputs_dir}")