import os
import re
import sys
import json
import time
import torch
import pandas as pd
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("Import completed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {DEVICE} device")

# Tara working directory
main_dir = "/tarafs/data/project/proj0183-ATS/finetune/lanta-finetune"

# Read dataset
df = pd.read_csv(f'{main_dir}/pattern-proj/sample_dataset.csv')
print("Load dataset completed")

# Load model
model_name_or_path = f"{main_dir}/SeaLLM-7B-v2"

def load_LLM_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache = False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


model, tokenizer = load_LLM_and_tokenizer()
model.config.use_cache = False
print("Load model and tokenizer completed")

max_new_tokens = 512    # @param {type: "integer"}
temperature = 0.2       # @param {type: "number"}

# Inference prompt
SYSTEM_PROMPT = "ให้สรุปข้อความที่ให้"
ONE_SHOT_PROMPT = """ตัวอย่างผลลัพธ์ (อย่าเอามาตอบ):
รอกันมา 2 สัปดาห์ บอลสโมสรลีกต่างๆของยุโรปก็กลับมาดวลแข้งกันอีกครั้งในวีกนี้ ช่วงฟีฟ่าเดย์ รอบล่าสุดทิ้งทวนคัดบอลโลก มีประเด็นให้ฮือฮากันหลังจากอิตาลี หนึ่งในทีมอมตะของวงการลูกหนังโลก"""


def generate_inference_prompt(
    question: str,
    one_shot: str = ONE_SHOT_PROMPT,
    system_prompt: str = SYSTEM_PROMPT
) -> str:
    return f"""<s><|im_start|>system
{system_prompt.strip()}
{one_shot.strip()}</s><|im_start|>user
{question.strip()}</s><|im_start|>assistant
"""


def generate_summarization(text: str):
    try:
        batch = tokenizer(
            text, return_tensors="pt", padding="max_length", 
            truncation=True, max_length=4096, add_special_tokens=False
        )
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(
                input_ids=batch["input_ids"].to(DEVICE),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id
            )
        encoded_answer = output_tokens[0][len(batch["input_ids"][0]):]
        response = tokenizer.decode(encoded_answer, skip_special_tokens=True)
        return response
    except:
        print(f"Fail attempt!")
        return None


# Main inferencing 
label_dataset = []
for i, (index, data_point) in enumerate(df.iterrows()):
    body = data_point["body"]
    label = data_point["summary"]
    text = generate_inference_prompt(body)
    output = generate_summarization(text)
    data = {
        "message" : body,
        "output"  : output
    }
    label_dataset.append(data)
    if i % 10 == 0:
        print(f"Processed {i} records")

# Save output to json file
with open(f"{main_dir}/pattern-proj/summarization_by_SeaLLMs-7B-v2.json", "w", encoding='utf8') as file:
    json.dump(label_dataset, file, ensure_ascii=False)