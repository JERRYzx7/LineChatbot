import os
import gc
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# -------------------------------
# 1️⃣ 讀取模型與 tokenizer
# -------------------------------
def cleanup_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def load_model_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 不要用 device_map="auto"，改成直接指定

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 建立 8bit 配置
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,          # outlier tensor 超過門檻用 fp16/bf16
    #     llm_int8_has_fp16_weight=False
    # )

    # 建立 4bit 配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # 量化類型（"nf4" 通常效果較好）
        bnb_4bit_use_double_quant=True,   # 是否使用 double quant，能再減少顯存
        bnb_4bit_compute_dtype=torch.bfloat16  # 計算精度，用 bf16 在 5090 上最好
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,   # 或 torch.float16，看 GPU 支援  16bit要跑100小時的樣子
        # quantization_config=bnb_config,  
        device_map="auto",              # 不要分配 meta
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu") #8bit 不需要
    return model, tokenizer

# -------------------------------
# 2️⃣ 套用 LoRA
# -------------------------------
def apply_lora(model, r=16, alpha=32, dropout=0.05):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],  # ✅ 確保 match Gemma
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # ✅ 直接印出 LoRA 有沒有掛上
    return model

# -------------------------------
# 3️⃣ 讀取 CSV 資料
# -------------------------------
def load_csv_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.fillna("")  # 空值填空字串
    return Dataset.from_pandas(df)

# -------------------------------
# 4️⃣ 轉 chat template 並 tokenize
# -------------------------------
def prepare_dataset(dataset, tokenizer, max_seq_length=1024):
    def generate_prompt(example):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        messages = []

        if instruction.strip():
            messages.append({"role": "system", "content": instruction})
        
        if input_text.strip():
            messages.append({"role": "user", "content": input_text})

        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True # 添加 Assistant 的開始標記
        )
        full_text = f"{prompt}{output}{tokenizer.eos_token}"
        return full_text
        

    def tokenize_function(example):
        text = generate_prompt(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length
        )
        labels = tokenized["input_ids"].copy()
        # 把 padding token 設成 -100，避免 loss 計算錯誤
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]
        tokenized["labels"] = labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=False, remove_columns=dataset.column_names) # ✅ 移除舊的欄位以避免 map 錯誤
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_dataset


# -------------------------------
# 5️⃣ 訓練
# -------------------------------
def train_model(model, tokenizer, tokenized_dataset, output_dir: str):

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=3e-4,
        bf16=True,                     # ✅ 5090/Blackwell 建議用 bf16
        fp16=False, 
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        push_to_hub=False
    )
    
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Finetune 完成！模型已存到 {output_dir}")

# -------------------------------
# 6️⃣ 整個 pipeline
# -------------------------------
def taide_finetune_pipeline(
    model_dir: str,
    csv_path: str,
    output_dir: str,
    max_seq_length=1024,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05
):
    model, tokenizer = load_model_tokenizer(model_dir)
    model = apply_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    model.enable_input_require_grads()      # ✅ 讓 input embeddings 支援梯度（配合 checkpointing 很關鍵）
    model.config.use_cache = False  
    # print("=== Trainable Parameters ===")
    # trainable_params = 0
    # all_params = 0
    # for name, param in model.named_parameters():
    #     all_params += param.numel()
    #     if param.requires_grad:
    #         print(name)
    #         trainable_params += param.numel()

    # print(f"Trainable params: {trainable_params} / {all_params} "f"({100 * trainable_params/all_params:.4f}%)")
    dataset = load_csv_data(csv_path)

    tokenized_dataset = prepare_dataset(dataset, tokenizer, max_seq_length)

    batch = tokenized_dataset[0]
    batch = {k: v.unsqueeze(0).to(model.device) for k, v in batch.items()}

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(**batch)

    print("DEBUG loss:", outputs.loss, "requires_grad:", outputs.loss.requires_grad)

    train_model(model, tokenizer, tokenized_dataset, output_dir)
    cleanup_model(model)
