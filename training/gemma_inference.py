import os
import random
import torch
import time
import re
from datetime import datetime
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# from train_model.trim import analyze_and_modify_response
from typing import List
# from utils import chroma


model_cache = {}
model_usage_counter = {}
max_cache_size = 3
usage_threshold = 5

total_memory = torch.cuda.get_device_properties(0).total_memory
threshold = int(total_memory * 0.75)

def load_memory(memory_path: str, max_history: int = 6):
    """讀取歷史對話 CSV，並只取最近 N 條"""
    if not os.path.exists(memory_path):
        return []

    df = pd.read_csv(memory_path)
    if len(df) > max_history:
        df = df.tail(max_history)

    history = []
    for _, row in df.iterrows():
        content = row.get("text")
        if not content or (isinstance(content, float) and math.isnan(content)):
            continue
        history.append({
            "role": row["role"].capitalize(),
            "content": content
        })
    return history

def save_memory(memory_path: str, role: str, text: str):
    """將新對話 append 到 CSV"""
    import math

    if not text or (isinstance(text, float) and math.isnan(text)):
        print(f"[INFO] Skip saving empty or NaN text for role {role}")
        return

    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "text": text
    }])
    if not os.path.exists(memory_path):
        entry.to_csv(memory_path, index=False)
    else:
        entry.to_csv(memory_path, mode='a', index=False, header=False)    

def manage_model_cache():
    global model_cache, model_usage_counter

    current_memory = torch.cuda.memory_allocated()
    print(
        f"[INFO] Current memory allocated: {current_memory / 1e9:.2f} GB (Threshold: {threshold / 1e9:.2f} GB)"
    )

    if current_memory >= threshold or len(model_cache) > max_cache_size:
        print("[INFO] Memory or cache size exceeded. Cleaning up cache...")
        least_used_models = sorted(model_usage_counter.items(), key=lambda x: x[1])
        for user_id, _ in least_used_models:
            if user_id in model_cache:
                print(f"[INFO] Removing model for user_id: {user_id}")
                del model_cache[user_id]
                del model_usage_counter[user_id]
                torch.cuda.empty_cache()

                current_memory = torch.cuda.memory_allocated()
                if current_memory < threshold and len(model_cache) <= max_cache_size:
                    break


model_cache = {}

def load_model(base_model_dir: str,model_dir: str):
    global model_cache

    # 如果已經載過，就直接用 cache
    if model_dir in model_cache:
        print(f"[INFO] Using cached model from: {model_dir}")
        return model_cache[model_dir]

    # 否則就載入
    print(f"[INFO] Loading model from: base model")
    model = AutoModelForCausalLM.from_pretrained(base_model_dir)
    adapter_config_path = os.path.join(model_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        if not hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(model, model_dir)
    else:
        print(f"[INFO] No PEFT adapter found in {model_dir}. Loading base model.")
    
    if hasattr(model, "peft_config") and model.peft_config is not None:
        print("LoRA adapter 已經套上")
    else:
        print("LoRA adapter 尚未套上")
    # 移到 GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # 存進 cache
    model_cache[model_dir] = (model, tokenizer)

    return model, tokenizer


def limit_stickers(text: str) -> str:
    max_stickers = 2
    sticker_tokens = text.split("[貼圖]")
    if len(sticker_tokens) > max_stickers:
        text = "[貼圖]".join(sticker_tokens[:max_stickers]) + sticker_tokens[max_stickers]

    return text

def clean_output(text: str) -> str:
    match = re.search(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()  # 不做過度處理


def build_prompt(chat_history: List[dict], user_input: str, system_prompt: str = None) -> List[dict]:
    prompt_list = []
    if system_prompt:
        prompt_list.append({"role": "system", "content": system_prompt})

    for msg in chat_history:
        content = msg.get("content")
        if content and not (isinstance(content, float) and math.isnan(content)):
            prompt_list.append({"role": msg["role"], "content": content})

    if user_input and not (isinstance(user_input, float) and math.isnan(user_input)):
        prompt_list.append({"role": "user", "content": user_input})

    return prompt_list

def gemma_inference(
    base_model_dir: str,
    model_dir: str,
    input_text: str,
    memory_path: str = "./chat/chat.csv",
    max_retries: int = 3,
    max_history: int = 100,
) -> List[str] | None:
    try:
        greetings = [
            "晚上好",
            "明天見",
            "安安",
            "午安",
            "晚安",
            "早安",
            "早阿",
            "早",
            "你好",
            "哈囉",
            "嗨",
            "掰掰",
            "拜拜",
            "掰",
            "拜",
            "掰囉",
            "拜囉",
            "掰掰囉",
            "拜拜囉",
            "再見",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]

        if input_text.lower().strip() in [greet.lower() for greet in greetings]:
            delay_seconds = random.uniform(3, 7)
            time.sleep(delay_seconds)
            return [input_text]

        model, tokenizer = load_model(base_model_dir,model_dir)

        # system_prompt = "想像你是一位充滿愛心與同理心的社工，專門陪伴長者、減少孤獨感。你以溫暖、傾聽和耐心著稱，讓長者感到親近與被理解。你們正在透過一個 AI 陪伴應用程式進行聊天。在這段對話中，你的角色是陪伴並引導閒聊。請確保訊息簡潔，每則回覆約 1 到 30 個字以內，每次回覆都包含一個自然的問題，引導長者回答或分享，讓對話能自然延續。記住，這是發送給長者的文字訊息，因此每則回覆最多不超過 30 個字。除非長者表示想結束，否則避免過早結束對話。不要過度熱情，例如不要在每則訊息末尾都加驚嘆號"

        history = load_memory(memory_path, max_history)
        prompt_list = build_prompt(history, input_text)
        prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in prompt_list])

        # if isinstance(user_history, list) and user_history:
        #     training_file = random.choice(user_history)
        # else:
        #     training_file = user_history

        # if training_file and os.path.exists(training_file.filename):
        #     with open(training_file.filename, "r") as f:
        #         df = pd.read_csv(f)
        #     num_samples = 5
        #     if len(df) > num_samples:
        #         df_sample = df.tail(n=num_samples)
        #     else:
        #         df_sample = df

        #     for _, row in df_sample.iterrows():
        #         chat.append(f"User: {row['input']}")
        #         chat.append(f"Assistant: {row['output']}")

        # rag_content = chroma.retrive_n_results(user_id=user_id, query_texts=input_text)
        # if rag_content:
        #     chat.append("System: 以下是檢索到跟使用者相關內容，如果對話提及相關話題可以參考：")
        #     chat.append(rag_content)
        

        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)
        print(f"This is prompt: \n{prompt}\n prompt end")
        for attempt in range(max_retries):
            try:
                # generate_two_responses = random.random() < 0.5
                # num_return_sequences = 2 if generate_two_responses else 1

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        do_sample=True,
                        # max_length=150,
                        max_new_tokens=150,
                        top_k=30,
                        top_p=0.85,
                        temperature=0.7,
                        num_return_sequences=1,
                        repetition_penalty=1.2,   # 懲罰重複
                        no_repeat_ngram_size=3,   # 不允許 3-gram 重複
                    )

                raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                print(f"[DEBUG] Raw model output:\n{raw_output}\n")
                generated_text = clean_output(raw_output)
                print(f"[DEBUG] After clean:\n{generated_text}\n")
                # generated_text = limit_stickers(generated_text)

                # generated_text = get_first_assistant(generated_text)

                # if "Assistant:" in generated_text:
                #     generated_text = generated_text.split("Assistant:")[-1].strip()

                tags_to_remove = [
                    "ANTER",
                    "問：",
                    "問題：",
                    "入題",
                    "回答：",
                    "答：",
                    "問題：",
                    "入題",
                    "回答：",
                    "[入戲]",
                    "ANCES",
                    "ANS",
                    "ANSE",
                    "ANSION",
                    "ANTS",
                    "[檔案]",
                    "<<SYS>>",
                    "INSTP",
                    "[/INST]",
                    "INST",
                    "[You]",
                    "[User]",
                    "User",
                    "[Assistant]",
                    "Assistant",
                    "\\n:",
                    "\\",
                    ":",
                    "[你]",
                    "[我]",
                    "[輸入]",
                    "ERM [/D]",
                    "ANCE ",
                    "S]",
                    "\\",
                    "/",
                    "(null)",
                    "null",
                    "[貼文]",
                    "[照片]",
                    "<end>",
                ]

                for tag in tags_to_remove:
                    generated_text = generated_text.replace(tag, "").strip()

                if input_text in generated_text:
                    generated_text = generated_text.replace(input_text, "").strip()
            
                generated_text = " ".join(
                    line for line in generated_text.splitlines() if line.strip()
                )

                save_memory(memory_path, "User", input_text)
                save_memory(memory_path, "Assistant", generated_text)
                
                if generated_text:
                    print(generated_text)
                    return generated_text  #responses不穩定
                # generated_text = analyze_and_modify_response(
                #     input_text, generated_text, modelname, chat, session_history
                # )
                # responses.append(generated_text)
                
                # if any(r.strip() for r in responses):
                #     return responses
                print(f"[WARN] Attempt {attempt + 1}: Empty response. Retrying...")
                time.sleep(1)

            except torch.cuda.OutOfMemoryError:
                print(
                    f"[ERROR] CUDA Out of Memory during attempt {attempt + 1}. Cleaning up..."
                )
                torch.cuda.empty_cache()
                time.sleep(2)
            except Exception as e:
                if "524" in str(e):
                    print(
                        f"[WARN] 524 Timeout encountered on attempt {attempt + 1}. Retrying..."
                    )
                else:
                    print(f"[ERROR] Inference attempt {attempt + 1} failed: {e}")

                print("[ERROR] All inference attempts failed or returned empty responses.")
                return None

    except Exception as e:
        print(f"Error in inference: {e}")
        return None
