from taide_inference import inference
from qwen3_finetune import finetune_pipeline
from qwen3_inference import qwen3_inference

def finetune():
    finetune_pipeline("./models/saved-taide-model-8B", "./data/teacher.csv", "./loras/lora_model-taide-8B-teacher")
    # finetune_pipeline("./models/saved-Qwen3-model-8B", "./data/elder.csv", "./loras/lora_model-Qwen3-8B-elder-2")

def inference():
    # qwen3_inference("./models/saved-taide-model-8B", "./loras/lora_model-taide-8B-teacher", "我會常常跟他們出去走走、運動")
    qwen3_inference("./models/saved-Qwen3-model-8B", "./loras/lora_model-Qwen3-8B-teacher", "我有兩個兒子")


if __name__ == "__main__":
    inference()
    # finetune()
