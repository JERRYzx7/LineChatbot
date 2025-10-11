# main.py
import argparse
from finetune import train  # 你提供的 train 函數
from inference import inference
from gemma_finetune import finetune_pipeline
from gemma_inference import gemma_inference
def main():
    parser = argparse.ArgumentParser(description="LoRA 微調 TAIDE-LX 模型")
    
    parser.add_argument(
        "--model_dir", type=str, default="./models/saved-taide-model-7B", help="Base model 資料夾"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./loras/lora_model-7B", help="儲存微調模型的資料夾"
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/data_modified.csv", help="訓練資料 CSV 路徑"
    )
    args = parser.parse_args()

    # 執行訓練
    train(
        model_dir=args.model_dir,
        save_dir=args.save_dir,
        data_path=args.data_path
    )

def inference_model():
    # inference("./models/saved-taide-model-8B", "./loras/lora_model-8B","因為不用上班的時候我可能就會比較有時間可以陪太太吧？說不定就會比較常常出去，比較會到處去逛一逛，還有就是說可能就是比較有時間可以去跟一些喜歡音響的朋友交流，不一定說有時間去參加那個協進會或是老人會，但是也是有可能會參加啦。")
    # inference("./models/saved-taide-model", "./loras/lora_model","因為不用上班的時候我可能就會比較有時間可以陪太太吧？說不定就會比較常常出去，比較會到處去逛一逛，還有就是說可能就是比較有時間可以去跟一些喜歡音響的朋友交流，不一定說有時間去參加那個協進會或是老人會，但是也是有可能會參加啦。")
    # inference("./models/saved-taide-model-8B", "./loras/lora_model-8B","你要是說在家裡無聊，你去一天就好了，為什麼你去那麼多去那麼久，對吧。,就一整群。")
    # inference("./models/saved-taide-model", "./loras/lora_model","你要是說在家裡無聊，你去一天就好了，為什麼你去那麼多去那麼久，對吧。,就一整群。")
    inference("./models/saved-taide-model-8B", "./loras/lora_model-8B","那今天的訪談也該結束了")
    # inference("./models/saved-taide-model", "./loras/lora_model","那今天的訪談也該結束了")

def finetune_gemma():
    finetune_pipeline("./models/saved-Qwen3-model-8B", "./data/elder.csv", "./loras/lora_model-Qwen3-8B-elder-2")

def inference_gemma():
    gemma_inference("./models/saved-Qwen3-model-8B", "./loras/lora_model-Qwen3-8B-teacher", "他還年輕，自己慢慢學就好。")


if __name__ == "__main__":
    # inference_model()
    # inference_gemma()
    # main()
    finetune_gemma()
