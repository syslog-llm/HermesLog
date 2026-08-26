from transformers.data.data_collator import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
import pandas as pd
import torch
import json
import numpy as np

class JsonToDataFrame:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path

    def load_json_to_dataframe(self):
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            
            df = pd.DataFrame(data)
            print("JSON文件已成功加载为DataFrame。")

            # df = self.process_chat_history(df)
            return df
        except Exception as e:
            print(f"加载JSON文件时出错: {e}")
            return None

    def process_chat_history(self, df):
        if "chat_history" not in df.columns:
            print("警告：DataFrame中没有chat_history列，跳过处理。")
            return df
        
        def extract_content(chat_history):
            content_dict = {}
            content_dict['system'] = chat_history[0]['content']
            content_dict['user_round1'] = chat_history[1]['content']
            content_dict['analysis'] = chat_history[2]['content']
            content_dict['user_round2'] = chat_history[3]['content']
            return content_dict

        content_df = df["chat_history"].apply(extract_content).apply(pd.Series)

        df = pd.concat([df, content_df], axis=1)
        df = df.drop(columns=['chat_history'])
        return df


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, alpha, beta, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.label_alpha = alpha
        self.reason_beta = beta

    def __call__(self, examples):
        batch = super().__call__([{k: v for k, v in ex.items() if (k != "formatted_text" and k != "offset_mapping")} for ex in examples])
        # print("="*20+"CustomDataCollator"+"="*20)
        # print(examples[0])
        loss_masks = []
        for i, example in enumerate(examples):
            text = example["formatted_text"]
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt",
                return_offsets_mapping=True
            )
            
            loss_mask = torch.ones(len(tokenized["input_ids"][0]), dtype=torch.float)

            think_start = text.find("<reason>")
            think_end = text.find("</reason>")
            label_start = text.find("<label>")
            label_end = text.find("</label>")

            # print(f"think_start: {think_start}, think_end: {think_end}, label_start: {label_start}, label_end: {label_end}")

            if think_start != -1 and think_end != -1:
                start_idx, end_idx = None, None
                for j, (start, end) in enumerate(tokenized.offset_mapping[0]):
                    if start <= think_start < end:
                        start_idx = j
                    if start <= think_end < end:
                        end_idx = j
                        break
                if start_idx is not None and end_idx is not None:
                    loss_mask[start_idx:end_idx+1] = self.reason_beta

            if label_start != -1 and label_end != -1:
                start_idx, end_idx = None, None
                for j, (start, end) in enumerate(tokenized.offset_mapping[0]):
                    if start <= label_start < end:
                        start_idx = j
                    if start <= label_end < end:
                        end_idx = j
                        break
                if start_idx is not None and end_idx is not None:
                    loss_mask[start_idx:end_idx+1] = self.label_alpha
            
            loss_masks.append(loss_mask)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            print("="*30)
            if torch.all(loss_mask == 1.):
                print("True")
            else:
                print("False")

        batch["loss_mask"] = torch.stack(loss_masks)
        print("="*30)
        for key, value in batch.items():
            print(f"Key: {key}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'Not a tensor'}")

        return batch
    
    
class DebugDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        print("Before super():", features)
        batch = super().__call__(features)
        print("After super():", batch.keys())
        return batch
    
    

if __name__ == "__main__":
    json_loader = JsonToDataFrame("./SLM_data/Aliyun/train4/labeled/step5/LLM_infer/LLM_infer_part_1.json")
    df = json_loader.load_json_to_dataframe()
    df.to_csv('', index=False, encoding='utf-8')
    if df is not None:
        print("DataFrame content：")
        print(df.columns)
