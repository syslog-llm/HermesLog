from sklearn.metrics import classification_report, confusion_matrix, precision_score
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from collections import Counter
from loguru import logger
import json, random, gc
from tqdm import tqdm
import torch
import argparse
import time
import os

parser = argparse.ArgumentParser(description="edgeLogLM LLM vllm Few-Shot-COT Label-Reason")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--dataset", type=str, default="aliyun")
parser.add_argument("--part", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_p", type=float, default=0.0)
parser.add_argument("--test_data_file", type=str, required=True, help="Path to the test data file")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the log file")
parser.add_argument("--peft_model_dir", type=str, required=True, help="Path to save the peft_model")
parser.add_argument("--output_dir", type=str, default="0313")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for inference. Set to 1 for per-case GPU memory measurement.")


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
part_num = args.part
DATASET = args.dataset
TEST_DATA_FILE = args.test_data_file
LOG_PATH = args.save_path
RESULTS_SAVE_PATH = args.save_path
peft_model_dir = args.peft_model_dir
TEMPERATURE = args.temperature
TOP_P = args.top_p
output_dir = args.output_dir
BATCH_SIZE = args.batch_size

if DATASET == "aliyun":
    label2name = {
        1: 'Processor CPU Caterr', 
        2: 'Memory Throttled | Uncorrectable Error Correcting Code', 
        3: 'Hard Disk Drive Control Error | Computer System Bus Short Circuit | Programmable Gate Array Device Unknown'
    }
elif DATASET == "zte":
    label2name = {
        1: 'Power Supply Fault', 
        2: 'Fan Fault', 
        3: 'Optics Module Fault', 
        4: 'Port Failure', 
        6: 'CRC Error (Cyclic Redundancy Check)', 
        7: 'STP Fault (Spanning Tree Protocol)', 
        8: 'BFD Down (Bidirectional Forwarding Detection)', 
        9: 'LACP Flapping (Link Aggregation Control Protocol)', 
        10: 'OSPF Neighbor Flapping (Open Shortest Path First)'
    }
else:
    raise ValueError(f"Unsupported dataset: {DATASET}. Please choose from 'aliyun' or 'zte'")
label2details = label2name
tokens_list = []

model = LLM(
        model=peft_model_dir, 
        dtype='bfloat16', 
        enable_prefix_caching=True, 
        tensor_parallel_size=torch.cuda.device_count(), 
        max_model_len=1024*3,
        gpu_memory_utilization=0.5
    )
tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)
sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=1024)

# def batch_getLogErrorID(logs_batch, tokenizer, max_len=1024*3):
#     batch_token_ids = []
#     batch_messages = []
#     sys_prompt = f'''
#     You will work as a text classification model to classify the operation error logs from the system to the defined categories.
#     The categories are: \n{", ".join(label2details.values())}
#     '''.strip()

#     for log in logs_batch:
#         user_prompt = f"""
#         The operation error log to analysis now is: {log}\n
#         You are now acting as a human labeler, please classify the error type of the log.
#         Choose from: {", ".join(label2details.values())}.
#         Please directly provide the error type only.
#         """.strip()

#         messages = [
#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": user_prompt}
#         ]
#         batch_messages.append(messages)
#         token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
#         batch_token_ids.append(token_ids[-max_len:])
#     return batch_token_ids, batch_messages

def batch_getLogErrorID(logs_batch, tokenizer, max_len=1024*3):
    batch_token_ids = []
    batch_messages = []

    sys_prompt = f'''
    You will work as a text classification model to classify the operation error logs from the system to the defined categories.
    The categories are: \n{", ".join(label2details.values())}
    '''.strip()

    for log in logs_batch:
        user_prompt = f"""
        The operation error log to analysis now is: {log}\n
        Possible error types are: {", ".join(label2details.values())}.
        Please classify the error type of the log.
        Please provide the error type only.
        """.strip()

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        batch_messages.append(messages)

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )

        batch_token_ids.append(token_ids[-max_len:])

    return batch_token_ids, batch_messages

def batch_getLogAnalysis(round1_messages_batch, labels_batch, logs_batch, tokenizer, max_len=1024*3):
    batch_token_ids = []
    chat_history_list = []
    for messages, label, log in zip(round1_messages_batch, labels_batch, logs_batch):
        history_with_label = messages + [{"role": "assistant", "content": label}]

        user_prompt_2 = f"""The operation error log is:{log}\n and the classified error type is: {label}\n
        Based on the error type you've just provided, please write key analysis points for the error.
        Compare it to all the possible error types.
        No need to repeat the label.
        """.strip()
        history_with_label.append({"role": "user", "content": user_prompt_2})
        token_ids = tokenizer.apply_chat_template(history_with_label, tokenize=True, add_generation_prompt=True)
        batch_token_ids.append(token_ids[-max_len:])
        chat_history_list.append(history_with_label)
    return batch_token_ids, chat_history_list

def load_data(filename):
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'Loaded {len(data)} records in total')
    
    labels = [item['label'] for item in data]
    label_counter = Counter(labels)
    logger.info(f'Label distribution: {label_counter}')

    return data

def truncateLogs(logs, max_len=1024):
    encoded_logs = logs.encode('utf-8')
    if len(encoded_logs) > max_len:
        encoded_logs = encoded_logs[:max_len]
    truncated_logs = encoded_logs.decode('utf-8')
    
    return truncated_logs

def ansMatchCheck(answer):
    answer_words = answer.lower().split()
    all_labels = list(label2name.values())
    max_common_words = 0
    matchedLabel = None
    inferLabel = None

    for label in all_labels:

        label_words = label.lower().split()
        common_words = len(set(answer_words).intersection(label_words))

        if common_words > max_common_words:
            max_common_words = common_words
            matchedLabel = label

    if matchedLabel is None:
        logger.warning(f'No matched label found for response: {answer}')

    for errortype, errordetail in label2name.items():
        if errordetail == matchedLabel:
            inferLabel = errortype

    return matchedLabel, inferLabel

def resultsAnalysis(results):
    correct = 0
    for result in results:
        if result['correct']:
            correct += 1

    accuracy = correct / len(results)
    logger.info(f'\nAccuracy: {accuracy*100:.2f}%')

    # get classification report
    y_true = [result['true_label'] for result in results]
    y_pred = [result['infer_label'] for result in results]

    error_ids = list(label2name.keys())
    error_details = list(label2name.values())
    report = classification_report(y_true, y_pred, target_names=error_details, labels=error_ids, digits=4, zero_division=0)
    
    logger.info(f'\nClassification Report:\n{report}')
    
    # get the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=error_ids)
    micro_accuracy = precision_score(y_true, y_pred, average='micro')
    
    logger.info(f'\nConfusion Matrix:\n{cm}', serialize=True)
    logger.info(f'\nmicro avg: {micro_accuracy}')

def saveResults(results, timestamp, filename):
    filename = filename.replace('.json', f'_{timestamp}.json')

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f'Results saved to {filename}')


def ModelInfer_vllm(data, batch_size=10):
    results = []
    total_records = len(data)
    gpu_count = torch.cuda.device_count()

    for i in tqdm(range(0, total_records, batch_size), ncols=100, desc='Processing Batches'):
        batch = data[i: i + batch_size]
        logs_batch = [truncateLogs(item['content']) for item in batch]

        # -------- Round 1 --------
        round1_inputs, round1_messages = batch_getLogErrorID(logs_batch, tokenizer)
        torch.cuda.reset_peak_memory_stats()
        round1_outputs = model.generate(
            [{"prompt_token_ids": ids} for ids in round1_inputs],
            sampling_params=sampling_params
        )
        mem_peak_round1 = max(
            torch.cuda.max_memory_allocated(device=d) for d in range(gpu_count)
        )

        labels_batch = []
        for item, output in zip(batch, round1_outputs):
            answer = output.outputs[0].text.strip()
            matchedLabel, inferLabel = ansMatchCheck(answer)
            if inferLabel is None:
                inferLabel = random.choice(list(label2name.keys()))
                matchedLabel = label2name[inferLabel]
                logger.warning(f'Randomly assigned label for case {item["caseid"]} part {item["part"]}: {inferLabel}')
            labels_batch.append(matchedLabel)
            item['infer_label'] = inferLabel
            item['matched_label'] = matchedLabel
            item['true_label_detail'] = label2name[item['label']],
            item['true_label'] = item['label']
            item['correct'] = inferLabel == item['label']
        # -------- Round 2 --------
        round2_inputs, chat_history_list = batch_getLogAnalysis(round1_messages, labels_batch, logs_batch, tokenizer)
        torch.cuda.reset_peak_memory_stats()
        round2_outputs = model.generate(
            [{"prompt_token_ids": ids} for ids in round2_inputs],
            sampling_params=sampling_params
        )
        mem_peak_round2 = max(
            torch.cuda.max_memory_allocated(device=d) for d in range(gpu_count)
        )

        # 两轮取峰值，作为该 batch 内每个 case 的显存峰值
        mem_peak_bytes = max(mem_peak_round1, mem_peak_round2)

        for item, analysis_output, chat_history in zip(batch, round2_outputs, chat_history_list):
            analysis = analysis_output.outputs[0].text.strip()
            chat_history.append({"role": "assistant", "content": analysis})
            item['analysis'] = analysis
            item['gpu_mem_peak_bytes'] = mem_peak_bytes
            item['gpu_mem_peak_gb'] = round(mem_peak_bytes / (1024**3), 4)
            item['chat_history'] = chat_history
            results.append(item)
            if item['matched_label'] != label2name[item['label']]:
                logger.warning(f'Processed case {item["caseid"]} part {item["part"]}: True Label: {item["true_label"]}, Infer Label ID: {item["infer_label"]}, Matched Label: {item["matched_label"]}')
            else:
                logger.success(f'Processed case {item["caseid"]} part {item["part"]}: True Label: {item["true_label"]}, Infer Label ID: {item["infer_label"]}, Matched Label: {item["matched_label"]}')


        gc.collect()
    return results

if __name__=='__main__':
    timestamp = time.strftime('%Y%m%d%H%M%S')
    logger.add(f"{LOG_PATH}/{output_dir}_label_reason_part_{part_num}_{TEMPERATURE}temp_{TOP_P}topp_{timestamp}.log")

    data = load_data(TEST_DATA_FILE)

    start_time = time.time()
    results = ModelInfer_vllm(data, batch_size=BATCH_SIZE)
    end_time = time.time()
    logger.info(f'Processing time: {end_time-start_time:.2f} seconds, average: {(end_time-start_time)/len(data):.2f} seconds per record')

    # GPU memory peak statistics
    mem_peaks_gb = [r['gpu_mem_peak_gb'] for r in results]
    logger.info(f'\n========== GPU Memory Peak Statistics (per case) ==========')
    logger.info(f'  Min:    {min(mem_peaks_gb):.4f} GB')
    logger.info(f'  Max:    {max(mem_peaks_gb):.4f} GB')
    logger.info(f'  Mean:   {sum(mem_peaks_gb)/len(mem_peaks_gb):.4f} GB')
    logger.info(f'  Median: {sorted(mem_peaks_gb)[len(mem_peaks_gb)//2]:.4f} GB')
    logger.info(f'=============================================================\n')

    saveResults(results, timestamp, f"{RESULTS_SAVE_PATH}/{output_dir}_label_reason_part_{part_num}_{TEMPERATURE}temp_{TOP_P}topp.json")
    resultsAnalysis(results)
