# HermesLog

Official implementation of **"HermesLog: A Bridging Model Coordinated CoT Reasoning between LLMs and SLMs Collaborative Fault Diagnosis with Edge Autonomy"**.

HermesLog is an unsupervised log-based fault diagnosis framework for 5G-enabled IIoT. It uses a frozen medium-sized language model (MLM) as a cognitive relay compiler to bridge cloud-based LLMs and edge-based SLMs, so that edge models inherit evidence-grounded reasoning and escalate only complex cases to the cloud.

## 🌟 Key Features

* **Reason-Then-Diagnose Probing (FOLFR)**: Shifts from post-hoc rationalization to a zero-shot, four-stage long-CoT workflow (*Observation → Analysis → Inference → Aggregation*). Extracts fault clues and binds them to raw log evidence without manual annotation.
* **Cognitive Relay Compilation (CRC-SCR)**: Uses a frozen medium-sized language model (Mistral-7B) as an offline compiler. It condenses verbose cloud LLM reasoning into a structured three-element representation $\langle \mathcal{F}, \mathcal{S}, L \rangle$ (key features, $\le 3$-step short-CoT chain, and diagnosis label) screened by rubric-guided confidence scores.
* **Stepwise-Aligned Edge Autonomy**: Trains lightweight edge SLMs via a three-level curriculum (Label $\rightarrow$ Feature $\rightarrow$ Short-CoT) with consistency regularization. Resolves over 94% of routine faults locally within seconds (0.54s–2.47s) and escalates only high-uncertainty cases to the cloud LLM.
* **Plug-and-Play Backbone Adaptability**: Supports diverse cloud LLMs (`GPT-4o`, `Claude 3.5 Sonnet`) and edge SLMs (`Gemma3-4b-it`, `Phi4-mini-3.8b`, `Qwen2.5-3b`) under a unified pipeline without changing prompts, thresholds, or training configurations.

---

## 🧩 Key Components

HermesLog operates across two phases through three core modules:

1. **Fault-Oriented Log Filtering and Reasoning (FOLFR)**
   * **Edge Preprocessing**: Applies content-guided DBSCAN clustering (weighted by log severity and IDF rarity) and gap-based session slicing to extract diverse, low-redundancy case windows from raw syslogs.
   * **Cloud Bootstrapping**: Guides the cloud LLM through two rounds of Q&A dialogue to generate long-CoT `Reasoning V1` ($\mathcal{R}_{LLM}$) and diagnosis labels. High-confidence samples ($\max(\gamma_j) > 0.90$) are retained as `Demo V1` ($\mathcal{D}_{demo}$).

2. **CRC-Condensed Short-CoT Reasoning (CRC-SCR)**
   * **Few-Shot Compilation**: Taking `Demo V1` as in-context demonstrations, the frozen MLM compresses four-stage long-CoT traces into $\langle \mathcal{F}, \mathcal{S}, L \rangle$ (`Reasoning V2`) in a single forward pass.
   * **Rubric-Guided Filtering**: Evaluates each compiled output across feature fidelity ($\gamma_{\mathcal{F}}$), chain consistency ($\gamma_{\mathcal{S}}$), and label consistency ($\gamma_L$). Samples passing structural hard constraints and $\gamma_{mid} > \tau_{mid}$ form the edge transfer set `Demo V2` ($\mathcal{D}_{trs}$).

3. **Stepwise-Aligned Autonomous Diagnosis (SLMs)**
   * **Offline Progressive Alignment**: Edge SLMs learn from `Demo V2` sequentially via Cross-Entropy label loss ($\mathcal{L}_{\text{base}}$), MSE feature loss ($\mathcal{L}_{\text{feat}}$), autoregressive chain loss ($\mathcal{L}_{\text{chain}}$), and adjacent-step consistency regularization ($\mathcal{L}_{\text{reg}}$).
   * **Online Collaborative Routing**: Incoming windows are screened at the edge via Mahalanobis distance $\text{Score}(W)$ and output consistency. Routine cases are diagnosed locally at the edge; anomalies exceeding threshold $\tau$ trigger cloud LLM escalation.

---

## 📊 Dataset Description

Experiments are conducted on two real-world industrial IoT (IIoT) syslog datasets collected from production server and network infrastructures. Raw logs are grouped into independent fault cases via content-guided DBSCAN clustering and gap-based session slicing.

* **Dataset 1 (Alibaba Cloud Server Logs)**: Collected over a 4-week period from production cloud servers hosting IIoT backend services ([Tianchi Competition](https://tianchi.aliyun.com/competition/entrance/531947/information)). It contains **35,781** retained log entries grouped into **1,786** fault cases across **3** fault types.
* **Dataset 2 (ZTE 5G Network Device Logs)**: Collected over a 24-day period from production switches and routers in a 5G IIoT network ([`data/dataset2`](https://github.com/syslog-llm/HermesLog/tree/main/data/dataset2)). It contains **59,592** retained log entries grouped into **1,239** fault cases across **9** fault types (e.g., `CRC_WARN`, `CRC_EXCEED`, `MAC_DROP`, `OSPF_DOWN`).

### Data Partition & Offline Cold-Start Protocol
To prevent temporal data leakage, fault cases are partitioned chronologically into an **Unlabeled Candidate Pool**, a **Validation Set**, and a **Held-out Test Set**:

| Datasets | # of retained entries | # of types | # of cases | Candidate | Demo V1 ($\vert\mathcal{D}_{demo}\vert$) | Demo V2 ($\vert\mathcal{D}_{trs}\vert$) | Validation | Test (SLM) | Per SLM (Candidate / Training) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset 1** | 35,781 | 3 | 1,786 | 60 | 54 | 48 | 60 | 1,666 | 20 / 16 |
| **Dataset 2** | 59,592 | 9 | 1,239 | 60 | 48 | 36 | 60 | 1,119 | 20 / 12 |

* **Nested Demo Pool**: Each of the 3 edge sites uploads 20 unlabeled candidate cases (60 total). After LLM confidence filtering ($\theta = 0.90$), 54 and 48 cases are kept as `Demo V1`. Following MLM CRC compilation and filtering ($\gamma_{mid} > \tau_{mid}$), 48 and 36 samples are admitted into `Demo V2` and routed back to their originating sites (averaging 16 and 12 training samples per SLM on Dataset 1 and Dataset 2).
* **Site-Isolated Evaluation**: The 60-case validation set is used strictly for selecting thresholds ($\tau_{mid}, \tau$) and decoding parameters. The held-out test set is divided into three disjoint edge-site partitions (**526 / 579 / 561** cases on Dataset 1; **354 / 375 / 390** cases on Dataset 2). During testing, model weights and demo pools remain frozen.
---


## ⚙️ Installation

**Requirements**

- Python 3.8
- NVIDIA GPU for the local MLM and SLMs (our experiments use NVIDIA A6000; peak memory is about 14 GB for Mistral-7B and 9 to 14 GB for the SLMs)
- API access to GPT-4o (and optionally Claude 3.5 Sonnet)

**Setup**

```bash
git clone https://github.com/syslog-llm/HermesLog.git
cd HermesLog

conda create -n hermeslog python=3.8 -y
conda activate hermeslog
pip install -r requirements/requirements.txt
```

**Models**

| Role | Model | Access |
| :--- | :--- | :--- |
| Cloud LLM (default) | GPT-4o | Official API |
| Cloud LLM (alternative) | Claude 3.5 Sonnet  | Official API |
| MLM | Mistral-7B | Open-source weights from Hugging Face |
| Edge SLMs | Gemma3-4b-it, Phi4-mini-3.8b, Qwen2.5-3b | Open-source weights from Hugging Face |

Set your API keys before running the cloud stage:

```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"   # only for the Claude backbone
```

> The two API snapshots above have been retired by their providers. To rerun the cloud stage, replace them with a currently available snapshot; results may differ slightly from those reported in the paper.

---
## 🚀 Core Code

All scripts are in `code/`. The pipeline has four stages: Stages 1 to 3 run offline, and Stage 4 performs edge diagnosis.

| Stage | Script | Model | Paper module |
| :--- | :--- | :--- | :--- |
| 1 | `stage1_LLM_infer_dataset1.py`, `stage1_LLM_infer_dataset2.py` | GPT-4o / Claude 3.5 Sonnet | FOLFR: zero-shot long-CoT reasoning |
| 2 | `stage2_MLM_CRC_dataset1.py`, `stage2_MLM_CRC_dataset2.py` | Mistral-7B (frozen) | CRC-SCR: Cognitive Relay Compilation |
| 3 | `stage3_SLM_distill.py` | Gemma3-4b-it / Phi4-mini-3.8b / Qwen2.5-3b | Stepwise layered alignment |
| 4 | `stage4_SLM_infer.py` | Aligned edge SLMs | Edge diagnosis with on-demand escalation |

`load_data.py` and `Trainer.py` are shared utilities for data loading and SLM training.

**Stage 1: Long-CoT reasoning with the cloud LLM**

```bash
cd code
python stage1_LLM_infer_dataset1.py   # or stage1_LLM_infer_dataset2.py
```

Reads the candidate pool `60-unlabeled.json`, runs four-stage long-CoT reasoning (observation, analysis, inference, aggregation), and produces "Reasoning V1". Outputs with confidence above θ = 0.90 are kept as "Demo V1".

**Stage 2: Cognitive Relay Compilation with the frozen MLM**

```bash
python stage2_MLM_CRC_dataset1.py     # or stage2_MLM_CRC_dataset2.py
```

Uses "Demo V1" as few-shot demonstrations and compiles each "Reasoning V1" into the compact representation ⟨F, S, L⟩ ("Reasoning V2"). Samples that pass the filtering threshold τ_mid form the transfer set "Demo V2". Mistral-7B only performs inference at this stage.

**Stage 3: Stepwise layered alignment of the edge SLMs**

```bash
python stage3_SLM_distill.py
```

Trains each edge SLM on the "Demo V2" samples of its own site, progressively aligning the diagnosis label L, the key features F, and the short-CoT chain S.

**Stage 4: Edge diagnosis**

```bash
python stage4_SLM_infer.py
```

Runs each aligned SLM on its site partition of the held-out test set (`test1666_part_{1,2,3}.json` for Dataset 1) and outputs the diagnosis label with a short-CoT explanation. Low-confidence cases are escalated to the cloud LLM.

**Default settings**

| Setting | Value |
| :--- | :--- |
| Demo V1 confidence threshold θ | 0.90 |
| Transfer-set filtering threshold τ_mid | 0.90 |
| Maximum short-CoT chain length n_max | 3 |
| Learning rate | 2e-5 |
| Epochs / batch size | 20 / 16 |
| Decoding temperature | 0.6 (Dataset 1), 0.8 (Dataset 2) |
| Top-p | 0.8 |
---


## 💻 Icore code

The pipeline is executed in four sequential stages corresponding to the scripts under `code/`:

### Stage 1: Cloud LLM Zero-Shot Long-CoT Probing (FOLFR)
Executes two-round zero-shot CoT prompting on the 60 unlabeled candidate cases to generate `Reasoning V1` and filter `Demo V1`:
```bash
cd code/Prompt-tuning/LLM-MLM
python Gpt4o_zeroshot-cot_stage1.py
```

### Stage 2: Frozen MLM Cognitive Relay Compilation (CRC-SCR)
Runs frozen `Mistral-7B` with few-shot CoT prompting and vLLM offline sampling to compress `Demo V1` into compact representations $\langle \mathcal{F}, \mathcal{S}, L \rangle$ (`Demo V2`):
```bash
# Generate pseudo few-shot CoT representations
python Mistral_pseudo-fewshot-cot_stage2.py

# Offline vLLM batch sampling and rubric filtering
cd ../../align-tuning/MLM-SLM
python Mistral_vllm-sample-offline-tuning_stage2.py
```

### Stage 3: Stepwise Layered Alignment Preparation & Training
Constructs the multi-level alignment dataset from `Demo V2` and trains the target models:
```bash
python make-align_stage3.py
python Run-train_stage3.py
```

### Stage 4: Edge SLM Progressive Training & Autonomous Diagnosis
Performs progressive curriculum alignment ($lr = 2\times 10^{-5}$, batch size = 16, epochs = 20) and evaluates autonomous inference on the held-out test partitions:
```bash
cd ../../training/SLM
python SLM_load_stage4.py
python Run_trainer_stage4.py

# Run online edge evaluation with zero-shot CoT reasoning
python SLM_vllm-zero-shot-cot-reason_stage4.py
python vllm_sample_offline-stage4.py
```

---

## 📂 Project Structure

```text
HermesLog/
├── code/                        # Icore code
├── data/                        # Input logs (Dataset 1 & Dataset 2)
├── requirements/                # Create an environment
└── README.md                    # Project description
```

```text
Prompt-tuning/LLM-MLM
├── Gpt4o_zeroshot-cot_stage1.py          # Stage 1: LLM four-stage long-CoT probing (FOLFR)
├── data
│   └── output.json                       # Preprocessed candidate log windows
├── Mistral_pseudo-fewshot-cot_stage2.py  # Stage 2: Frozen MLM CRC compilation (CRC-SCR)
├── monitor_gpu.sh                        # GPU memory & energy monitoring script
├── output_stage1
│   ├── Gpt-4o.log
│   └── Gpt-4o_results.json               # Filtered Demo V1 (Reasoning V1)
├── output_stage2-pseudo-fewshot-cot
│   ├── mistral.log
│   └── mistral_results.json              # Compiled Demo V2 (<F, S, L>)
├── README.md
├── test_case_id.txt
└── tree.txt
```

```text
(base) ➜  dataset ls
Gpt4o_zeroshot-cot.py                     output:config-stage1
Mistral_pseudo-fewshot-cot-stage2.py      output:vllm_sample_offline-stage2
monitor_gpu.sh                            tree.txt
output-stage2
```

```text
align-tuning/MLM-SLM
├── config_stage2.py                                  # Hyperparameters & threshold configs
├── Mistral_vllm-sample-offline-tuning_stage2.py      # Offline vLLM compilation & rubric screening
├── make-align_stage3.py                              # output:make_align-stage3 (Build D_trs targets)
├── Run-train_stage3.py                               # output:run_train-stage3 (Alignment training)
└── vllm-sample-offline.py
```

```text
training/SLM
├── SLM_load_stage4.py                                # Load edge SLM backbones (Gemma3/Phi4/Qwen2.5)
├── SLM_vllm-zero-shot-cot-reason_stage4.py           # Online autonomous diagnosis & reasoning
├── Run_trainer_stage4.py                             # Stepwise layered curriculum trainer
└── vllm_sample_offline-stage4.py                     # Held-out test set batch evaluation
```

## 🔗 Links

- Code: https://github.com/syslog-llm/HermesLog
