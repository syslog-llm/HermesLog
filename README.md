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

All scripts are located in `code/`. Following the architecture in the paper, the pipeline consists of four sequential stages (Stages 1–3 for offline cloud-to-edge compilation and alignment, and Stage 4 for online edge diagnosis) supported by two shared utility modules.

| Stage | Script | Model Role | Paper Module & Description |
| :--- | :--- | :--- | :--- |
| **Stage 1** | `stage1_LLM_FOLFR_dataset{1,2}.py` | Cloud LLM (GPT-4o / Claude 3.5 Sonnet) | **FOLFR**: Four-stage zero-shot long-CoT reasoning and self-filtering ($\theta$) |
| **Stage 2** | `stage2_MLM_CRC—SCR_dataset{1,2}.py` | Frozen MLM (Mistral-7B) | **CRC-SCR**: Cognitive Relay Compilation into structured triples $\langle\mathcal{F}, \mathcal{S}, L\rangle$ |
| **Stage 3** | `stage3_SLM_align-tuning.py` | Edge SLMs (Gemma3-4b / Phi4-mini / Qwen2.5-3b) | **Stepwise Layered Alignment**: Progressive alignment of $L$, $\mathcal{F}$, and $\mathcal{S}$ |
| **Stage 4** | `stage4_SLM_diagnosis.py` | Aligned Edge SLMs | **Edge Fault Diagnosis**: Low-latency edge inference with short-CoT explanation |


**Shared Utilities:**
* `load_data.py`: Data loading and preprocessing utilities for candidate pools and site partitions.
* `Trainer.py`: Custom training loop and objective scheduling for Stage 3 stepwise layered alignment.

---

### Step-by-Step Execution

**Stage 1: Zero-Shot Long-CoT Reasoning with Cloud LLM (FOLFR)**

```bash
cd code
python stage1_LLM_FOLFR_dataset1.py   # or stage1_LLM_FOLFR_dataset2.py
```
Reads the unlabeled candidate pool (`60-unlabeled.json`), executes four-stage long-CoT reasoning (*Observation, Analysis, Inference, Aggregation*), and generates `"Reasoning V1"`. Samples with self-evaluated confidence above $\theta = 0.90$ are retained as `"Demo V1"`.

**Stage 2: Cognitive Relay Compilation with Frozen MLM (CRC-SCR)**

```bash
python stage2_MLM_CRC_dataset1.py     # or stage2_MLM_CRC_dataset2.py
```
Uses `"Demo V1"` as in-context demonstrations to guide the frozen Mistral-7B model, compiling verbose `"Reasoning V1"` traces into compact structured representations $\langle\mathcal{F}, \mathcal{S}, L\rangle$ (`"Reasoning V2"`). High-quality compiled samples passing $\tau_{\text{mid}} = 0.90$ form the transfer set `"Demo V2"` and are routed back to their corresponding edge sites.

**Stage 3: Stepwise Layered Alignment of Edge SLMs**

```bash
python stage3_SLM_align-tuning.py
```
Invokes `Trainer.py` to fine-tune each edge SLM on its site-specific `"Demo V2"` transfer set, progressively aligning the diagnostic label ($L$), key anomaly features ($\mathcal{F}$), and compressed reasoning chain ($\mathcal{S}$).

**Stage 4: Edge Fault Diagnosis**

```bash
python stage4_SLM_diagnosis.py
```
Evaluates each aligned edge SLM on its respective site test partition (e.g., `test1666_part_{1,2,3}.json` for Dataset 1), outputting the predicted fault label alongside a concise short-CoT explanation.

---




## 📁 Project Structure

```text
HermesLog/
├── code/
│   ├── Trainer.py                          # Training utilities for SLM layered alignment
│   ├── load_data.py                        # Data loading and partitioning utilities
│   ├── stage1_LLM_FOLFR_dataset1.py        # Stage 1: Cloud LLM long-CoT reasoning (Dataset 1)
│   ├── stage1_LLM_FOLFR_dataset2.py        # Stage 1: Cloud LLM long-CoT reasoning (Dataset 2)
│   ├── stage2_MLM_CRC-SCR_dataset1.py          # Stage 2: Frozen MLM compilation into <F, S, L> (Dataset 1)
│   ├── stage2_MLM_CRC-SCR_dataset2.py          # Stage 2: Frozen MLM compilation into <F, S, L> (Dataset 2)
│   ├── stage3_SLM_align-tuning.py          # Stage 3: Stepwise layered alignment of edge SLMs
│   └── stage4_SLM_diagnosis.py             # Stage 4: Edge fault diagnosis with short-CoT explanation
├── data/
│   ├── dataset1/                           # Server device logs (Alibaba Cloud)
│   │   ├── 60-unlabeled.json               # Unlabeled candidate pool (60 cases)
│   │   ├── 60-unlabeled_part_1.json        # Candidate cases from Edge Site 1 (20 cases)
│   │   ├── 60-unlabeled_part_2.json        # Candidate cases from Edge Site 2 (20 cases)
│   │   ├── 60-unlabeled_part_3.json        # Candidate cases from Edge Site 3 (20 cases)
│   │   ├── output.json                     # Preprocessed log windows after session slicing
│   │   ├── test1666.json                   # Full held-out test set (1,666 cases)
│   │   ├── test1666_part_1.json            # Held-out test partition for Edge Site 1 (526 cases)
│   │   ├── test1666_part_2.json            # Held-out test partition for Edge Site 2 (579 cases)
│   │   ├── test1666_part_3.json            # Held-out test partition for Edge Site 3 (561 cases)
│   │   └── test_case_id.txt                # Case IDs of the held-out test set
│   └── dataset2/                           # Network device logs (ZTE), same layout as dataset1
│       ├── 60-unlabeled.json               # Unlabeled candidate pool (60 cases)
│       ├── 60-unlabeled_part_1.json        # Candidate cases from Edge Site 1 (20 cases)
│       ├── 60-unlabeled_part_2.json        # Candidate cases from Edge Site 2 (20 cases)
│       ├── 60-unlabeled_part_3.json        # Candidate cases from Edge Site 3 (20 cases)
│       └── ...                             # Held-out test set (1,119 cases) and site partitions
├── requirements/
│   └── requirements.txt                    # Python environment dependencies
└── README.md
```


## 🔗 Links

- Code: https://github.com/syslog-llm/HermesLog
