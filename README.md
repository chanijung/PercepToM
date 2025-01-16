This repository contains the code and dataset for the paper ["**Perceptions to Beliefs: Exploring Precursory Inferences for Theory of Mind in Large Language Models**"](https://aclanthology.org/2024.emnlp-main.1105/), presented at EMNLP 2024.

---

## Table of Contents

- [Dataset](#dataset)
  - [Percept-ToMi](#percept-tomi)
  - [Percept-FANToM](#percept-fantom)
- [Code](#code)
  - [Evaluating Language Models on Percept-ToMi](#evaluating-language-models-on-percept-tomi)
  - [Evaluating Language Models on Percept-FANToM](#evaluating-language-models-on-percept-fantom)

---

## Dataset

### Percept-ToMi

**File:** `dataset/Percept-ToMi.csv`

| Column                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `index`                  | Index of each story                                                        |
| `qTypeRaw`               | Type of the question paired with the story (from original annotation of ToMi dataset) |
| `qType`                  | Type of the question paired with the story (with '0' and '1' removed from `qTypeRaw`) |
| `story`                  | Text of the story                                                          |
| `question`               | Text of the question                                                       |
| `char1`                  | The character whose first-order belief the question is asking about         |
| `char2`                  | The character whose second-order belief the question is asking about        |
| `perceivers`             | Perceivers of each scene in the story                                       |
| `story_with_perceivers`  | Scene descriptions in the story paired with corresponding perceivers. JSON-formatted to provide ground truth perception information to the language model. |
| `persp_ctx`              | Perspective context corresponding to the story and the question            |
| `cands`                  | Candidate answer choices                                                   |
| `answer`                 | Answer to the question                                                     |

---

### Percept-FANToM

1. **File:** `dataset/Percept_FANToM/Percept-FANToM.csv`

| Column                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `part_id`                | ID of each conversation part in FANToM                                      |
| `utterance`              | Single utterance in a conversation part                                    |
| `utterance_symbol_removed` | Utterance with symbols removed                                            |
| `perceivers`             | Perceivers of the utterance                                                |

2. **File:** `dataset/Percept_FANToM/Percept-FANToM-conv_with_perceivers.csv`

| Column                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `part_id`                | ID of each conversation part in FANToM                                     |
| `conversation_with_perceivers` | Utterances in the conversation part paired with corresponding perceivers. Formatted in JSON to provide ground truth perception information to the language model. |

---

## Code

### Evaluating Language Models on Percept-ToMi

#### 1. Run Evaluation Script

Execute the evaluation script with the desired model and method:

```bash
python eval_percept_tomi.py --model {model_name} --method {method_name}
```

**Result files saved in `results/Percept-ToMi/{model_name}-{method_name}/`:**

•`acc.csv`: Question answering accuracies for four qTypes — first order/second order, true belief (no ToM) / false belief (ToM)

•`acc_raw.csv`: Question answering accuracies for eight qTypeRaws 

•`result.csv`: Evaluated results of each datapoint, including prompts, model responses, and correctness

**Notes:**

•For all methods except _perception inference_ (`perc_inf`), running this script performs language model inference and evaluation, producing all evaluation results.

•The _perception inference_ method (`perc_inf`) does not perform question answering. Running the script with this method only performs language model inference and produces result.csv.

### 2.**Perception Inference Evaluation (Only for** `perc_inf`**)**

To evaluate perception inference results, run the following script:

```bash
python eval_percept_tomi-perc_inf.py --model {model_name} --method {method_name}
```

**Result files saved in ```results/Percept-ToMi/{model_name}-{method_name}/``` (the same directory as step 1):**

•`evaluated_utterance.csv`: Model predictions and ground truth of scene-level perceivers

•`evaluated_context.csv`: Model predictions and ground truth of context (story)-level perceivers

•`acc_qType.csv`: Perception inference accuracies for four qTypes — first order/second order, true belief (no ToM) / false belief (ToM)

•`acc_scenario.csv`: Perception inference accuracies for two scenarios — true belief (no ToM) / false belief (ToM)

## Evaluating Language Models on Percept-FANToM

### 1.**Run Evaluation Script**

Execute the evaluation script with the desired model and method:

```bash
python eval_percept_fantom.py --model {model_name} --method {method_name}
```

**Result files saved in `results/Percept-FANToM/{model_name}-{method_name}/`:**

• `REPORT_short_input_{model_name}.json`: Question answering accuracies for different question types of FANToM for the contexts of false belief scenarios

• `control_task_report_short_input_{model_name}.json`: Question answering accuracies for different question types of FANToM for the contexts of true belief scenarios

• `evaluated_responses_short_input_{model_name}.json/csv`: Evaluated results of each datapoint, including prompts, model responses, and correctness

**Notes:**

•For all methods except _perception inference_ (`perc_inf`), running this script performs language model inference and evaluation, producing all evaluation results.

•The _perception inference_ method (`perc_inf`) does not include the question answering step. Running the script with this method only performs language model inference and produces `evaluated_responses_short_input_{model_name}.json/csv`.

### 2.**Perception Inference Evaluation (Only for** `perc_inf`**)**

To evaluate perception inference results, run the following script:

```bash
python eval_percept_fantom-perc_inf.py --model {model_name} --method {method_name}
```

**Output:**

The evaluation results will be saved in the same directory as step 1:

`results/Percept-FANToM/{model_name}-{method_name}/`

**Result Files:**

•`evaluated_utterance.csv`: Model predictions and ground truth of utterance-level perceivers

•`evaluated_context.csv`: Model predictions and ground truth of context (part of conversation)-level perceivers

•`perception_inference_accuracy.csv`: Perception inference accuracies for two scenarios — true belief (no ToM) / false belief (ToM)

