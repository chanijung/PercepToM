This repository contains the code and dataset for the paper ["**Perceptions to Beliefs: Exploring Precursory Inferences for Theory of Mind in Large Language Models**"](https://aclanthology.org/2024.emnlp-main.1105/), presented at EMNLP 2024.

## Dataset

### Percept-ToMi

`dataset/Percept-ToMi.csv`

- `index`: Index of each story
- `qtype`: Type of the question paired with the story
- `story`: Text of the story
- `perceivers`: Perceivers of each scene in the story
- `story_with_perceivers`: Scene descriptions in the story paired with corresponding perceivers. Formatted to provide ground truth perception information to the language model. 


### Percept-FANToM

1. `dataset/Percept_FANToM/Percept-FANToM.csv`

   - `part_id`: ID of each conversation part in FANToM
   - `utterance`: Single utterance in a conversation part
   - `utterance_symbol_removed`: Symbol-removed utterance
   - `perceivers`: Perceivers of the utterance

2. `dataset/Percept_FANToM/Percept-FANToM-conv_with_perceivers.csv`

   - `part_id`: ID of each conversation part in FANToM
   - `conversation_with_perceivers`: Utterances in the conversation part paired with corresponding perceivers. Formatted to provide ground truth perception information to the language model. 


## Implementation

This section will be updated soon.
