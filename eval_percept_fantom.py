import os
import json
import argparse
import random
import evaluate
from collections import Counter
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_model
import sys
sys.path.append(os.path.abspath("dataset/Percept_FANToM/task/dataset_loader.py"))
import dataset.Percept_FANToM.task.dataset_loader as loader
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from utils import load_model, final_answer_prompt_formatting
import pprint
import re



class FantomDataset(Dataset):
    def __init__(self, texts, args):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        text = self.texts[index]
        return text
    
def get_target_char(complete_question, second_order, qtype):
    print(qtype)
    if "belief" in qtype:
        try:
            char1 = complete_question.split("Question: What does ")[1].split()[0]
        except:
            if " believe " in complete_question:
                char1 = complete_question.split(" believe ")[0].split()[-1]
            elif " think " in complete_question:
                char1 = complete_question.split(" think ")[0].split()[-1]
        if second_order:
            words = complete_question.split(char1)[1].split()
            for w in words:
                if w!=w.lower():
                    break
            try:
                char2 = w.strip("'s")
            except Exception as e:
                print(e)
                print(complete_question)
                print("char1: ",char1)
        else:
            char2 = None
    elif "list" in qtype:
        char1 = "all"
        char2 = None
    elif "binary" in qtype:
        char1 = complete_question.split("Question: Does ")[1].split()[0]
        char2 = None
    return char1, char2

def get_persp_ctx_first_order(perc, char):
    char_kg = []
    for p in perc:
        if char in list(p.values())[0]:
            char_kg.append(list(p.keys())[0])
    return "\n".join(char_kg)

def get_persp_ctx(context, complete_question, perc, char1, char2):
    if char1=="all":
        all_chars = set()
        for line in context.split("\n"):
            all_chars.add(line.split(":")[0])
        kgt_prompts = []
        for char in all_chars:
            kgt_prompt = get_persp_ctx_prompt(get_persp_ctx_first_order(perc, char), char, None, fakeQ=False)
            kgt_prompts.append(kgt_prompt)
        gt_kgt_prompt = "\n\n".join(kgt_prompts)
    else:
        if (char2!=char2) or (char2 is None):  # first-order questions
            manual_kgt = get_persp_ctx_first_order(perc, char1)
        else:  # second-order questions
            char_kg = []
            for p in perc:
                obs = list(p.values())[0]
                if char1 in obs and char2 in obs:
                    char_kg.append(list(p.keys())[0])
            manual_kgt = "\n".join(char_kg)
        gt_kgt_prompt = get_persp_ctx_prompt(manual_kgt, char1, char2, fakeQ=False)
    return gt_kgt_prompt+f"\n\n{complete_question}"

def get_s2a_prompt(dialog, question):
    prompt = f"""Given the following dialogue, extract the part that is related and useful, so that using that text alone would be good context for providing an accurate and correct answer to the question. Only provide the extracted dialogue without explanation. Do not answer the question.

{dialog}

{question}

Context related to the question (includes all content except unrelated sentences):"""
    return prompt

def get_persp_ctx_prompt(dialog, char1, char2, fakeQ=False):
    if char2 and char2==char2:
        prompt = f"""Here are the past utterances in sequence that {char1} thinks {char2} is aware of.\n\n{dialog}"""
    elif char1:
        prompt = f"""Here are the past utterances in sequence that {char1} is aware of.\n\n{dialog}"""
    else:
        print("no char1 and char2")
        assert(False)
    return prompt

def get_s2a_answering_prompt(s2a_response, question):
    prompt = f"""{s2a_response}

{question}"""
    return prompt


def get_gt_perc_prompt(json_response, question):
    try:
        json_data = json.loads(json_response)
        perc_inf_res = pprint.pformat(json_data, compact=True).replace("'",'"') if type(json_data)==list else json_data
    except Exception as e:
        print(f"{e} - json_response:\n{json_response}")
        perc_inf_res = json_response
    prompt = f"""Each JSON object in the following list represents a consecutive utterance in a dialogue and its audience.

{perc_inf_res}

{question}"""
    return prompt
def extract_json(x):
    return "{" +x.split("{")[1].split("}")[0]+ "}" if "{" in x else ""

def extract_json_array(x):
    begin = re.search('\[\s*\{', x)
    if begin:
        begin_idx = begin.start()
    else:
        print("no beginning of json array")
        return x
    end = re.search('\}\s*\]', x)
    if end:
        end_idx = end.end()
        return x[begin_idx:end_idx]
    else:
        print("no end of json array")
        return x[begin_idx:]

def report_df_with_count(EVAL_DIR_PATH, target_scenario, output_filename_suffix):
    if target_scenario=="inaccessible":
        report_file = "REPORT" + output_filename_suffix
    else:
        report_file = "control_task_report" + output_filename_suffix
    data = json.load(open(os.path.join(EVAL_DIR_PATH, report_file)))

    pf_df = pd.DataFrame(data).round(3).astype(object).set_index(pd.Series(['score', '#questions']))

    wr_df = None
    wr_key = f"{target_scenario}:tom:lists:wrong_reasons:freq"
    if wr_key in data.keys():
        wr = dict([(key, data[wr_key][key]) for key in data[wr_key]])
        wr_df = pd.DataFrame.from_records([wr]).set_index(pd.Series(['score']))

    return pf_df, wr_df

class FantomEvalAgent():
    def __init__(self, args):
        self.args= args
        self.prompt_header = "This is a theory-of-mind test. Please answer the question regarding facts or beliefs, based on the following in-person conversation between individuals who have just met.\n\n"
        self.output_filename_suffix = '_{}_input_{}.json'.format(self.args.conversation_input_type, self.args.model)
        self.load_fantom()
        self.setup_fantom()
        # self.model = self.load_model()
        self.model = load_model(self.args.model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer('sentence-transformers/all-roberta-large-v1').to(self.device)
    def load_fantom(self):
        self.fantom_df = loader.load()
    def respond(self, prompt):
        response = self.model.interact(prompt)
        return response
    def compute_f1(self, ground_truth, model_response):
        """
        Compute the F1 score between the ground truth and model response.
        Args:
            ground_truth (str): The ground truth text.
            model_response (str): The model's response text.
        Returns:
            float: The F1 score.
        """
        ground_truth = ground_truth.split()
        model_response = model_response.split()
        common = Counter(ground_truth) & Counter(model_response)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(model_response)
        recall = 1.0 * num_same / len(ground_truth)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    def evaluate_belief_q(self, qa, model_response, metric='cosine'):
        """
        Evaluate the belief question by comparing the model's response with the correct answer and wrong answer.
        Args:
            qa (dict): A dictionary containing the question and answers.
            model_response (str): The model's response to the question.
            metric (str, optional): The similarity metric to use for comparison. Defaults to 'cosine'.
        Returns:
            tuple: A tuple containing a boolean value indicating if the model's response matches the correct answer,
                   and the lexical overlap score between the model's response and the corresponding answer.
        """
        wrong_tom_view = qa['wrong_answer']
        if metric == "cosine":
            wrong_tom_view_emb = self.embedder.encode(wrong_tom_view)
            personx_view_emb = self.embedder.encode(qa['correct_answer'])
            model_response_emb = self.embedder.encode(model_response)
            similarity_wrong_tom_view = cosine_similarity(model_response_emb.reshape(1, -1), wrong_tom_view_emb.reshape(1, -1))[0][0]
            similarity_personx_view = cosine_similarity(model_response_emb.reshape(1, -1), personx_view_emb.reshape(1, -1))[0][0]
        else:
            raise NotImplementedError
        if similarity_wrong_tom_view >= similarity_personx_view:
            wrong_view_lexical_overlap = self.compute_f1(wrong_tom_view, model_response)
            return False, wrong_view_lexical_overlap
        else:
            personx_view_lexical_overlap = self.compute_f1(qa['correct_answer'], model_response)
            return True, personx_view_lexical_overlap
    def evaluate_mc_belief_q(self, qa, model_response):
        """
        Evaluate the multiple-choice version belief question.
        Args:
            qa (dict): The question and answer information.
            model_response (str): The model's response to the question.
        Returns:
            bool: True if the model's response matches the correct answer, False otherwise.
        """
        int_to_alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        answer = int_to_alphabet[int(qa['correct_answer'])]
        response = model_response.lower()
        if response.startswith("(" + answer + ")") or response.startswith(answer + ")") or response.startswith(answer + ".") or response.startswith(answer + ":") or response.startswith(answer + ",") or "({})".format(answer) in response or answer == response: # a) or a. or a or (a)
            return True
        else:
            return False
    def evaluate_list_q_bracket(self, qa, model_response):
        """
        Check whether all the characters in the correct answer are in the model's response
        and none of the characters in the wrong answer are in the model's response
        Args:
            qa (dict): A dictionary containing the question and answer information.
            model_response (str): The response generated by the model.
        Returns:
            tuple: A tuple containing three values:
                - A boolean indicating whether the model's response satisfies the evaluation criteria.
                - A boolean indicating whether any aware characters were excluded from the model's response.
                - A boolean indicating whether any unaware characters were included in the model's response.
        """
        if model_response.count("[")>1:
            print(f"more than one brackets:\n{model_response}")
            answer_span = model_response
        elif model_response.count("[")==0 or model_response.count("]")==0:
            if self.args.method!="cot":
                print(f"no bracket:\n{model_response}")
            answer_span = model_response
        else:    
            answer_span = model_response[model_response.index("["):model_response.index("]")+1]
        excluded_aware_character = False
        included_unaware_character = False
        if type(qa['correct_answer'])==str:
            qa['correct_answer'] = str_to_list(qa['correct_answer'])
            qa['wrong_answer'] = str_to_list(qa['wrong_answer'])
        for character in qa['correct_answer']:
            if character.lower() not in answer_span.lower():
                excluded_aware_character = True
                break
        for character in qa['wrong_answer']:
            if character.lower() in answer_span.lower():
                included_unaware_character = True
                break
        return not(excluded_aware_character or included_unaware_character), excluded_aware_character, included_unaware_character, answer_span
    def map_binary_answer_to_int(self, model_response):
        """
        Maps a binary answer to an integer value.
        Args:
            model_response (str): The model's response.
        Returns:
            int: The mapped integer value. Returns 1 for positive answers (e.g., 'yes', 'true'), 
                 0 for negative answers (e.g., 'no', 'false'), and -1 for other cases.
        """
        model_answer = model_response.lower().strip("'").strip('"')
        if " yes," in model_answer or " yes " in model_answer or model_answer.startswith("yes") or " yes." in model_answer or " knows " in model_answer or model_answer.lower().startswith("true"):
            return 1
        elif " no," in model_answer or " no " in model_answer or model_answer.startswith("no") or " no." in model_answer or " does not know " in model_answer or " doesn't know " in model_answer or model_answer.lower().startswith("false"):
            return 0
        else:
            return -1
    def evaluate_binary_q_with_f1(self, qa, model_response):
        """
        Evaluates a binary question with F1 score.
        Args:
            qa (dict): A dictionary containing the question and correct answer.
            model_response (str): The response generated by the model.
        Returns:
            bool: True if the model's response contains the correct answer, False otherwise.
        """
        tom_answer = qa['correct_answer'].split(":")[0] # for no:long
        model_answer = model_response.split()[0].lower().strip(",")
        if tom_answer in model_answer:
            return True
        else:
            return False
    def evaluate_fact_q(self, qa, model_response):
        result = self.compute_f1(qa['correct_answer'].lower(), model_response.lower())
        return result
    def yesno_to_int(self, yesno_str):
        mapping = {'yes': 1, 'no': 0, 'no:long': 0, 'error': -1}
        return mapping[yesno_str]
    def evaluate_response(self, qas, predictions):
        """
        Evaluates the model's response for a list of questions and predictions.
        Args:
            qas (list): List of question-answer pairs.
            predictions (list): List of model predictions.
        Returns:
            list: Updated list of question-answer pairs with evaluation results and predictions.
        """
        print("Running evaluation...")
        assert len(qas) == len(predictions), "Number of questions and model predictions should be the same."
        for qa, pred in tqdm(zip(qas, predictions), total=len(qas)):
            if qa['question_type'].startswith("tom:belief:"):
                if qa['question_type'].endswith(":multiple-choice"):
                    result = self.evaluate_mc_belief_q(qa, pred)
                else:
                    result, word_overlap = self.evaluate_belief_q(qa, pred)
                    qa['word_overlap'] = word_overlap
            elif qa['question_type'].endswith(":list"):
                result, excluded_aware_character, included_unaware_character, answer_span = self.evaluate_list_q_bracket(qa, pred)
                qa['excluded_aware_character'] = excluded_aware_character
                qa['included_unaware_character'] = included_unaware_character
                qa['prediction_answer_span'] = answer_span
            elif qa['question_type'].endswith(":binary"):
                _binary_answer = self.map_binary_answer_to_int(pred)
                if self.yesno_to_int(qa['correct_answer']) == _binary_answer:
                    result = True
                else:
                    result = False
                qa['binarized_model_answer'] = _binary_answer
            elif qa['question_type'].startswith("fact"):
                result = self.evaluate_fact_q(qa, pred)
            else:
                raise NotImplementedError
            qa['result'] = result
            qa['prediction'] = pred
        return qas

    def score_and_analyze(self, df, target_scenario='inaccessible'):
        """
        Aggregates scores and performs analysis on the model responses and evaluation results.
        Args:
            df (pandas.DatfaFrame): The dataframe containing the FANToM QA pairs, model responses, and evaluation results.
            target_scenario (str, optional): The target scenario for analysis. Defaults to 'inaccessible'.
        Returns:
            dict: A dictionary containing the calculated scores and analysis results.
        """
        ff = pd.DataFrame(self.flattened_fantom)
        report = {}
        f1_metric = evaluate.load("f1")
        
        if type(df.result.iloc[0])==str:
            df['result'] = df['result'].map(lambda x:x=='True' if x.endswith('e') else float(x))
        tom_df = df[df['question_type'].str.startswith("tom")].copy()
        target_df = tom_df[tom_df['missed_info_accessibility'] == target_scenario].copy()
        
        if target_scenario == 'accessible':
            # Filter out the set_ids that have all the questions that are labeled as accessible for the ALL* and ALL scores
            # This is because in sets where there are belief questions labeled as 'inaccessible' (i.e., there is an unaware character), all the other question types are also treated as 'inaccessible'.
            # As a result, in the accessible scenario, there are many sets that are only left with a few belief questions. This leads to exaggerated ALl* and ALL scores.
            # As a quick & dirty solution, we will focus only on the sets where all the questions are labeled as accessible when measuring the the ALL* and ALL scores.
            _target_df = tom_df[tom_df['missed_info_accessibility'] == target_scenario].copy()
            set_ids = _target_df['set_id'].unique()
            target_sets = []
            for set_id in set_ids:
                if tom_df[tom_df['set_id'] == set_id]['missed_info_accessibility'].eq(target_scenario).all():
                    target_sets.append(set_id)
        else:
            target_sets = target_df['set_id'].unique()
        

        ############ Scores #############
        # ALL score
        df1 = target_df[target_df['set_id'].isin(target_sets)].groupby("set_id")['result'].all()
        report[target_scenario+':set:ALL*'] = [df1.mean(), len(df1)]
        report[target_scenario+':set:ALL'] = [df1.mean(), len(df1)]

        # Belief Questions: multiple-choice
        df1 = target_df[target_df['question_type'].str.endswith(":multiple-choice")]['result']
        report[target_scenario+':belief:multiple-choice'] = [df1.mean(), len(df1)]

        # Answerability Questions: ALL, list, binary
        df1 = target_df[target_df['question_type'].str.startswith("tom:answerability")].groupby("set_id")['result'].all()
        report[target_scenario+':answerability:set:ALL'] = [df1.mean(), len(df1)]
        df1 = target_df[target_df['question_type'] == "tom:answerability:list"]['result']
        report[target_scenario+':answerability:list'] = [df1.mean(), len(df1)]
        if 'binarized_model_answer' in target_df.columns:
            answerability_model_responses = target_df[target_df['question_type'] == 'tom:answerability:binary']['binarized_model_answer'].to_list()
            answerability_references = target_df[target_df['question_type'] == 'tom:answerability:binary']['correct_answer'].map(self.yesno_to_int).to_list()
            report[target_scenario+':answerability:binary-f1'] = [f1_metric.compute(predictions=answerability_model_responses, references=answerability_references, pos_label=0, average="weighted")['f1'], len(answerability_model_responses)]

        # Info Accessibility Questions: All, list, binary
        df1 = target_df[target_df['question_type'].str.startswith("tom:info_accessibility")].groupby("set_id")['result'].all()
        report[target_scenario+':info_accessibility:set:ALL'] = [df1.mean(), len(df1)]
        df1 = target_df[target_df['question_type']=="tom:info_accessibility:list"]['result']
        report[target_scenario+':info_accessibility:list'] = [df1.mean(), len(df1)]
        if 'binarized_model_answer' in target_df.columns:
            accessibility_model_responses = target_df[target_df['question_type'] == 'tom:info_accessibility:binary']['binarized_model_answer'].to_list()
            accessibility_references = target_df[target_df['question_type'] == 'tom:info_accessibility:binary']['correct_answer'].map(self.yesno_to_int).to_list()
            report[target_scenario+':info_accessibility:binary-f1'] = [f1_metric.compute(predictions=accessibility_model_responses, references=accessibility_references, pos_label=0, average="weighted")['f1'], len(accessibility_model_responses)]

        # Fact Questions
        df1 = df[df['question_type'].str.startswith("fact")]['result']
        report['fact_word-f1'] = [df1.mean(), len(df1)]

        ############ Error Analysis #############
        # why the model got the list questions wrong: only for answerability
        if "tom:answerability:list" in target_df['question_type'].unique() and 'excluded_aware_character' in target_df.columns:
            list_wrong = target_df[(target_df['question_type']=="tom:answerability:list") & (target_df['result'] == False)][['excluded_aware_character', 'included_unaware_character']].copy()
            list_wrong['both'] = list_wrong['excluded_aware_character'] & list_wrong['included_unaware_character']
            list_wrong['reason'] = list_wrong.apply(lambda x: 'did_both' if x['both'] else 'excluded_aware_character' if x['excluded_aware_character'] else 'included_unaware_character', axis=1)
            report[target_scenario+':tom:lists:wrong_reasons:freq'] = list_wrong['reason'].value_counts(normalize=False).to_dict()

        # why the model got the binary questions wrong
        if 'binarized_model_answer' in target_df.columns:
            binary_wrong_reasons = target_df[(target_df['question_type'].str.endswith(":binary")) & (target_df['result'] == False)]['binarized_model_answer'].value_counts(normalize=False).to_dict()
            if 0 in binary_wrong_reasons.keys():
                binary_wrong_reasons['false_negative'] = binary_wrong_reasons.pop(0)
            if 1 in binary_wrong_reasons.keys():
                binary_wrong_reasons['false_positive'] = binary_wrong_reasons.pop(1)
            if -1 in binary_wrong_reasons.keys():
                binary_wrong_reasons['irrelevant_response'] = binary_wrong_reasons.pop(-1)
            report[target_scenario+':tom:binary:wrong_reasons:freq'] = binary_wrong_reasons
            
        ############# More Analysis #############
        # 1. Results for each tom_order type in Belief questions: first order and second order
        if "tom:belief:inaccessible:multiple-choice" in tom_df.question_type.unique():
            belief_df = tom_df[tom_df['question_type'] == ('tom:belief:' + target_scenario+":multiple-choice")].copy() # XXX: only consider the BeliefQ[choice] questions
            belief_df['tom_order'] = belief_df['tom_type'].map(lambda x: x.split(":")[0])
            df1 = belief_df.groupby('tom_order')['result']
            tom_order_results = df1.value_counts(normalize=True)
            tom_order_counts = df1.value_counts()
            for idx in tom_order_results.index:
                if idx[1] == True:
                    report[target_scenario + ":" + idx[0]] = tom_order_results[idx], int(tom_order_counts[idx[0]].sum())

        # 2. Cyclic vs Acyclic second order belief questions
            df1 = belief_df.groupby('tom_type')['result']
            belief_results = df1.value_counts(normalize=True)
            belief_counts = df1.value_counts()
            for idx in belief_results.index:
                if idx[1] == True:
                    report[target_scenario + ":" + idx[0]] = belief_results[idx], int(belief_counts[idx[0]].sum())
                    
        # 3. Character tracking analysis 
        binary_qas = ff[(ff['question_type'].str.endswith(":binary"))].copy()
        binary_qas['target_character'] = binary_qas['question'].map(lambda x: x.removeprefix("Does ").split(" know")[0].lower())
        belief_qas = target_df[(target_df['question_type'].str.startswith("tom:belief"))].copy()
        belief_qas['target_character'] = belief_qas['question'].map(lambda x: x.lower().split("does ")[1].split()[0].lower())
        answerability_list_qas = target_df[target_df['question_type'].str.endswith("answerability:list")].set_index("set_id", drop=False)
        accessibility_list_qas = target_df[target_df['question_type'].str.endswith("info_accessibility:list")].set_index("set_id", drop=False)

        # Tile the list question responses to the binary question level for each character
        binary_answerability_qas = binary_qas[binary_qas['question_type'].str.startswith('tom:answerability:')]
        tiled_answerability_list_qas = binary_answerability_qas[["set_id", 'target_character', 'correct_answer']].join(answerability_list_qas[['prediction', "set_id"]], on="set_id", how='outer', lsuffix='-binary')
        tiled_answerability_list_qas['binarized_model_answer'] = tiled_answerability_list_qas.apply(lambda x: str(x['target_character']).lower() in str(x['prediction']).lower(), axis=1)
        tiled_answerability_list_qas['binarized_correct_answer'] = tiled_answerability_list_qas['correct_answer'].map(lambda x: True if x =='yes' else False)
        tiled_answerability_list_qas['result'] = tiled_answerability_list_qas.apply(lambda x: x['binarized_model_answer'] == x['binarized_correct_answer'], axis=1)
        binary_accessibility_qas = binary_qas[binary_qas['question_type'].str.startswith('tom:info_accessibility:')]
        tiled_accessibility_list_qas = binary_accessibility_qas[["set_id", 'target_character', 'correct_answer']].join(accessibility_list_qas[['prediction', "set_id"]], on="set_id", how='outer', lsuffix='-binary')
        tiled_accessibility_list_qas['binarized_model_answer'] = tiled_accessibility_list_qas.apply(lambda x: str(x['target_character']).lower() in str(x['prediction']).lower(), axis=1)
        tiled_accessibility_list_qas['binarized_correct_answer'] = tiled_accessibility_list_qas['correct_answer'].map(lambda x: True if x =='yes' else False)
        tiled_accessibility_list_qas['result'] = tiled_accessibility_list_qas.apply(lambda x: x['binarized_model_answer'] == x['binarized_correct_answer'], axis=1)
        df_for_all_character_metric = pd.concat([belief_qas[['target_character', "set_id", 'result']], tiled_answerability_list_qas[['target_character', "set_id", 'result']], tiled_accessibility_list_qas[['target_character', "set_id", 'result']]])
        df1 = df_for_all_character_metric.groupby(["set_id", 'target_character'])['result'].all()
        report[target_scenario+':set:ALL_character'] = df1.mean(), len(df1)
        df_for_character_consistency = pd.concat([tiled_answerability_list_qas[['target_character', "set_id", 'binarized_model_answer']], tiled_accessibility_list_qas[['target_character', "set_id", 'binarized_model_answer']]])
        df1 = df_for_character_consistency.reset_index(drop=True).groupby(["set_id", 'target_character'])['binarized_model_answer'].nunique().eq(1)
        report[target_scenario+':set:character_answer_consistency'] = df1.mean(), len(df1) # how often the model gives the "same answer" for the list questions for the same character

        report_df = pd.DataFrame(report, index=['score', '#questions'])
        report_df = report_df.drop(f"{target_scenario}:set:ALL*",axis=1).round(3).astype(object)
        report_df.dropna(axis=1, how='any', inplace=True)
        report = {key: report_df[key].to_dict() for key in report_df.columns}
        
        return report
    
    
    def run_reports(self, qa_results):
        """
        Create report after scoring and analyzing the results
        Input:
        - qa_results: a list of qa results
        Output:
        - report: a dic        - report: a dictionary of scores and analysis
tionary of scores and analysis
        Note:
        We can further increase the difficulty of the task by changing the aggregation target from 'set_id' to 'part_id' or 'conversation_id'.
        A conversation part refers to the brief section of the conversation that is the relevant part to the question.
        Each conversation part comprises multiple sets of questions, and every conversation consists of multiple conversation parts.
        For instance, if you designate 'part_id' as the aggregation target, the ALL scores will be aggregated for each individual part of the conversation.
        This adjustment will result in the ALL score being aggregated across multiple sets.
        Currently, the default conversation-input-type is 'short' and the ALL scores are aggregated for each set of questions (i.e., aggregation-target to 'set'), which will be the easiest setup for the models.
        The most difficult setup will be to give the full conversation input to the model (i.e., conversation-input-type to 'full') and aggregate the ALL scores for each conversation (i.e., aggregation-target to 'conversation_id')
        """
        df = pd.DataFrame(qa_results)

        # Drop binary questions with no:long answer when input type is short
        if self.args.conversation_input_type == "short":
            df.drop(df[(df['question_type'].str.endswith(":binary")) & (df['correct_answer'] == 'no:long')].index, inplace=True)
        df['conversation_id'] = df['set_id'].map(lambda x: x.split("-")[0])
        df['part_id'] = df['set_id'].map(lambda x: "-".join(x.split("-")[:2]))

        report = self.score_and_analyze(df, target_scenario='inaccessible') 
        control_question_report = self.score_and_analyze(df, target_scenario='accessible') 
        
        reports = {'fantom': report, 'control_task': control_question_report}
        
        print("\n[[ FANToM input type: {} ]]".format(self.args.conversation_input_type))
        print("[[ Model: {} ]]\n".format(self.args.model))
        for k, v in reports['fantom'].items():
            print(k, ":", v)
            print()
        return reports
    def dump_report_outputs(self, reports, evaluation_outputs):
        """
        Dump the reports and the evaluation outputs
        """
        evaluated_responses_filename = "evaluated_responses" + self.output_filename_suffix
        output_dict = {'model': self.args.model, 'results': evaluation_outputs}
        os.makedirs(EVAL_DIR_PATH, exist_ok=True)
        with open(os.path.join(EVAL_DIR_PATH, evaluated_responses_filename), 'w') as f:
            json.dump(output_dict, f, indent=4)
        controlq_report_filename = "control_task_report" + self.output_filename_suffix
        with open(os.path.join(EVAL_DIR_PATH, controlq_report_filename), 'w') as f:
            json.dump(reports['control_task'], f, indent=4)
        report_filename = "REPORT" + self.output_filename_suffix
        with open(os.path.join(EVAL_DIR_PATH, report_filename), 'w') as f:
            json.dump(reports['fantom'], f, indent=4)
        print(">>>>> Dumped evaluation outputs and the report at {}!".format(EVAL_DIR_PATH))
        print(">>>>> Evaluated model responses filename: {}".format(evaluated_responses_filename))
        print(">>>>> REPORT filename: {}".format(report_filename))
    def set_beliefQA_multiple_choices(self, qa):
        if qa['question_type'].endswith(":inaccessible"):
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']
        else:
            option_a = qa['wrong_answer']
            option_b = qa['correct_answer']
        answer_goes_last = random.choice([True, False])
        if answer_goes_last:
            choices = [option_a, option_b]
            answer = 1
        else:
            choices = [option_b, option_a]
            answer = 0

        # option letters iterate over the alphabet
        option_letters = ["(" + chr(x) + ")" for x in range(ord('a'), len(choices) + ord('a'))]
        choices_text = ""
        for letter, option in zip(option_letters, choices):
            choices_text += "{} {}\n".format(letter, option)
        return choices_text, answer
    def setup_fantom(self):
        """
        Flatten the dictionary and add short and full conversation context to each question.
        The result will be a list of questions and list of short or full inputs to be used as input for the models.
        """
            
        # Fix typos in context
        df = self.fantom_df
        typo_index = df[df.short_context.map(lambda x:"Mackenzie" in x and "Makenzie" in x)].index
        df.loc[typo_index,'short_context'] = df.loc[typo_index].short_context.map(lambda x: x.replace("Makenzie", "Mackenzie"))

        typo_index = df[df.short_context.map(lambda x:"Everett" in x and "Everevt" in x)].index
        df.loc[typo_index,'short_context'] = df.loc[typo_index].short_context.map(lambda x: x.replace("Everevt", "Everett"))

        typo_index = df[df.short_context.map(lambda x:"extricurricular activities" in x)].index
        df.loc[typo_index,'short_context'] = df.loc[typo_index].short_context.map(lambda x: x.replace("extricurricular", "extracurricular"))

        percept_fantom_df = pd.read_csv(os.path.join(DATA_DIR,"Percept-FANToM.csv"))
        df = df[df.part_id.isin(percept_fantom_df.part_id)]
        
        # Sampling conversations
        if self.args.num_conv:
            in_sample = df.conv_id.apply(lambda x: x in range(self.args.num_conv))
            self.fantom_df = df[in_sample]
        else:
            self.fantom_df = df

        perc_inf_prompt_suffix = """Create a JSON array consisting of JSON objects. \
        Each object should include an utterance from the dialogue and the audience for that utterance. \
        Assume that characters in the story can hear every utterance that occurs while they are involved \
        in the dialogue, but not those that occur when they are absent. \
        Also, ensure that the speaker of each utterance is included in the audience. \
        Provide only the JSON array in the following format. Do not include any explanations.
        [{"Noah: Hi, Emma.": ["Noah", "Emma"]},]
        """
            
        self.fantom_df_to_run = self.fantom_df

        inputs = []
        qas = []
        for idx, _set in self.fantom_df_to_run.iterrows():
            if self.args.conversation_input_type == "short":
                context = _set['short_context'].strip()
            elif self.args.conversation_input_type == "full":
                context = _set['full_context'].strip()
            
            set_id = _set['set_id']
            part_id = _set['part_id']
            fact_q = _set['factQA']['question']
            fact_a = _set['factQA']['correct_answer']

            # Fact Question
            _set['factQA']['context'] = context
            input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, fact_q)
            _set['factQA']['input_text'] = input_text
            _set['factQA']['set_id'] = set_id
            _set['factQA']['part_id'] = part_id
            # Do not evaluate on fact questions
#             qas.append(_set['factQA'])
#             inputs.append(input_text)
            for _belief_qa in _set['beliefQAs']:
                # Belief Questions
                _belief_qa['context'] = context
                one_sent_instr = "Answer in one sentence."
                input_text = "{}\n\nQuestion: {} {}\nAnswer:".format(context, _belief_qa['question'], one_sent_instr)
                _belief_qa['input_text'] = input_text
                _belief_qa['set_id'] = set_id
                _belief_qa['part_id'] = part_id
                # Do not evaluate on BELIEFQ[DIST.]
#                 qas.append(_belief_qa)
#                 inputs.append(input_text)

                # Multiple Choice Belief Questions
                _mc_belief_qa = {**_belief_qa}
                choices_text, answer = self.set_beliefQA_multiple_choices(_mc_belief_qa)
                mc_question_plain = "Question: {} Choose between (a) and (b).\n{}".format(_belief_qa['question'], choices_text.strip())
                mc_question = "Question: {} Choose between (a) and (b). Do not include any explanation.\n{}\n\nAnswer:".format(_belief_qa['question'], choices_text.strip())
                
                if self.args.method in ["perc_inf", "perceptom"]:
                    input_text = f"{context}\n\n{perc_inf_prompt_suffix}"
                elif self.args.method=="s2a":
                    input_text = get_s2a_prompt(context, mc_question_plain)
                else:
                    if self.args.method=="cot":
                        mc_question = mc_question_plain
                    input_text = "{}\n\n{}".format(context, mc_question)
                
                _mc_belief_qa['complete_question'] = mc_question
                _mc_belief_qa['question_type'] = _mc_belief_qa['question_type'] + ":multiple-choice"
                _mc_belief_qa['choices_text'] = choices_text
                _mc_belief_qa['choices_list'] = choices_text.strip().split("\n")
                _mc_belief_qa['correct_answer'] = answer
                _mc_belief_qa['input_text'] = input_text
                qas.append(_mc_belief_qa)
                inputs.append(input_text)

            # Answerability List Questions
            _set['answerabilityQA_list']['fact_question'] = fact_q
            _set['answerabilityQA_list']['context'] = context
            list_add_instr_plain = "Include the character who knows the answer before the dialogue happens, too. Provide the names in a square bracket (e.g., [character_1, character_2, ..., character_n])."
            question = "Target: {}\nQuestion: {} {} Do not include any explanation.\n\nAnswer:".format(fact_q, _set['answerabilityQA_list']['question'], list_add_instr_plain)
            question_plain = "Target: {}\nQuestion: {} {}".format(fact_q, _set['answerabilityQA_list']['question'], list_add_instr_plain)
            if self.args.method in ["perc_inf", "perceptom"]:
                input_text = f"{context}\n\n{perc_inf_prompt_suffix}"
            elif self.args.method=="s2a":
                input_text = get_s2a_prompt(context, question_plain)
            else:
                if self.args.method == "cot":
                    question = question_plain
                input_text = "{}\n\n{}".format(context, question)
            _set['answerabilityQA_list']['input_text'] = input_text
            _set['answerabilityQA_list']['set_id'] = set_id
            _set['answerabilityQA_list']['part_id'] = part_id
            _set['answerabilityQA_list']['complete_question'] = question
            if self.args.conversation_input_type == "full" and len(_set['answerabilityQA_list']['wrong_answer']) > 0:
                _set['answerabilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['answerabilityQA_list'])
            inputs.append(input_text)

            # Answerability Binary Questions
            if self.args.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['answerabilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['answerabilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'
            
            for _answerability_qa in _set['answerabilityQAs_binary']:
                question = "Target: {}\nQuestion: {} Answer yes or no without explnation.\nAnswer:".format(fact_q, _answerability_qa['question'])
                _answerability_qa['fact_question'] = fact_q
                _answerability_qa['context'] = context
                if self.args.method in ["perc_inf", "perceptom"]:
                    input_text = f"{context}\n\n{perc_inf_prompt_suffix}"
                elif self.args.method=="s2a":
                    input_text = get_s2a_prompt(context, question.strip("\nAnswer:"))
                else:
                    if self.args.method=="cot":
                        question = "Target: {}\nQuestion: {} Answer yes or no.".format(fact_q, _answerability_qa['question'])
                    input_text = "{}\n\n{}".format(context, question)
                _answerability_qa['input_text'] = input_text
                _answerability_qa['set_id'] = set_id
                _answerability_qa['part_id'] = part_id
                _answerability_qa['complete_question'] = question
                if self.args.conversation_input_type == "full":
                    _answerability_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_answerability_qa)
                inputs.append(input_text)

            # Info Accessibility List Questions
            _set['infoAccessibilityQA_list']['fact_question'] = fact_q
            _set['infoAccessibilityQA_list']['fact_answer'] = fact_a
            _set['infoAccessibilityQA_list']['context'] = context
            question = "Information: {} {}\nQuestion: {} {} Do not include any explanation.\n\nAnswer:".format(fact_q, fact_a, _set['infoAccessibilityQA_list']['question'], list_add_instr_plain)
            question_plain = "Information: {} {}\nQuestion: {} {} ".format(fact_q, fact_a, _set['infoAccessibilityQA_list']['question'], list_add_instr_plain)
            if self.args.method in ["perc_inf", "perceptom"]:
                input_text = f"{context}\n\n{perc_inf_prompt_suffix}"
            elif self.args.method=="s2a":
                input_text = get_s2a_prompt(context, question_plain)
            else:
                if self.args.method=="cot":
                    question = question_plain
                input_text = "{}\n\n{}".format(context, question)
            _set['infoAccessibilityQA_list']['input_text'] = input_text
            _set['infoAccessibilityQA_list']['set_id'] = set_id
            _set['infoAccessibilityQA_list']['part_id'] = part_id
            _set['infoAccessibilityQA_list']['complete_question'] = question
            if self.args.conversation_input_type == "full" and len(_set['infoAccessibilityQA_list']['wrong_answer']) > 0:
                _set['infoAccessibilityQA_list']['missed_info_accessibility'] = 'inaccessible'
            qas.append(_set['infoAccessibilityQA_list'])
            inputs.append(input_text)
            
            # Info Accessibility Binary Questions
            if self.args.conversation_input_type == "full":
                missed_info_accessibility_for_full = _set['infoAccessibilityQAs_binary'][0]['missed_info_accessibility']
                for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                    if _info_accessibility_qa['correct_answer'] != "yes":
                        missed_info_accessibility_for_full = 'inaccessible'
            
            for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                question = "Information: {} {}\nQuestion: {} Answer yes or no without explanation.\nAnswer:".format(fact_q, fact_a, _info_accessibility_qa['question'])
                _info_accessibility_qa['fact_question'] = fact_q
                _info_accessibility_qa['fact_answer'] = fact_a
                _info_accessibility_qa['context'] = context
                if self.args.method in ["perc_inf", "perceptom"]:
                    input_text = f"{context}\n\n{perc_inf_prompt_suffix}"
                elif self.args.method=="s2a":
                    input_text = get_s2a_prompt(context, question.strip("\nAnswer:"))
                else:
                    if self.args.method=="cot":
                        question = "Information: {} {}\nQuestion: {} Answer yes or no.".format(fact_q, fact_a, _info_accessibility_qa['question'])
                    input_text = "{}\n\n{}".format(context, question)
                _info_accessibility_qa['input_text'] = input_text
                _info_accessibility_qa['set_id'] = set_id
                _info_accessibility_qa['part_id'] = part_id
                _info_accessibility_qa['complete_question'] = question
                if self.args.conversation_input_type == "full":
                    _info_accessibility_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
                qas.append(_info_accessibility_qa)
                inputs.append(input_text)


        self.inputs = inputs
        self.flattened_fantom = qas
    def parse_response(self, response):
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "Choose an answer from above:" in response:
            response = response.split("Choose an answer from above:")[-1].strip()
        return response
    def get_last_savepoint(self):
        responses_filename = "model_responses" + self.output_filename_suffix + "l" # jsonl
        model_responses_filename_path = os.path.join(EVAL_DIR_PATH, responses_filename)
        
        # Check if model outputs file exists
        if os.path.exists(model_responses_filename_path):
            print("File {} exists. Reading responses from file...".format(model_responses_filename_path))
            df = pd.read_json(model_responses_filename_path, lines=True)
            if len(df) > 0:
                last_idx = df.iloc[-1]['index']
                model_responses = df['response'].tolist()
            else:
                last_idx = -1
                model_responses = []
        else:
            last_idx = -1
            model_responses = []
        
        return last_idx, model_responses, model_responses_filename_path
    def get_final_input_json(self, input_prompt, idx):
        if self.args.method=="ptob_inf":
            input_prompt = get_gt_perc_prompt(input_prompt, self.flattened_fantom[idx]['complete_question'])
        elif self.args.method=="perceptom":
            try:
                qtype = self.flattened_fantom[idx]['question_type']
                tom_type = self.flattened_fantom[idx]['tom_type'] if "belief" in qtype else "first"
                char1, char2 = get_target_char(self.flattened_fantom[idx]['complete_question'], "second" in tom_type, qtype)
                self.flattened_fantom[idx]['target_character1'] = char1
                self.flattened_fantom[idx]['target_character2'] = char2
            except Exception as e:
                print(f"{e}:\n{self.flattened_fantom[idx]['complete_question']}")
                return e
            try:
                perc_inf_res = self.flattened_fantom[idx]['perc_inf_response']
                perc = json.loads(perc_inf_res)
            except Exception as e:
                print(f"{e}:\nerror in json.loads\n{perc_inf_res}")
                return "invalid perception inference response"
            persp_ctx = get_persp_ctx(self.flattened_fantom[idx]['context'], self.flattened_fantom[idx]['complete_question'], perc, char1, char2)
            input_prompt = persp_ctx
        return input_prompt


    def run_inference(self):
        print("run_inference")
        target_data = self.inputs
        
        model_responses = []

        # Load model responses if exists
        last_idx, model_responses, response_filename_path = self.get_last_savepoint()
        
        # Load self.flattened_fantom if exists
        ff_path = os.path.join(EVAL_DIR_PATH, "flattened_fantom.csv")
        if os.path.exists(ff_path):
            self.flattened_fantom = pd.read_csv(ff_path).to_dict('records')

        if self.args.method=="perceptom":
            eval_res_df_path = os.path.join(PERC_INF_RESULT_DIR, f"evaluated_responses_short_input_{args.model}.csv")
            if os.path.exists(eval_res_df_path):
                print(">>> Reading perception inference responses from file... ")
                perc_df = pd.read_csv(eval_res_df_path, index_col="index") 
                ff_df = put_index(pd.DataFrame(self.flattened_fantom)).set_index('index')
                perc_inf_cols = ['raw_perc_inf_response', 'perc_inf_response']
                ff_df.loc[:,'perc_inf_prompt'] = perc_df.loc[ff_df.index, 'final_input_prompt']
                ff_df.loc[:,perc_inf_cols] = perc_df.loc[ff_df.index, perc_inf_cols]
                self.flattened_fantom = ff_df.to_dict('records') 

        print(">>> Obtain responses...")
        for idx, input_prompt in enumerate(tqdm(target_data)):
            if idx <= last_idx:
                continue
            if self.args.method=="cot":
                cot_input_prompt = input_prompt.removesuffix("Answer:") + "\n\nLet's think step by step."
                self.flattened_fantom[idx]["cot_input_prompt"] = cot_input_prompt
                cot_response = self.parse_response(self.model.interact(cot_input_prompt, max_tokens=512))
                self.flattened_fantom[idx]["cot_response"] = cot_response
                input_prompt = cot_input_prompt + " " + cot_response + "\n\nTherefore, the answer is:"
            elif self.args.method=="s2a":
                print(f"<input_prompt>\n{input_prompt}\n")
                self.flattened_fantom[idx]['s2a_prompt'] = input_prompt
                s2a_response = self.parse_response(self.model.interact(input_prompt))
                self.flattened_fantom[idx]["s2a_response"] = s2a_response
                input_prompt = get_s2a_answering_prompt(s2a_response, self.flattened_fantom[idx]['complete_question'])
            elif self.args.method in ["perceptom", "ptob_inf"]: #ToM questions only
                if self.args.method=="perceptom":
                    if not os.path.exists(eval_res_df_path):
                        if idx>0 and self.flattened_fantom[idx-1]['final_input_prompt'] == input_prompt:
                            self.flattened_fantom[idx]['raw_perc_inf_response'] = self.flattened_fantom[idx-1]['raw_perc_inf_response']
                            perc_inf_response = self.flattened_fantom[idx-1]['perc_inf_response']
                        else:
                            raw_perc_inf_response = self.model.interact(input_prompt)
                            self.flattened_fantom[idx]['raw_perc_inf_response'] = raw_perc_inf_response
                            perc_inf_response = extract_json_array(raw_perc_inf_response)
                        self.flattened_fantom[idx]['perc_inf_prompt'] = input_prompt
                        self.flattened_fantom[idx]['perc_inf_response'] = perc_inf_response
                    print(f"<perception inference response>\n{self.flattened_fantom[idx]['perc_inf_response']}\n")
                input_prompt = self.get_final_input_json(input_prompt, idx) 
            elif self.args.method=="perc_inf":
                # Use the perception inference response for the same context in the previous question
                if idx>0 and self.flattened_fantom[idx-1]['final_input_prompt'] == input_prompt:
                    self.flattened_fantom[idx]['raw_perc_inf_response'] = self.flattened_fantom[idx-1]['raw_perc_inf_response']
                    response = self.flattened_fantom[idx-1]['perc_inf_response']
                else:
                    raw_perc_inf_response = self.model.interact(input_prompt)
                    self.flattened_fantom[idx]['raw_perc_inf_response'] = raw_perc_inf_response
                    response = extract_json_array(raw_perc_inf_response)
                self.flattened_fantom[idx]['perc_inf_response'] = response
                model_responses.append(response)
                print(f"<perception inference response>\n{response}\n")
            print(f"<final input prompt>\n{input_prompt}\n")
            self.flattened_fantom[idx]['final_input_prompt'] = input_prompt
            if self.args.method!="perc_inf":
                if input_prompt in ["invalid perception inference response"]:
                    response = "invalid perception inference response"
                else:
                    response = self.model.interact(input_prompt, max_tokens=256)
                    response = self.parse_response(response)
                model_responses.append(response)
                self.flattened_fantom[idx]['prediction'] = response
                print(f"<response>\n{response}\n")
            
            # Save the model responses in a file on the fly
            with open(response_filename_path, 'a') as f:
                json.dump({'index': idx, 'input_prompt': input_prompt, 'response': response}, f)
                f.write("\n")
            pd.DataFrame(self.flattened_fantom).to_csv(ff_path)
        return model_responses
    
    def run(self):
        os.makedirs(EVAL_DIR_PATH, exist_ok=True)
        model_responses = self.run_inference()
        if args.method!="perc_inf":
            evaluated_outputs = self.evaluate_response(self.flattened_fantom, model_responses)
            reports = self.run_reports(evaluated_outputs)
            self.dump_report_outputs(reports, evaluated_outputs)
            
    def get_responses_from_file(self, response_filename):
        setup = response_filename.removeprefix("model_responses").removesuffix(".jsonl")
        assert setup == self.output_filename_suffix.removesuffix(".json"), "The response file name does not match the output file name"
        response_file = os.path.join(EVAL_DIR_PATH, response_filename)
        df = pd.read_json(response_file, lines=True)
        model_responses = df['response'].to_list()
        return model_responses
    
def extract_q(x):
    if "Target:" in x:
        return x[x.index("Target: "):].strip()
    if "Information:" in x:
        return x[x.index("Information: "):].strip()
    return x[x.index("Question: "):].strip() if "Question: " in x else x

def str_to_list(s):
    l = s.split(",")
    return [c.strip(" []'") for c in l]

def put_index(df):
    ff_orig = pd.read_csv(os.path.join (DATA_DIR, "flattened_fantom.csv"))
    indices = []
    for idx, row in df.iterrows():
        cand1 = ff_orig[(ff_orig.set_id==row.set_id)&(ff_orig.question_type==row.question_type)]
        cand = cand1[cand1.question.map(lambda x: x.split("\n")[0] in row.question)]
        try:
            assert(len(cand)==1)
        except:
            print(f'Assertion Failure:\n{row.question}\n{cand1.question.values}\n{cand}\n')
            return df
        indices.append(cand.index.values[0])
    df['index'] = indices
    return df
from datetime import datetime

def get_timestamp():
    now = datetime.now()
    timestamp = now.strftime('%y%m%d%H%M')
    return timestamp

def main(args):

    # EVAL_DIR_PATH_SUFFIX = f"{args.model}-{args.method}"
    # if args.num_conv:
    #     EVAL_DIR_PATH_SUFFIX += f"-numConv_{args.num_conv}"
    # if len(args.run_name)>0:
    #     EVAL_DIR_PATH_SUFFIX += f"_{args.run_name}"
    # # EVAL_DIR_PATH_SUFFIX += f"-{get_timestamp()}"
    
    # EVAL_DIR_PATH = os.path.join('results', 'Percept-FANToM', EVAL_DIR_PATH_SUFFIX)
    
    # if args.method=="perceptom":
    #     PERC_INF_RESULT_DIR = f"results/Percept-FANToM/{EVAL_DIR_PATH_SUFFIX.replace('perceptom','perc_inf')}"
    #     print("perc_inf_result_dir: ", PERC_INF_RESULT_DIR)

    random.seed(RANDOM_SEED)

    evaluator = FantomEvalAgent(args)
    
    # Load perceiver file
    if args.method=="ptob_inf":
        gt_perc_df = pd.read_csv(os.path.join(DATA_DIR,"Percept-FANToM-conv_with_perceivers.csv"))
        ff_df = put_index(pd.DataFrame(evaluator.flattened_fantom)).set_index('index')

        gt_perception = []
        for idx, row in ff_df.iterrows():
            matching_row = gt_perc_df[gt_perc_df.part_id==row.part_id]
            assert(len(matching_row)==1)
            gt_perception.append(matching_row.conversation_with_perceivers.values[0])

        ff_df['gt_perception'] = gt_perception

        evaluator.flattened_fantom = ff_df.to_dict('records')
        evaluator.inputs = ff_df.gt_perception.values
    evaluator.run()
    
    # Save evaluation result in csv

    evaluated_responses_filename = "evaluated_responses" + evaluator.output_filename_suffix
    evaluated_responses_path = os.path.join(EVAL_DIR_PATH, evaluated_responses_filename)

    import json
    if args.method=="perc_inf":
        eval_res_df = put_index(pd.read_csv(os.path.join(EVAL_DIR_PATH, "flattened_fantom.csv")))
    else:
        f = open(evaluated_responses_path)
        data = json.load(f)
        eval_res_df = put_index(pd.DataFrame(data['results']))
        f.close()

    cols = ['index','set_id', 'part_id', 'context', 'missed_info_accessibility', 'tom_type', 'question_type', 
        'complete_question', 'question', 'correct_answer',
        'wrong_answer', 'choices_text', 'choices_list', 'fact_question',
        'fact_answer']
    if args.method=="ptob_inf":
        cols += ['gt_perception']
    elif args.method=="perc_inf": 
        cols += ['final_input_prompt', 'raw_perc_inf_response', 'perc_inf_response' ]
    elif args.method=="perceptom": 
        cols += ['perc_inf_prompt', 'raw_perc_inf_response', 'perc_inf_response', 'target_character1', 'target_character2']
    elif args.method=="cot":
        cols += ['cot_input_prompt', 'cot_response',]
    elif args.method=="s2a":
        cols += ['s2a_prompt', 's2a_response']
    if args.method!="perc_inf": 
        cols += ['final_input_prompt', 'prediction', 'binarized_model_answer', 'result', 
                'excluded_aware_character', 'included_unaware_character', 'prediction_answer_span']

    eval_res_df = eval_res_df[cols]
    eval_res_df.to_csv(evaluated_responses_path.replace(".json", ".csv"))
    
    for col in ['prediction', 'raw_perc_inf_response', 'cot_response', 's2a_response']:
        if col in eval_res_df.columns:
            print(f"# Failures in API call among '{col}': ", eval_res_df[col].map(lambda x:"no output" in x).sum())
            print(f"# Blank response among '{col}': ", eval_res_df[col].map(lambda x:len(str(x))==0).sum())
    if args.method!="perc_inf":
        num_invalid_json = (eval_res_df.prediction=="invalid perception inference response").sum()
        print(f"# Failures in json loading: {num_invalid_json}")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse arguments for the script.")
    parser.add_argument('--model', type=str, required=True, help="Specify the model to evaluate.", choices=[
        'gpt-4-1106-Preview', 
        'gpt-35-turbo-1106', 
        'gpt-4o',
        'claude-3-haiku-20240307',
        'claude-3-sonnet-20240229',
        'gemini-pro',
        'Llama-3-70b-chat-hf',
        'Mixtral-8x22B-Instruct-v0.1'
    ])
    parser.add_argument('--method', type=str, required=True, help="Specify the method to run.", choices=["vanilla", "cot", "s2a", "perc_inf", "perceptom", "ptob_inf"])
    parser.add_argument('--conversation_input_type', type=str, default='short', 
                        help="Type of conversation input, default: 'short').")
    parser.add_argument('--num_conv', type=int, default=None, 
                        help="Number of sample conversations to use (default: None).")
    parser.add_argument('--run_name', type=str, default="", 
                        help="Name of the run (default: '').")
    args = parser.parse_args()
    
    DATA_DIR = os.path.join("dataset", "Percept_FANToM")
    RANDOM_SEED = 99
    EVAL_DIR_PATH_SUFFIX = f"{args.model}-{args.method}"
    if args.num_conv:
        EVAL_DIR_PATH_SUFFIX += f"-numConv_{args.num_conv}"
    if len(args.run_name)>0:
        EVAL_DIR_PATH_SUFFIX += f"_{args.run_name}"
    # EVAL_DIR_PATH_SUFFIX += f"-{get_timestamp()}"

    EVAL_DIR_PATH = os.path.join('results', 'Percept-FANToM', EVAL_DIR_PATH_SUFFIX)

    if args.method=="perceptom":
        PERC_INF_RESULT_DIR = f"results/Percept-FANToM/{EVAL_DIR_PATH_SUFFIX.replace('perceptom','perc_inf')}"
        print("perc_inf_result_dir: ", PERC_INF_RESULT_DIR)
    
    main(args)