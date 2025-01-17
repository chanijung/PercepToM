import os
import argparse 
import json
import traceback
from collections import Counter
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from llm_agents import load_model


TOMI_QUESTION_TYPES = [
    'reality', 'memory',
    'first_order_0_tom', 'first_order_1_tom', 'first_order_0_no_tom', 'first_order_1_no_tom',
    'second_order_0_tom', 'second_order_1_tom', 'second_order_0_no_tom', 'second_order_1_no_tom'
]

def get_logs_dir(args):
    logs_directory = f"results/Percept-ToMi/{args.model.split('/')[-1]}"
    logs_directory += f"-{args.method}"
    if args.max_questions_per_qTypeRaw!=75:
        logs_directory += f"-numq_{args.max_questions_per_qTypeRaw}"
    if args.run_name:
        logs_directory += f"-{args.run_name}"
    return logs_directory

tomi_assump_instr = " Assume that characters in the story can perceive every scene occurring in their location but not scenes occurring elsewhere."
answering_helper_instr = "State the most detailed position possible. (e.g., in A in B) Answer in one sentence without explanation."

def get_cot_prompt(story, question):
    prompt = f"""Story: {story}.

Question: {question}"""
    prompt += " Assume that characters in the story can perceive every scene occurring in their location but not scenes occurring elsewhere."
    prompt += "\n\nLet's think step by step."
    return prompt, 200

def get_cot_answering_prompt(cot_response):
    prompt = f"{cot_response}\
\n\nBased on this reasoning, please answer the question in one sentence. Provide the most detailed response possible (e.g., A in B), but do not include any explanation.\n\nAnswer:"
    return prompt, 30

def get_s2a_prompt(story, question):
    prompt = f"""Given the following story, extract the part that is related and useful, so that using that text alone would be good context for providing an accurate and correct answer to the question. Only provide the extracted story without explanation. Do not answer the question.

Story: {story}.

Question: {question}{tomi_assump_instr}

Context related to the question (includes all content except unrelated sentences):"""
    return prompt, 300


def get_s2a_answering_prompt(extracted_story, question):
    prompt = f"""Story: {extracted_story}

Question: {question}{tomi_assump_instr} {answering_helper_instr}

Answer: """
    return prompt, 30


def get_simtom_prompt(story, char1):
    prompt = f"""Your job is to output only the events that the specified character, {char1}, knows about.
Here are a few rules:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place. 
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.
Story: {story}
What events does {char1} know about?
Only output the events according to the above rules, do not provide an explanation."""
    return prompt, 200


def get_simtom_answering_prompt(simtom_response, char1, question):
    prompt = f"""{simtom_response}
You are {char1}.
Based on the above information, answer the following question:
{question}
Keep your answer concise, one sentence enough. You must choose one of the above choices."""
    return prompt, 30


def get_perc_inf_prompt(story):
    prompt = f"""Story: {story}.

Create a JSON array consisting of JSON objects. Each object should contain a sentence from the story and the perceivers of the scene described in that sentence.\
{tomi_assump_instr} Also, include the actant of any action as a perceiver of that action.
Provide only a JSON array in the following format. Do not include any explanation.
[{{"Noah exited the living room.": ["Noah", "Emma"]}},]"""
    return prompt, 600

import re
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



def get_persp_ctx(perc_inf_response, char1, char2):
    perc_inf_response = json.loads(perc_inf_response.replace("\'", "\""))
    
    char_kg = []
    if (char2!=char2) or (char2 is None):  # first-order questions
        for p in perc_inf_response:
            if char1 in list(p.values())[0]:
                char_kg.append(list(p.keys())[0])
    else:  # second-order questions
        for p in perc_inf_response:
            obs = list(p.values())[0]
            if char1 in obs and char2 in obs:
                char_kg.append(list(p.keys())[0])
    return " ".join(char_kg)



def get_kgt_prompt(json_data, char1, char2):
    tracking = pprint.pformat(json_data, compact=True).replace("'",'"') if type(json_data)==list else json_data
    prompt = f"""Each JSON object in the following list contains the description of a consecutive scene in a story and its perceivers.

{tracking}"""
    if char2:
        prompt += f"""\n\nReconstruct the story by concatenaing the scene descriptions that {char1} thinks {char2} knows about"""
    elif char1:
        prompt += f"""\n\nReconstruct the story by concatenaing the scene descriptions that {char1} knows about"""
    else:
        print("no char1 and char2")
        assert(False)
    prompt += " (e.g., A entered X. B entered Y. P is in Q.). The scene descriptions should be the exact sentences in the story. Do not include any explanation."
    return prompt



def get_kgt_story(story, char1, char2):
    if char2 and str(char2)!="nan":
        prompt = f"""Here are the past scenes in sequence that {char1} thinks {char2} knows about."""
    elif char1:
        prompt = f"""Here are the past scenes in sequence that {char1} knows about."""
    else:
        print("no char1 and char2")
        assert(False)
    prompt += f"""\n\n{story.strip('."')}"""
    return prompt

import pprint
def get_ptob_inf_prompt(perc_inf, question):
    perc_inf_formatted = pprint.pformat(perc_inf, compact=True).replace("'",'"') if type(perc_inf)==list else perc_inf
    prompt = f"""Each JSON object in the following list contains the description of a consecutive scene in a story and its perceivers.

{perc_inf_formatted}

Question: {question} {answering_helper_instr}

Answer: """
    return prompt


def final_answer_prompt_formatting(model_name, reconstructed_story, rephrased_question):
    max_length = 30
    prompt = f"""Story: {reconstructed_story}.

Question: {rephrased_question}"""
    prompt += """ Assume that characters in the story can perceive every scene occurring in their location but not scenes occurring elsewhere. \
State the most detailed position possible (e.g., in A in B). Answer in one sentence without explanation.

Answer: """
        
    return prompt, max_length


def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for running the model")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "gpt-35-turbo-1106",
            "gpt-4o",
            "gpt-4-1106-Preview",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "gemini-pro",
            "Llama-3-70b-chat-hf",
            "Mixtral-8x22B-Instruct-v0.1"
        ],
        help="The model to use."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "perc_inf", 
            "ptob_inf", 
            "perceptom", 
            "vanilla", 
            "cot", 
            "s2a", 
            "simtom"
        ],
        help="The method to use."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        help="The run name specified for the method run (default: None)."
    )
    parser.add_argument(
        "--max_questions_per_qTypeRaw",
        type=int,
        default=75,
        choices=range(1, 76), 
        help="The maximum number of questions per qTypeRaw. Must be an integer between 1 and 75 (default: 75)."
    )
    
    args = parser.parse_args()
    args.logs_dir = get_logs_dir(args)
    args.perc_inf_result_path = os.path.join(args.logs_dir.replace('perceptom','perc_inf'), "result.csv")
    print(args.perc_inf_result_path)
    args.perc_inf_result_exists = os.path.exists(args.perc_inf_result_path)
    return args


if __name__=='__main__':
    args = parse_args()

    percept_tomi_df = pd.read_csv("dataset/Percept-ToMi.csv").set_index("index")
    model = load_model(model_name=args.model)

    os.makedirs(args.logs_dir, exist_ok=True)

    remaining_questions_by_type = {
        question_type: args.max_questions_per_qTypeRaw * (1 if 'tom' in question_type else 2)
        for question_type in TOMI_QUESTION_TYPES if question_type not in ["memory", "reality"]
    }

    rows = []

    preds = []
    kgt_responses = []

    corrects = []

    correct_per_question_type = Counter()
    total_per_question_type = Counter()

    if args.method=="perceptom":
        if args.perc_inf_result_exists:
            perc_inf_res_df = pd.read_csv(args.perc_inf_result_path, index_col="index")
            percept_tomi_df['prompt'] = perc_inf_res_df.prompt
            percept_tomi_df['raw_perc_inf_responses'] = perc_inf_res_df.raw_perc_inf_responses
            percept_tomi_df['perc_inf_responses'] = perc_inf_res_df.perc_inf_responses
            

    for i, row in percept_tomi_df.iterrows():
        if remaining_questions_by_type.get(row['qTypeRaw'], 0) == 0:
            continue
            
        print(f'{len(rows)}th STORY (index: {i})')
        print(f"<Question Type>\n{row['qType']}")
        logs_filename = f'example_{i}.json'
        
        if not args.method.startswith("gt_"):
            story = row['story'].strip('.')
        try:
            if args.method=="cot":
                prompt, max_length = get_cot_prompt(story, row['question'])
            elif args.method=="s2a":
                prompt, max_length = get_s2a_prompt(story, row['question'])
            elif args.method=="simtom":
                prompt, max_length = get_simtom_prompt(story, row.char1)
            elif args.method.startswith("perc_inf") or (args.method=="perceptom" and not args.perc_inf_result_exists): 
                question = row['question']
                prompt, max_length = get_perc_inf_prompt(story)
            elif (args.method=="ptob_inf") and (row['qTypeRaw'] not in ["memory", "reality"]):
                gt_perc_inf = json.loads(percept_tomi_df.loc[i].story_with_perceivers.replace("\'", "\""))
                prompt = get_ptob_inf_prompt(gt_perc_inf, row['question'])
                max_length = 600
            elif args.method=="perceptom" and args.perc_inf_result_exists:
                prompt = None
                generation = None
            elif args.method=="vanilla":
                question = row['question']
                prompt, max_length = final_answer_prompt_formatting(args.model, story, question)
            if prompt:
                print(f'\n<Prompt>\n{prompt}\n')
                row['prompt']=prompt
                
                # 3. Feed to Language Model
                generation = model.interact(prompt, max_tokens=max_length)
            
            prompt_2 = None
            if args.method=="cot":
                row['cot_response'] = generation
                prompt_2, max_length = get_cot_answering_prompt(prompt+" "+generation)
                pred = model.interact(prompt_2, max_tokens=max_length)
                print(f'<CoT Answering Prompt>\n{prompt_2}\n')
            elif args.method=="s2a":
                row['s2a_response'] = generation
                prompt_2, max_length = get_s2a_answering_prompt(generation, row['question'])
                print(f'<S2A Answering prompt>\n{prompt_2}\n')
                pred = model.interact(prompt_2, max_tokens=max_length)
            elif args.method=="simtom":
                row['simtom_response'] = generation
                prompt_2, max_length = get_simtom_answering_prompt(generation, row.char1, row.question)
                pred = model.interact(prompt_2, max_tokens=max_length)
                print(f'<SimToM Answering Prompt>\n{prompt_2}\n')
            elif (args.method.startswith("perc_inf")) or (args.method=="perceptom" and not args.perc_inf_result_exists):
                try:
                    perc_inf_response = json.loads(extract_json_array(generation))
                except Exception as e:
                    print(e)
                    print(generation)
                    perc_inf_response = generation
                row['raw_perc_inf_responses'] = generation
                row['perc_inf_responses'] = str(perc_inf_response)
                print(f'<Perception Inference>\n{perc_inf_response}\n')
            elif generation:
                pred = generation
                
            if args.method=="perceptom":
                persp_ctx = get_persp_ctx(row.perc_inf_responses, row.char1, row.char2)
                kgt_story = get_kgt_story(persp_ctx, row.char1, row.char2)
                prompt_2, max_length = final_answer_prompt_formatting(args.model, kgt_story, row['question'])
                prompt_2 = prompt_2.lstrip("Story: ")
                print(f'<PercepToM Answering Prompt>\n{prompt_2}\n')
                pred = model.interact(prompt_2, max_tokens=max_length)
            
            if prompt_2:
                row['prompt_2'] = prompt_2
            if args.method!="perc_inf":
                if pred=="no output":
                    print(f'Skipped datapoint #{i}')
                    continue
                preds.append(pred)

                correct = row['answer'] in pred
                correct_per_question_type[row['qTypeRaw']] += correct
                corrects.append(correct)

                print(f'<Prediction>\n{pred}\n')
                print(f'<Answer>\n{row["answer"]}\n')
                print(f'<Result>\n{correct}\n')
                print() 
            
            rows.append(row)
            
        
        except KeyboardInterrupt:
            import sys
            sys.exit()
        except Exception as e:
            print(f'Skipped datapoint #{i}: {e}')
            traceback.print_exc()
        remaining_questions_by_type[row['qTypeRaw']] -= 1
        total_per_question_type[row['qTypeRaw']] += 1

    import pandas as pd
    # Prediction results
    result = pd.concat([row.to_frame().transpose() for row in rows], axis=0)
    if args.method!="perc_inf":
        result['predictions'] = preds
        result['correctness'] = corrects
    result.reset_index(inplace=True)
    result.set_index("index",inplace=True)
    result.to_csv(os.path.join(args.logs_dir, "result.csv"))


    # ### Obtain ACC from result file
    if args.method!="perc_inf":
        acc_raw = result.groupby('qTypeRaw').correctness.mean().round(3)
        acc_raw_df = pd.DataFrame(acc_raw).transpose()
        acc_raw_df.loc['#Questions'] = result.groupby('qTypeRaw').correctness.count().astype(int)
        acc_raw_df.rename(index={'correctness':'Accuracy'}, inplace=True)
        acc_raw_df.to_csv(os.path.join(args.logs_dir,"acc_raw.csv"))
        
        ordered_qtypes = ['first_order_no_tom', 'first_order_tom', 'second_order_no_tom', 'second_order_tom']
        correct = result.groupby('qType').correctness
        acc = correct.mean().round(3)
        acc_df = pd.DataFrame(acc).transpose()
        acc_df.loc['#Questions'] = correct.count().astype(int)
        acc_df = acc_df[ordered_qtypes].rename(index={'correctness':'Accuracy'})
        acc_df.to_csv(os.path.join(args.logs_dir,"acc.csv"))
        print(acc_df)

