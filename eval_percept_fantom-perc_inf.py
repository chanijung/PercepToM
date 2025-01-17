import re
import string
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
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
        default="perc_inf",
        choices=["perc_inf", "perceptom"],
        help="The method from which the result will be evaluated (default: perc_inf)."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=False,
        default=None,
        help="The run name specified for the method run (default: None)."
    )
    parser.add_argument('--num_conv', type=int, default=None, 
                        help="Number of sample conversations to use (default: None).")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # model_dir = Path(args.working_dir) / f'{args.model}-{args.method}'
    EVAL_DIR_PATH_SUFFIX = f"{args.model}-{args.method}"
    if args.num_conv:
        EVAL_DIR_PATH_SUFFIX += f"-numConv_{args.num_conv}"
    if args.run_name:
        EVAL_DIR_PATH_SUFFIX += f"_{args.run_name}"
    import os
    model_dir = os.path.join('results', 'Percept-FANToM', EVAL_DIR_PATH_SUFFIX)

    result_path = model_dir / Path(f"evaluated_responses_short_input_{args.model}.csv")
    result_df = pd.read_csv(result_path)
    gt_perc_inf_df = pd.read_csv("dataset/Percept_FANToM/Percept-FANToM.csv")
    cont_utter_2_perc = defaultdict(dict)
    for _, row in gt_perc_inf_df.iterrows():
        pid = row['part_id']
        utter = row['utterance']
        utter_sym_removed = row['utterance_symbol_removed']
        utterer = utter.split(':')[0]
        perceivers = eval(row['perceivers'])
        if utterer in ['Everyone', 'All']:
            utterer = [p for p in perceivers]
        else:
            utterer = re.split(r' & |, ', utterer)
        cont_utter_2_perc[pid][utter_sym_removed] = {
            'utterer': utterer,
            'utter': utter,
            'perceivers': perceivers
        }

    utter_result = {
        'part_id': [],
        'context': [],
        'utter': [],
        'pred': [],
        'gt': [],
    }
    ctx_result = {
        'part_id': [],
        'set_id': [],
        'missed_info_accessibility': [],
        'context': [],
        'tom_order': [],
        'question_type': [],
        'question': [],
        'perceiver_prediction': [],
        'perceiver_gt': [],
        'perceiver_accuracy': []
    }
    for i, row in tqdm(result_df.iterrows()):
        part_id = row.part_id
        try:
            perc_inf_response = eval(row['perc_inf_response'])
        except Exception as e:
            print(i, 'Exception in parsing', e)
            continue
        utter2perc = cont_utter_2_perc[part_id]
        perceiver_gt, perceiver_prediction = [], []
        correct = 0
        total = 0
        for pred in perc_inf_response:
            for utter in pred:
                total += 1
                for utter_key in utter2perc.keys():
                    if ': '.join(utter.split(': ')[1:]).translate(str.maketrans('', '', string.punctuation)).replace('’', '').replace('…', '') in utter_key:
                        utter_result['part_id'].append(part_id)
                        utter_result['context'].append(row['context'])
                        utter_result['utter'].append(utter)
                        utter_result['gt'].append(utter2perc[utter_key]['perceivers'])
                        perceiver_gt.append(utter2perc[utter_key]['perceivers'])
                        utterer = utter2perc[utter_key]['utterer']
                        try:
                            if set(pred[utter]) - set(utterer) == set(utter2perc[utter_key]['perceivers']) - set(utterer):
                                correct += 1
                            utter_result['pred'].append(pred[utter])
                            perceiver_prediction.append(pred[utter])
                        except Exception as e:
                            print(e, "\n")
                            total -= 1
                        break
                else:
                    utter_result['part_id'].append(part_id)
                    utter_result['context'].append(row['context'])
                    utter_result['utter'].append(utter)
                    utter_result['pred'].append(pred[utter])
                    utter_result['gt'].append('')
                    perceiver_prediction.append(pred[utter])
                    perceiver_gt.append('')
        if total == 0:
            print(f"{part_id} total == 0\n")
            continue
        acc = correct / total
        ctx_result['part_id'].append(part_id)
        ctx_result['set_id'].append(row['set_id'])
        ctx_result['missed_info_accessibility'].append(row['missed_info_accessibility'])
        ctx_result['context'].append(row['context'])
        ctx_result['tom_order'].append(row['tom_type'])
        ctx_result['question_type'].append(row['question_type'])
        ctx_result['question'].append(row['question'])
        ctx_result['perceiver_gt'].append(perceiver_gt)
        ctx_result['perceiver_prediction'].append(perceiver_prediction)
        ctx_result['perceiver_accuracy'].append(acc)
    df_ctx = pd.DataFrame(ctx_result)
    df_ctx.to_csv(model_dir / Path('ctx_level_eval.csv'), index=False)
    print(f"Context-level evaluation result saved in 'ctx_level_eval.csv'")

    df_utter = pd.DataFrame(utter_result)
    df_utter.to_csv(model_dir/ Path('utter_level_eval.csv'), index=False)
    print(f"Utterance-level evaluation result saved in 'utter_level_eval.csv'\n")
    df_ctx_sets = df_ctx.drop_duplicates(subset=['set_id'], keep='last')
    result_df_sets = result_df.drop_duplicates(subset=['set_id'], keep='last')
    overall_acc_utter = round(df_ctx_sets['perceiver_accuracy'].mean(),3)
    overall_acc_context = round((df_ctx_sets['perceiver_accuracy'] == 1).mean(), 3)

    accs_utt = [overall_acc_utter]
    accs_context = [overall_acc_context]
    num_total_sets = [len(result_df_sets)]

    for accessibility in ["accessible", "inaccessible"]:
        df_ctx_accs = df_ctx[df_ctx['missed_info_accessibility'] == accessibility]
        df_cont_accs_sets = df_ctx_accs.drop_duplicates(subset=['set_id'], keep='last')
        result_df_accs = result_df[result_df['missed_info_accessibility'] == accessibility]
        result_df_accs_sets = result_df_accs.drop_duplicates(subset=['set_id'], keep='last')
        accs_acc_utter = round(df_cont_accs_sets['perceiver_accuracy'].mean(), 3)
        accs_acc_context = round((df_cont_accs_sets['perceiver_accuracy']==1).mean(), 3)
        accs_utt.append(accs_acc_utter)
        accs_context.append(accs_acc_context)
        num_total_sets.append(len(result_df_accs_sets))

    acc_df = pd.DataFrame([accs_utt, accs_context, num_total_sets], index=["Utterance-level Acc. (Perc. Inf. Acc.)", "Context-level Acc.", "# Total Sets"], columns=["Overall", "True Belief", "False Belief"])
    print(acc_df)

    acc_df.to_csv(model_dir / Path('perc_inf_acc.csv'))
    print(f"Accuracies saved in 'perc_inf_acc.csv'")