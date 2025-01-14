import string
import argparse
import pandas as pd
from pathlib import Path


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
        "--run_name",
        type=str,
        required=False,
        default=None,
        help="The run name specified for the method run (default: None)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="perc_inf",
        choices=["perc_inf", "perceptom"],
        help="The method from which the result will be evaluated (default: perc_inf)."
    )
    parser.add_argument(
        "--max_questions_per_qTypeRaw",
        type=int,
        default=75,
        choices=range(1, 76),  # Valid range is 1 to 75
        help="The maximum number of questions per qTypeRaw. Must be an integer between 1 and 75 (default: 75)."
    )
    
    args = parser.parse_args()
    return args

def remove_symbols(u):
    return u.translate(str.maketrans('', '', string.punctuation))

def get_logs_dir(args):
    logs_directory = f"results/Percept-ToMi/{args.model.split('/')[-1]}"
    logs_directory += f"-{args.method}"
    if args.max_questions_per_qTypeRaw!=75:
        logs_directory += f"-numq_{args.max_questions_per_qTypeRaw}"
    if args.run_name:
        logs_directory += f"-{args.run_name}"
    return logs_directory


if __name__ == '__main__':
    args = parse_args()

    model_name = args.model
    logs_dir = Path(get_logs_dir(args))
    result_df = pd.read_csv( logs_dir / f'result.csv')
    percept_tomi_df = pd.read_csv(f"dataset/Percept-ToMi.csv")

    cont_utter_2_perc = {}
    for _, row in percept_tomi_df.iterrows():
        idx = row['index']
        utter2per = {}
        for utter, perceivers in zip(eval(row['story']), eval(row['perceivers'])):
                utter_punc_removed = remove_symbols(utter)
                utter2per[utter_punc_removed] = {
                    'utter': utter,
                    'perceivers': [p for p in perceivers]
                }
        cont_utter_2_perc[idx] = utter2per

    utter_result = {
        'context_id': [],
        'utter': [],
        'pred': [],
        'gt': [],
    }
    
    ctx_result = {
        # 'index': [],
        'context_id': [],
        'context': [],
        'question_type': [],
        'question': [],
        # 'result': [],
        'perceiver_prediction': [],
        'perceiver_gt': [],
        'perceiver_accuracy': []
    }
    
    totals = []
    for i, row in result_df.iterrows():
        try:
            row['story_with_perceivers'] = row['story_with_perceivers'].replace('```json', '').replace('```', '')
            perc_inf_response = eval(row['story_with_perceivers'])
            utter2perc = cont_utter_2_perc[row['index']]
        except Exception as e:
            if row['qType'] in ['reality', 'memory']:
                continue
            else:
                print(f"[{row['index']}] parsing error - {e}\n")
                continue
        perceiver_gt, perceiver_prediction = [], []
        correct = 0
        total = 0
        for utter_pred in perc_inf_response:
            key_matched = False
            utter = list(utter_pred.keys())[0]
            if any(w in utter for w in ["loves", "likes", "hates", "dislikes", "exited", "entered"]):
                continue
            total += 1
            pred = utter_pred[utter]
            for utter_key in utter2perc.keys():
                if remove_symbols(utter) in utter_key:
                    utter_result['context_id'].append(row['index'])
                    utter_result['utter'].append(utter)
                    utter_result['gt'].append(utter2perc[utter_key]['perceivers'])
                    perceiver_gt.append(utter2perc[utter_key]['perceivers'])
                    try:
                        if set(pred) == set(utter2perc[utter_key]['perceivers']):
                            correct += 1
                        utter_result['pred'].append(pred)
                        perceiver_prediction.append(pred)
                    except Exception as e:
                        print(e,"\n")
    #                     total -= 1
                    key_matched = True
                    break
            if not key_matched:
                print(f"[{row['index']}] utter not in utter2perc\n- {utter}\n- {utter2perc.keys()}\n- {row.story}\n")
                utter_result['context_id'].append(row['index'])
                utter_result['utter'].append(utter)
                utter_result['pred'].append(pred)
                utter_result['gt'].append('')
                perceiver_prediction.append(pred)
                perceiver_gt.append('')
        if total == 0:
            print(f"[{row['index']}] total == 0\n")
            continue
        acc = correct / total
        totals.append(total)

        ctx_result['context_id'].append(row['index'])
        ctx_result['context'].append(row['story'])
        ctx_result['question_type'].append(row['qType'])
        ctx_result['question'].append(row['question'])
        # ctx_result['result'].append(row['correctness'])
        ctx_result['perceiver_gt'].append(perceiver_gt)
        ctx_result['perceiver_prediction'].append(perceiver_prediction)
        ctx_result['perceiver_accuracy'].append(acc)

    df_ctx = pd.DataFrame(ctx_result)
    df_ctx.to_csv( logs_dir / f'evaluated_context.csv', index=False)
    print(f"Context-level evaluation result saved in 'evaluated_context.csv'")

    df_utter = pd.DataFrame(utter_result)
    df_utter.to_csv( logs_dir / f'evaluated_utterance.csv', index=False)
    print(f"Scene-level evaluation result saved in 'evaluated_utterance.csv'")

    print(f"\n<Utterance-level Acc. (Perc. Inf. Acc. defined in the paper)>")
    accs = []
    num_valid = []
    num_total = []
    print(f"{'overall':>19} {df_ctx['perceiver_accuracy'].mean():.3f}")
    for qtype in ["first_order_no_tom", "first_order_tom", "second_order_no_tom", "second_order_tom"]:
        df_qtype = df_ctx[df_ctx['question_type'] == qtype]
        acc = df_qtype['perceiver_accuracy'].mean()
        accs.append(round(acc,3))
        num_valid.append(len(df_qtype))
        num_total.append((len(result_df[result_df['qType'] == qtype])))
        print(f"{qtype:>19} {acc:.3f}")

    print(f"\n<Context-level Acc.>")
    print(f"{'overall':>19} {(df_ctx['perceiver_accuracy'] == 1).mean():.3f} ({len(df_ctx)})")
    for qtype in ["first_order_no_tom", "first_order_tom", "second_order_no_tom", "second_order_tom"]:
        df_qtype = df_ctx[df_ctx['question_type'] == qtype]
        print(f"{qtype:>19} {(df_qtype['perceiver_accuracy'] == 1).mean():.3f} ({len(df_qtype)})")
    
    result_df["num_scenes"] = totals
    
    acc_qtype_df = pd.DataFrame([accs, num_valid,  num_total], index=["acc", "# valid samples", "# total samples"], columns=["first order true belief", "first order false belief", "second order true belief", "second order false belief"]).round(3)
    acc_qtype_df.to_csv(logs_dir / "acc_qType.csv")
    
    scenario_accs = []
    num_valids = []
    num_totals = []
    for scenario in ["true belief", "false belief"]:
        scenario_acc = 0
        num_valid = 0
        num_total = 0
        for order in ["first order", "second order"]:
            scenario_acc += acc_qtype_df.loc["acc",f"{order} {scenario}"]*acc_qtype_df.loc["# valid samples",f"{order} {scenario}"]
            num_valid += acc_qtype_df.loc["# valid samples",f"{order} {scenario}"]
            num_total += acc_qtype_df.loc["# total samples",f"{order} {scenario}"]
        scenario_acc /= num_valid
        scenario_accs.append(scenario_acc)
        num_valids.append(num_valid)
        num_totals.append(num_total)
    acc_scenario_df = pd.DataFrame([scenario_accs, num_valids, num_totals], index=["acc", "num_valids", "num_total"], columns=["true belief", "false belief"]).round(3)
    acc_scenario_df.to_csv(logs_dir / "acc_scenario.csv")
    print(f"\n<Utterance-level Acc. aggregating first and second order questions>\n{acc_scenario_df}")