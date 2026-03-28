import pandas as pd
import json
import re
import random
import argparse
from typing import List
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_from_openai import build_prompt, build_domain_shots, build_nshot_prefix
import numpy as np
mapper = {
            "Software Architecture Creational Patterns" : "CP",
            "Software Architecture Structural Patterns" : "SPA",
            "Software Architecture Behavioral Patterns" : "BP",
            "Software Architecture Quality Attributes" : "QA",
            "Software Architecture Solutions" : "ASO",
            "Software Architecture Scalable Practices": "SPR",
            "Software Architecture Quantum Computing" : "QC",
            "Software Architecture Styles" : "AST",
        }

LABELS = ['A', 'B', 'C', 'D']
MODELS_BENCHMARK = {
    "claude_haiku":"Claude Haiku 4.5",
    "claude_sonnet":"Claude Sonnet 4.6",
    "claude_opus":"Claude Opus 4.6",
    "deepseek":"Deepseek v3.2",
    "gemini":"Gemini 3.1 Flash Lite",
    "gpt_5_4_nano":"GPT 5.4 Nano",
    "gpt_5_4_base":"GPT 5.4",
    "grok_4_1": "Grok 4.1",
    "grok_4_2": "Grok 4.2",
    "mistral":"Mistral 4 Small",
    "qwen3":"Qwen 3 235B A22B",
}

def extract_word_count(query : str):
    """
    Extract the number of words from the query
    """
    return len(query.split())

def extract_predictions(model_output : str):
    """
    Extracts the predicted answer from the model output.
    """
    match = re.match(r'^[A-Da-d]$', model_output)
    if match:
        return match.group(0).upper()

    match = re.match(r'^\s*([A-Da-d])\.?\s*$', model_output)
    if match:
        return match.group(1).upper()
    match = re.search(r'(?i)\b([A-D])\b', model_output)
    if match:
        return match.group(1).upper()
    
    return random.choice(LABELS)



def compute_model_accuracy(model_outputs_path : str, n_shots : int, dataset_path : str = "data/sake-dataset.json"):
    """
    Parses the model outputs from a JSON file and extracts the predicted answers.
    Returns a list of predicted answers.
    """
    random.seed(100)  
    correct = 0
    total =0
    correct_per_domain = {}
    total_per_domain = {}

    outcome = []
    with open(model_outputs_path, 'r') as f:
        data = json.load(f)
    
    nshots_pront = {}
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    domain_shots = build_domain_shots(dataset["validation"])

    for idx, row in enumerate(data):
        base_prompt = build_prompt(
            prompt_path="prompt_lib/base.txt",
            question=row["query"],
            choices=row["choices"],
            category=row["domain"],
        )
        nshot_prefix = build_nshot_prefix(
            nshot_prompt_path="prompt_lib/nshot.txt",
            domain_shots=domain_shots,
            n_shots=n_shots,
            domain=row["domain"],
        )
        prompt = base_prompt if nshot_prefix == "" else nshot_prefix + "\n\n" + base_prompt
        word_count = extract_word_count(row["query"])
        predicted_answer = extract_predictions(row["model_output"])
        golden = row["answer"].strip().upper()
        if predicted_answer == golden:
            correct += 1
            correct_per_domain[row["domain"]] = correct_per_domain.get(row["domain"], 0) + 1
        total += 1
        total_per_domain[row["domain"]] = total_per_domain.get(row["domain"], 0) + 1

        outcome.append(
            {
                "query_idx" : idx,
                "domain" : row["domain"],
                "world_count" : word_count,
                "is_correct" : predicted_answer == golden,
            }
        )
    
    if total != 2114:
        print(f"Warning: Total samples is {total}, expected 2114. Check the input file for missing or extra entries.")
        raise ValueError("Total samples mismatch.")
    accuracy_overall = 100*correct / total if total > 0 else 0
    accuracy_per_domain = {domain: 100*correct_per_domain.get(domain, 0) / total_per_domain.get(domain, 1) for domain in total_per_domain.keys()}

    word_counts = [entry["world_count"] for entry in outcome]
    q1, q2, q3 = np.percentile(word_counts, [25, 50, 75])
    e1, e2, e3 = int(np.ceil(q1)), int(np.ceil(q2)), int(np.ceil(q3))

    bin_labels = [
        f"wcount_0_{e1}",
        f"wcount_{e1 + 1}_{e2}",
        f"wcount_{e2 + 1}_{e3}",
        f"wcount_{e3 + 1}_inf",
    ]

    for entry in outcome:
        wc = entry["world_count"]
        if wc <= q1:
            entry["word_count_bin"] = bin_labels[0]
        elif wc <= q2:
            entry["word_count_bin"] = bin_labels[1]
        elif wc <= q3:
            entry["word_count_bin"] = bin_labels[2]
        else:
            entry["word_count_bin"] = bin_labels[3]

    accuracy_per_bin = {}
    for bin_label in bin_labels:
        bin_entries = [entry for entry in outcome if entry["word_count_bin"] == bin_label]
        if len(bin_entries) > 0:
            accuracy_per_bin[bin_label] = 100*sum(entry["is_correct"] for entry in bin_entries) / len(bin_entries)
        else:
            accuracy_per_bin[bin_label] = 0
    
    aso_entries = [entry for entry in outcome if entry["domain"] == "Software Architecture Solutions"]
    accuracy_per_bin_aso = {}
    for bin_label in bin_labels:
        bin_entries = [entry for entry in aso_entries if entry["word_count_bin"] == bin_label]
        if len(bin_entries) > 0:
            accuracy_per_bin_aso[bin_label] = 100*sum(entry["is_correct"] for entry in bin_entries) / len(bin_entries)
        else:
            accuracy_per_bin_aso[bin_label] = 0
    
    
    return accuracy_overall, accuracy_per_domain, accuracy_per_bin, accuracy_per_bin_aso


def parse_outputs_in_dirs(outputs_dir : List[str]):
    """
    Parses the directory with results.
    """
    import os
    results = []
    bin_column_order = []
    for _dir in outputs_dir:
        for filename in os.listdir(_dir):
            if filename.endswith('.json'):
                model_name = filename[:-5]  
                n_shots = re.search(r'_(\d+)shot', model_name)
                n_shots = int(n_shots.group(1)) if n_shots else 0
                matched_model = None
                for benchmark_key in MODELS_BENCHMARK.keys():
                    if benchmark_key in model_name:
                        matched_model = MODELS_BENCHMARK[benchmark_key]
                        break

                model_name = matched_model if matched_model else model_name
                model_name = re.sub(r'_\d+shot', '', model_name)
                accuracy_overall, accuracy_per_domain, accuracy_per_bin, accuracy_per_bin_aso = compute_model_accuracy(os.path.join(_dir, filename), n_shots)
                for bin_label in accuracy_per_bin.keys():
                    if bin_label not in bin_column_order:
                        bin_column_order.append(bin_label)
                result = {
                    "model_name": model_name,
                    "n_shots": n_shots,
                    "overall": accuracy_overall,
                }
                for bin_label in bin_column_order:
                    result[bin_label] = accuracy_per_bin.get(bin_label, 0)
                    result[f"{bin_label}_ASO"] = accuracy_per_bin_aso.get(bin_label, 0)
                for domain, acc in accuracy_per_domain.items():
                    result[f"{domain}"] = acc
                result["per_domain_avg"] = sum(accuracy_per_domain.values()) / len(accuracy_per_domain) if accuracy_per_domain else 0
                results.append(result)
        results_df = pd.DataFrame(results).round(2)
        model_names = list(MODELS_BENCHMARK.values())
        results_df = results_df.sort_values(by='model_name', key=lambda x: x.map({name: i for i, name in enumerate(model_names)}))

        results_df.rename(columns={'model_name': 'Model'}, inplace=True)
        results_df.rename(columns=mapper, inplace=True)
        results_df.to_csv(os.path.join(".", 'results_summary.csv'), index=False)

        table0 = results_df[results_df['n_shots'] == 0][['Model', 'overall']].copy()
        table0.rename(columns={'overall': '0-shot Overall'}, inplace=True)
        table0_5shot = results_df[results_df['n_shots'] == 5][['Model', 'overall']].copy()
        table0_5shot.rename(columns={'overall': '5-shot Overall'}, inplace=True)
        table0 = pd.merge(table0, table0_5shot, on='Model')
        table0.columns = ['Model', '0-shot Overall', '5-shot Overall']
        table0.to_csv(os.path.join(".", 'table2.csv'), index=False)

        table3_columns = ['Model'] + bin_column_order + [f"{bin}_ASO" for bin in bin_column_order]
        table3 = results_df[table3_columns].copy()
        table3 = table3[results_df['n_shots'] == 0].round(1)
        table3.to_csv(os.path.join(".", 'table5.csv'), index=False)
        
        model_names = list(MODELS_BENCHMARK.values())
        results_df_0shot = results_df[results_df['n_shots'] == 0].sort_values(by='Model', key=lambda x: x.map({name: i for i, name in enumerate(model_names)})).round(1)
        results_df_5shot = results_df[results_df['n_shots'] == 5].sort_values(by='Model', key=lambda x: x.map({name: i for i, name in enumerate(model_names)})).round(1)

        domain_columns = list(mapper.values())
        table2_0shot = results_df_0shot[['Model'] + domain_columns + ['per_domain_avg']].copy()
        table2_0shot.rename(columns={'per_domain_avg': 'Mean'}, inplace=True)
        average_row = {'Model': 'Average'}
        for col in domain_columns + ['Mean']:
            average_row[col] = table2_0shot[col].mean()
        table2_0shot = pd.concat([table2_0shot, pd.DataFrame([average_row]).round(1)], ignore_index=True)
        table2_0shot.to_csv(os.path.join(".", 'table3.csv'), index=False)

        table2_5shot : pd.DataFrame = results_df_5shot[['Model'] + domain_columns + ['per_domain_avg']].copy()
        table2_5shot.rename(columns={'per_domain_avg': 'Mean'}, inplace=True)
        average_row = {'Model': 'Average'}
        for col in domain_columns + ['Mean']:
            average_row[col] = table2_5shot[col].mean()
        table2_5shot = pd.concat([table2_5shot, pd.DataFrame([average_row]).round(1)], ignore_index=True)
        table2_5shot.to_csv(os.path.join(".", 'table4.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model outputs and compute accuracy.')
    parser.add_argument('--output_dirs', type=list, default=["outputs"], help='Directory containing model output JSON files.')
    args = parser.parse_args()
    parse_outputs_in_dirs(args.output_dirs)