import os
import json
import argparse

from helpers import utils
import eval

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default='results.json')
parser.add_argument('--redo', action='store_true')
args = parser.parse_args()

model_paths = []
# Get all directories in models

# Uncomment the line below to run baseline models
# model_paths += utils.get_base_models(include_prequant=False, relevant_models_only=True)
model_paths += utils.get_files('../cache/packed')

# testcases for perplexity calculation
datasets = ['wikitext2', 'c4_new', 'ptb_new_sliced']

# read previous results
if os.path.exists(args.output_file):
    with open(args.output_file) as f:
        all_results = json.load(f)
else:
    all_results = {}

new_results = {}  # results that are newly calculated, to be printed at the end

total_tests_to_run = {}  # tasks to be run will be stored here
skipped_models = []  # models that are skipped will be stored here

# Check which models/testcases need to be run
# This is done first so that we know how many tasks there are in total,
# and thus we can print the progress
for model_path in model_paths:
    model_name = os.path.basename(model_path)
    model_jobs = {'to_print': [], 'ppl': [], 'lm-eval': []}
    datasets_with_results = [dataset for dataset in datasets
                             if dataset in all_results.get(model_name, {}).get('ppl', {})]
    if not args.redo:
        model_jobs['ppl'] = [testcase for testcase in datasets if testcase not in datasets_with_results]
        if not model_jobs['ppl'] and not model_jobs['lm-eval']:
            # All results of the target model/testcases and model/tasks combination exist, skip
            skipped_models.append(model_name)
            continue
        else:
            if datasets_with_results:
                model_jobs['to_print'].append(f"Skipping datasets: "
                                              f"{datasets_with_results} because results already exist")
    else:
        if datasets_with_results:
            model_jobs['to_print'].append(f"Redoing all datasets, overwriting for {datasets_with_results}")
        else:
            model_jobs['to_print'].append("No previous ppl results to overwrite.")
        model_jobs['ppl'] = datasets
    model_jobs['to_print'].append(f"Running datasets: {model_jobs['ppl']}")
    total_tests_to_run[model_path] = model_jobs

total_ppl_job_count = sum(len(model_tasks['ppl']) for model_tasks in total_tests_to_run.values())
if skipped_models:
    print(f">> {len(skipped_models)} models will be skipped because all dataset results already exist.")
    # print('\n'.join(skipped_models) + '\n')
print(f">> Summary: {total_ppl_job_count} ppl jobs"
      f" over {len(total_tests_to_run)} models:")
print('\n'.join(os.path.basename(model_path) for model_path in total_tests_to_run) + '\n')

# Run all tasks
for i, model_path in enumerate(total_tests_to_run):
    model_name = os.path.basename(model_path)
    model_jobs = total_tests_to_run[model_path]
    to_print = model_jobs['to_print']
    datasets_to_evaluate = model_jobs['ppl']
    tasks_to_evaluate = model_jobs['lm-eval']
    print("==================================================")
    print(f" Model: {model_name}")
    print(f"Progress: {i + 1}/{len(total_tests_to_run)}")
    print("==================================================")
    datasets_with_results = [testcase for testcase in datasets if testcase in all_results.get(model_name, {})]

    for line in to_print:
        print('>> ' + line)

    ppl_results = {}
    lm_eval_results = {}

    # Run evaluation
    tokenizer_type, tokenizer, model = eval.auto_model_load(model_path)
    if datasets_to_evaluate:
        ppl_results = eval.evaluate_ppl(model, tokenizer, datasets_to_evaluate, verbose=True,
                                        chunk_size=2048, tokenizer_type=tokenizer_type)
    # Update new results
    new_results[model_name] = {}
    if ppl_results:
        new_results[model_name]['ppl'] = ppl_results

    # Update all results
    if ppl_results:
        all_results.setdefault(model_name, {}).setdefault('ppl', {}).update(ppl_results)

    # save results
    with open(args.output_file, 'w') as f:
        all_results = dict(sorted(all_results.items()))  # sort by key
        json.dump(all_results, f, indent=4)

    print()


print("---------------------- All Results ----------------------")
# print new results as formatted json
print(json.dumps(new_results, indent=4))

if len(total_tests_to_run) == 0:
    exit(1)
