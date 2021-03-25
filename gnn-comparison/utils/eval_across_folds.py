import json
from argparse import ArgumentParser
import glob
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dir", dest="results_dir", required=True)

    args = parser.parse_args()
    results_dir = args.results_dir
    num_folds = len(glob.glob(f'{results_dir}/GIN_NCI1_assessment/*/*/'))
    num_config = len(glob.glob(f'{results_dir}/GIN_NCI1_assessment/*/OUTER_FOLD_1/HOLDOUT_MS/*/'))
    
    results = {}
    config_values = {}
    for config_id in range(1, num_config + 1):
        config_result_jsons = glob.glob(f'{results_dir}/GIN_NCI1_assessment/*/*/HOLDOUT_MS/config_{config_id}/config_results.json')
        count = 0
        values = []
        for json_path in config_result_jsons:
            with open(json_path, 'r') as file:
                obj = json.load(file)
            count += 1
            values.append(obj['VL_score'])
        if count > 0:
            config_values[config_id] = obj['config']
            results[config_id] = (np.mean(values), np.std(values), count)
    
    sorted_configs = [(k, v) for k, v in sorted(results.items(), key=lambda item: item[1][0], reverse=True)]
    for config_id, results in sorted_configs:
        print(f'Config id: {config_id}: {results[0]} std: {results[1]} (count: {results[2]})')
        print(config_values[config_id])
        print()