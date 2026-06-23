import pickle
import os
from collections import defaultdict
import statistics

task = 'pheno-24-notes'
filepath = f"/mnt/data/yihua/master/implementations/FuseMoE/src/run/TS_Text/{task}_TS_Text_TS_mTAND_64_Text_mTAND_64_layer1_moe_['laplace']_joint_[16, 5]_top_[4, 4]_batch_0.0004_8_8_128_1_2_512"
print('Experiment name: ' + os.path.basename(filepath))

with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

reformatted_result = defaultdict(lambda: defaultdict(list))

for seed, metrics in result.items():
    for metric_name, datasets in metrics.items():
        for dataset_name, value in datasets.items():
            reformatted_result[metric_name][dataset_name].append(value)

reformatted_result = {k: dict(v) for k, v in reformatted_result.items()}

metric_list = ['auc', 'ave_auc_macro', 'f1', 'macro_f1', 'precision', 'recall', 'macro_precision', 'macro_recall']

for metric_name, value_dict in reformatted_result.items():
    if metric_name in metric_list:
        for dataset_name, values in value_dict.items():
            print(f'{metric_name:>15} ({len(values)} run {dataset_name:4}): {statistics.fmean(values)*100:2.2f}±{statistics.stdev(values)*100:1.2f}')