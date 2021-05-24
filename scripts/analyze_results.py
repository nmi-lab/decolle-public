import yaml
import os
import tabulate
import numpy as np
import argparse
import operator
import glob

parser = argparse.ArgumentParser(description='Script for analysing results of several experiments')
parser.add_argument('--experiments_dir', type=str, default='logs', help='Directory containing the results of the experiments')

args = parser.parse_args()

path = args.experiments_dir

# Reading all yml files in path recursively
all_params = {}
param_files = glob.glob(os.path.join(path, '**/*.yml'), recursive=True)
common_path = os.path.commonpath(param_files)
all_same_names = len(np.unique([os.path.basename(x) for x in param_files])) == 1
for d in param_files:
    with open(d, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if all_same_names:
        all_params[os.path.dirname(os.path.relpath(d, common_path))] = params
    else:
        all_params[os.path.relpath(d, common_path)] = params

# Adding all existing keys to all dictionaries
key_list = [list(x.keys()) for x in all_params.values()]
cum_keys = key_list[0]
for l in key_list:
   for k in l:
       if k not in cum_keys:
           cum_keys.append(k)

for x in all_params.values():
    for k in cum_keys:
        if k not in x.keys():
            x[k] = None

# Sort all dictionaries by keys
for x in all_params:
    all_params[x] = dict(sorted(all_params[x].items(), key=operator.itemgetter(0)))

# Filter columns with only one value
table = np.array([list(x.values()) for x in all_params.values()], dtype=object)
multi_val_cols = []
single_val_cols = []
for i in range(table.shape[1]):
    is_all_same = True
    for d in table[:, i]:
        is_all_same &= table[:, i][0] == d
    if not is_all_same:
        multi_val_cols.append(i)
    else:
        single_val_cols.append(i)

#Making one table for parameters common to all experiments
single_val_rows = [[x] for x in np.array(list(list(all_params.values())[0].values()), dtype=object)[single_val_cols]]
single_val_params = list(np.array(list(list(all_params.values())[0].keys()), dtype=object)[single_val_cols])
print(tabulate.tabulate(single_val_rows, ['Common Values'], showindex=single_val_params, tablefmt='pretty'))

#Making another table summarising the parameters which differ from one experiment to another
rows = [np.array(list(x.values()), dtype=object)[multi_val_cols] for x in all_params.values()]
headers = np.array(list(list(all_params.values())[0].keys()), dtype=object)[multi_val_cols]
print(tabulate.tabulate(rows, headers, showindex=all_params.keys(), tablefmt='pretty'))

