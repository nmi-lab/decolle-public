import yaml
import os
import tabulate
import numpy as np
import argparse
import operator
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

parser = argparse.ArgumentParser(description='Script for analysing results of several experiments')
parser.add_argument('--experiments_dir', type=str, default='logs', help='Directory containing the results of the experiments')
parser.add_argument('--sort_by', type=str, default='max_test_accuracy', help='Parameter to sort by')
parser.add_argument('--save_to', type=str, help='If set saves to specified path. Type will be inferred from the extension. Available extensions (xlsx, csv)')

args = parser.parse_args()

path = args.experiments_dir

# Reading all yml files in path recursively
all_params = {}
param_files = glob.glob(os.path.join(path, '**/*.yml'), recursive=True)
if not param_files:
    print(f"Provided directory {path}, does not contain any params file")
    exit(0)

common_path = os.path.commonpath(param_files)
all_same_names = len(np.unique([os.path.basename(x) for x in param_files])) == 1
moving_avg_window = 10
for d in param_files:
    with open(d, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # If directory contains also tfevents add column with loss and accuracy to table
    containing_dir = os.path.dirname(d)
    for file in os.listdir(containing_dir):
        if file.startswith('events.out.tfevents'):
            ea = EventAccumulator(os.path.join(containing_dir, file))
            ea.Reload()
            test_losses = [[y.value for y in ea.Scalars(x)] for x in ea.Tags()['scalars'] if x.startswith('test_loss')]
            test_accuracies = [[y.value for y in ea.Scalars(x)] for x in ea.Tags()['scalars'] if x.startswith('test_acc')]
            
            # Taking the min loss and max accuracy after smoothing the data with moving average
            min_losses = [round(min(np.convolve(x, np.ones(moving_avg_window)/moving_avg_window, mode='valid')), 2) for x in test_losses]
            max_accuracies = [round(max(np.convolve(x, np.ones(moving_avg_window)/moving_avg_window, mode='valid')), 2) for x in test_accuracies]
            params['min_test_loss'] = min_losses
            params['max_test_accuracy'] = max_accuracies
            break
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
single_val_df = pd.DataFrame(single_val_rows, index=single_val_params, columns=['Common Values'])
print(tabulate.tabulate(single_val_df, headers=['Common Values'], tablefmt='pretty'))

#Making another table summarising the parameters which differ from one experiment to another
rows = np.array([np.array(list(x.values()), dtype=object)[multi_val_cols] for x in all_params.values()])
headers = np.array(list(list(all_params.values())[0].keys()), dtype=object)[multi_val_cols]
multi_val_df = pd.DataFrame(rows, columns=headers, index=all_params.keys())
if args.sort_by in multi_val_df.columns:
    multi_val_df.sort_values(args.sort_by, ascending=False)
else:
    multi_val_df.sort_index(inplace=True)

print(tabulate.tabulate(multi_val_df, headers='keys', tablefmt='pretty'))

if args.save_to is not None:
    extension = os.path.splitext(args.save_to)[-1]
    if extension == '.csv':
        multi_val_df.to_csv(args.save_to)
    elif extension == '.xlsx':
        multi_val_df.to_excel(args.save_to)
    else:
        print(f'Could not determine file type for extension {extension}')
