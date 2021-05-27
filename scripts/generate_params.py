import yaml
import os
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Parameter file generator for grid search')
    parser.add_argument('--param_name', '-p', dest='param_name', type=str, required=True, nargs=1, action='append',
                        help='Name of parameter to change as it appears in default_params_file')
    parser.add_argument('--values_list', '-v', dest='values_list', required=True, nargs='+', action='append',
                        help='Set of values to test for the specified parameter. By default parsed as int. '
                             'If float add decimal (i.e -v 2.0). If list type comma separated entries in turn separated by space.'
                             'Single element list must have a comma at the end (i.e -v 1,2,3 4, 5,6')
    parser.add_argument('--default_params_file', type=str, default='params.yml',
                        help='File containing all of the default parameters and the specific parameter to test')
    parser.add_argument('--append', default=False,action='store_true',
                        help='Set this flag to append parameters file to those already present in params_to_test dir')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    param_list = [args.default_params_file]

    out_dir = 'params_to_test'
    if os.path.exists(out_dir) and not args.append:
        if input(f'Directory {out_dir} already exists. Are you sure you want to replace all of its content? (y/n)\n') == 'y':
            shutil.rmtree(out_dir)
        else:
            print('Exiting')
            exit(0)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.append and os.listdir(out_dir):
        i = int(os.path.splitext(sorted(os.listdir(out_dir))[-1])[0])
    else:
        i = 0

    first_file = i
    
    for param_name, value_list in zip(args.param_name, args.values_list):
        param_name = param_name[0]
        t = None
        subt = None
        for param_file in param_list:

            for val in value_list:

                with open(param_file, 'r') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)
                params_out = params.copy()

                # Parsing values
                p_split = val.split(',')
                if len(p_split) > 1:
                    if not p_split[-1]:
                        del p_split[-1]
                    try:
                        params_out[param_name] = [int(e) for e in p_split]
                    except ValueError:
                        try:
                            params_out[param_name] = [float(e) for e in p_split]
                        except ValueError:
                            params_out[param_name] = p_split
                else:
                    try:
                        params_out[param_name] = int(p_split[0])
                    except ValueError:
                        try:
                            params_out[param_name] = float(p_split[0])
                        except ValueError:
                            params_out[param_name] = p_split[0]
                if params_out == params and param_file != args.default_params_file:
                    continue

                i += 1
                with open(os.path.join(out_dir, '{:06d}.yml'.format(i)), 'w') as outfile:
                    yaml.dump(params_out, outfile)

                # Checking type consistency
                warn = False

                if isinstance(params_out[param_name], list):
                    if subt is not None:
                        if not isinstance(params_out[param_name][0], subt):
                            warn = True
                    subt = type(params_out[param_name][0])

                if t is not None:
                    if not isinstance(params_out[param_name], t):
                        warn = True
                t = type(params_out[param_name])

                if warn:
                    print(f'WARNING! Check type consistency for {param_name}')
        param_list = [os.path.join(out_dir, x) for x in sorted(os.listdir(out_dir))[first_file:]]


if __name__ == '__main__':
    main()

