import yaml
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Parameter file generator for grid search')
    parser.add_argument('--param_name', '-p', dest='param_name', type=str, required=True,
                        help='Name of parameter to change as it appears in default_params_file')
    parser.add_argument('--values_list', '-v', dest='values_list', required=True, nargs='+',
                        help='Set of values to test for the specified parameter. By default parsed as int. '
                             'If float add decimal (i.e -v 2.0). If list type comma separated entries in turn separated by space.'
                             'Single element list must have a comma at the end (i.e -v 1,2,3 4, 5,6')
    parser.add_argument('--default_params_file', type=str, default='params.yml',
                        help='File containing all of the default parameters and the specific parameter to test')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open(args.default_params_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = 'params_to_test'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    t = None
    subt = None
    for i, p in enumerate(args.values_list):
        # Parsing values
        params_out = params.copy()
        p_split = p.split(',')
        if len(p_split) > 1:
            if not p_split[-1]:
                del p_split[-1]
            try:
                params_out[args.param_name] = [int(e) for e in p_split]
            except ValueError:
                params_out[args.param_name] = [float(e) for e in p_split]
        else:
            try:
                params_out[args.param_name] = int(p_split[0])
            except ValueError:
                params_out[args.param_name] = float(p_split[0])
        with open(os.path.join(out_dir, '{}.yml'.format(i)), 'w') as outfile:
            yaml.dump(params_out, outfile)

        # Checking type consistency
        warn = False

        if isinstance(params_out[args.param_name], list):
            if subt is not None:
                if not isinstance(params_out[args.param_name][0], subt):
                    warn = True
            subt = type(params_out[args.param_name][0])

        if t is not None:
            if not isinstance(params_out[args.param_name], t):
                warn = True
        t = type(params_out[args.param_name])

        if warn:
            print('WARNING! Check type consistency')


if __name__ == '__main__':
    main()

