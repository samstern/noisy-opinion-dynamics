import argparse
from pathlib import Path
import subprocess
import yaml
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', action='store',
                        required=True,
                        help='the name of the stage to run. should match one of the stages in the config file')
    parser.add_argument('--out_dir', action='store',
                        required=True,
                        help='path to location where outputs are to be stored',
                        )
    parser.add_argument('--config_yaml', action='store',
                        required=True,
                        help='location of the config file'
                        )
    parser.add_argument('--num_graphs', action='store',type=int,
                        required=True,
                        help='number of graphs to generate'
                        )
    parser.add_argument('--num_nodes', action='store',type=int,
                        required=True,
                        help='number of nodes for each graph'
                        )
    parser.add_argument('--eta', type=float, default=0.0,
                        help='small value that is subtracted from the adjacency matrix to ensure that it '
                             'is substochastic')
    parser.add_argument('--num_samples', action='store',type=int,
                        required=True,
                        help='the number of simulations run over each graph')

    parser.add_argument('--num_timesteps', action='store',
                        required=True, type=int,
                        help='the number of timesteps of each simulation')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--save_full_simulation', dest='save_full', action='store_true',
                       help="save all the results of the simulations")
    group.add_argument('--save_final_state', dest='save_full', action='store_false',
                       help='save only the final state of each simulation')
    parser.set_defaults(save_full=True)


    return parser.parse_args()

def job_submission_formatter(file_name, script_name, params):
    string = f'#!/usr/bin/env bash \n' \
             f'\n' \
             f'#$ -l tmem=16G\n' \
             f'#$ -l h_vmem=16G\n' \
             f'#$ -l h_rt=7:0:0\n' \
             f'#$ -S /bin/bash\n' \
             f'#$ -j y\n' \
             f'#$ -wd /home/sstern/Opinion_Diversity\n' \
             f'\n' \
             f'source /share/apps/examples/source_files/python/python-3.7.2.source\n' \
             f'python3 -m op_div_simulation.{script_name} {params}'
    with open(file_name,'w') as f:
        f.write(string)

def main():

    args = parse_args()

    # create output directory if id doesn't already exist
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # import the config yaml file that describes what processes to run and in which order
    with open(args.config_yaml, 'r') as f:
        raw_yaml_str = ''.join(f.readlines())

    yaml_replacement_dict = dict(
        OUT_DIR=args.out_dir,
        NUM_GRAPHS=args.num_graphs,
        NUM_NODES=args.num_nodes,
        NUM_SAMPLES=args.num_samples,
        NUM_TIMESTEPS=args.num_timesteps,
        ETA=args.eta,
        SAVE_FULL=args.save_full,
    )

    yaml_config = yaml.full_load((raw_yaml_str.format(**yaml_replacement_dict)))

    running_processes = list()
    for i,item in enumerate(yaml_config[args.stage]):
        script_name = item['script_name']
        try:
            model_type = item['params']['model_type']
            noise_type = item['params']['noise_type']
            identifier_str = f'_{model_type}_{noise_type}'
        except:
            identifier_str = '_'
        params = ' '.join({f'--{k} "{v}"' for k, v in item['params'].items()})
        temp_file_name = f'temp_sub_file{identifier_str}_{i}.sh'
        job_submission_formatter(temp_file_name, script_name, params)
        str_to_run = f"qsub {temp_file_name}"
        # print(subprocess.getoutput(str_to_run))
        print(str_to_run)
        process = subprocess.Popen(str_to_run, shell=True)

        running_processes.append(process)
    for process in running_processes:
        process.wait()


if __name__ == '__main__':
    main()
