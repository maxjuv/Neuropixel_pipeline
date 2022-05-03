from configuration import *
from generate_scripts import *


def submit_one_kilosort_sorting(module, function, run_key, sorter, gpu= 'turing:1', mem=45000, time='3:00:00'):

    generate_kilosort_batch(module, function, run_key, sorter, gpu= gpu, mem=mem, time=time)
    # generate_matlab_script(filepath, kilosort_version)
    filepath = f'{slurmdir}/{run_key}/{sorter}/'

    os.system(f'cp {workdir}/Neuropixel_cluster_script/configuration.py {filepath}')
    os.system(f'cp {workdir}/Neuropixel_cluster_script/dataio.py {filepath}')
    os.system(f'cp {workdir}/Neuropixel_cluster_script/run_spikesorting.py {filepath}')

    os.system(f'sbatch {filepath}/batch.sh')


def submit_one_python_sorting(module, function, run_key, sorter, cpu = 20, mem=45000, time='10:00:00'):

    generate_python_batch(module, function, run_key, sorter, cpu=cpu, mem=mem, time=time)
    # generate_matlab_script(filepath, kilosort_version)
    filepath = f'{slurmdir}/{run_key}/{sorter}/'

    os.system(f'cp {workdir}/Neuropixel_cluster_script/configuration.py {filepath}')
    os.system(f'cp {workdir}/Neuropixel_cluster_script/dataio.py {filepath}')
    os.system(f'cp {workdir}/Neuropixel_cluster_script/run_spikesorting.py {filepath}')

    os.system(f'sbatch {filepath}/batch.sh')


if __name__ == '__main__':
    # filepath = '/home/users/j/juventin/Neuropixel_test/data/complete/'
    # kilosort_version = 'kilosort2'
    # for dataset in ['sample','complete']:
    #     filepath = f'/home/users/j/juventin/Neuropixel_test/data/{dataset}/'
    #     for kilosort_version in ['kilosort3','kilosort2.5','kilosort2']:
    #         run_one_sorting(filepath, kilosort_version)
    submit_one_kilosort_sorting('run_spikesorting', 'run_sorting_one_sorting','test', 'kilosort2_5')
    submit_one_kilosort_sorting('run_spikesorting', 'run_sorting_one_sorting','test', 'kilosort2')
    submit_one_kilosort_sorting('run_spikesorting', 'run_sorting_one_sorting','test', 'kilosort3')
    # submit_one_python_sorting('run_spikesorting', 'run_sorting_one_sorting','sample', 'tridesclous')
    # submit_one_python_sorting('run_spikesorting', 'run_sorting_one_sorting','test', 'spykingcircus')
