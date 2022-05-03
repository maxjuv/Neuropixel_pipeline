from configuration import *

import subprocess as sp


#TODO simplify args (function_args, slurm_args)
def generate_kilosort_batch(module, function, run_key, sorter, gpu= 'turing', cpu = 1, mem=45000, time=f'3:00:00'):
    #### gpu is turing or volta
    filename = f'{slurmdir}/{run_key}/{sorter}/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    batch_file = f'{filename}/batch.sh'
    print(batch_file)
    error_path = f"{filename}slurm-%j.err"
    output_path = f"{filename}slurm-%j.out"
    jobname= sorter
    python = sp.getoutput('which python')

    # if not os.path.exists(filename):
    #     f = open(filename, 'x')
    # else :


    f = open(batch_file, "w")

#SBATCH --output ={output_path}\n\
#SBATCH --error={error_path}\n\

    f.write(
    f"""#!/bin/sh\n\
#SBATH --licenses matlab@matlablm.unige.ch\n\
#SBATCH --error={error_path}\n\
#SBATCH --output={output_path}\n\
#SBATCH --job-name={jobname}\n\
#SBATCH --partition shared-gpu\n\
#SBATCH --gpus={gpu}\n\
#SBATCH --cpus-per-task={cpu}\n\
#SBATCH --mem={mem}\n\
#SBATCH --time={time}\n\

\n\
\n\
\n\
module load CUDAcore/11.0.2\n\
\n\
srun {python} -c "import {module}; {module}.{function}('{run_key}', '{sorter}')"
""")
#
# module load MATLAB/2021a\n\
# \n\
# module load CUDAcore/11.0.2\n\   CUDA/10.1.243


# module load matlab/2019b\n\      ####SEGMENTATION ERROR
# \n\
# module load CUDA/10.0.130\n\
# srun matlab -nodisplay -nodesktop -r "run('{filename}/run_sorting.m');quit;"\n\

    f.close()

def generate_python_batch(module, function, run_key, sorter, cpu = 20, mem=45000, time=f'3:00:00'):
    #### gpu is turing or volta
    filename = f'{slurmdir}/{run_key}/{sorter}/'
    if not os.path.exists(filename):
        os.makedirs(filename)
    batch_file = f'{filename}/batch.sh'
    print(batch_file)
    error_path = f"{filename}slurm-%j.err"
    output_path = f"{filename}slurm-%j.out"
    jobname= sorter
    python = sp.getoutput('which python')

    # if not os.path.exists(filename):
    #     f = open(filename, 'x')
    # else :
    f = open(batch_file, "w")

#SBATCH --output ={output_path}\n\
#SBATCH --error={error_path}\n\

    f.write(
    f"""#!/bin/sh\n\
#SBATCH --error={error_path}\n\
#SBATCH --output={output_path}\n\
#SBATCH --job-name={jobname}\n\
#SBATCH --partition shared-cpu\n\
#SBATCH --cpus-per-task={cpu}\n\
#SBATCH --mem={mem}\n\
#SBATCH --time={time}\n\

\n\
\n\
srun {python} -c "import {module}; {module}.{function}('{run_key}', '{sorter}')"
""")
#
# module load MATLAB/2021a\n\
# \n\
# module load CUDAcore/11.0.2\n\   CUDA/10.1.243


# module load matlab/2019b\n\      ####SEGMENTATION ERROR
# \n\
# module load CUDA/10.0.130\n\
# srun matlab -nodisplay -nodesktop -r "run('{filename}/run_sorting.m');quit;"\n\

    f.close()


if __name__ == '__main__':
# run_sorting_one_sorting(run_key, sorter,)
    # for kilosort_version in ['kilosort3','kilosort2.5','kilosort2']:
    generate_batch('test','run_spikesorting', 'run_sorting_one_sorting','test', 'kilosort2_5')
