# -*- coding: utf-8 -*-

import os
import sys
import stat
import subprocess

from pathlib import Path
import json
import time


import random
import string

import joblib
import xarray as xr


job_list = {}


def register_job(job):
    global job_list
    assert job.job_name not in job_list
    job_list[job.job_name] = job

def retrieve_job(job_name):
    return job_list[job_name]



def get_path(base_folder, job_name, params):

    hash = joblib.hash(params)
    save_path = Path(base_folder) / job_name / hash

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        with open(save_path / '__params__.json', mode='w') as f:
            json.dump(params, f, indent=4)

    return save_path



def _run_one_job_task(base_folder, job_name, params, func, keys, force_recompute):
    job = Job(base_folder, job_name, params, func)
    job.compute(keys, force_recompute=force_recompute)


_slurm_script = """#! {python}
import sys
sys.path.append("{module_folder}")
from jobtools import _run_one_job_task

from {module_name} import {job_instance_name} as job

job.compute({keys}, force_recompute={force_recompute})
"""


def compute_job_list(job, list_keys, force_recompute=True, engine='loop', **engine_kargs):

    if not force_recompute:
        cleaned_list_key = []
        #clean the list
        for keys in list_keys:
            output_filename = job.get_filename(*keys)
            if not os.path.exists(output_filename):
                cleaned_list_key.append(keys)
            else:
                print(job.job_name , 'already processed',keys)
        list_keys = cleaned_list_key

    t0 = time.perf_counter()
    if engine == 'loop':
        for keys in list_keys:
            job.compute(keys, force_recompute=force_recompute)
    elif engine == 'dask':
        #~ raise(NotImplementedError)
        #~
        client = engine_kargs['client']

        tasks = []
        for keys in list_keys:
            #~ print('submit', keys)
            task = client.submit(_run_one_job_task, job.base_folder, job.job_name, job.params, job.func, keys, force_recompute)
            tasks.append(task)

        for task in tasks:
            task.result()

    elif engine == 'joblib':
        n_jobs = engine_kargs['n_jobs']
        #~ joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(job.compute)(keys) for keys in list_keys)
        #~ print(job.base_folder, job.job_name, job.params, job.func, list_keys[0])
        joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_run_one_job_task)(job.base_folder, job.job_name, job.params, job.func, keys, force_recompute) for keys in list_keys)

    elif engine == 'slurm':
        # create python script and launch then with "srun"
        rand_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        slurm_script_folder = Path('.') / 'slurm_scripts'
        slurm_script_folder = slurm_script_folder.absolute()
        print(slurm_script_folder)
        slurm_script_folder.mkdir(exist_ok=True)

        for i, keys in enumerate(list_keys):
            key_txt = '_'.join([str(e) for e in keys])
            script_name = slurm_script_folder / f'{job.job_name}_{key_txt}_{rand_name}_{i}.py'
            output_name = slurm_script_folder / f'{job.job_name}_{key_txt}_{rand_name}_{i}_%j.out'
            # output_name = slurm_script_folder / f'{job.job_name}_{rand_name}_{i}.log'
            print()
            print('###', script_name.stem, '###')
            # print(output_name)

            module_name = engine_kargs['module_name']
            with open(script_name, 'w') as f:

                slurm_script = _slurm_script.format(
                    python=sys.executable,

                    module_folder=Path('.').absolute(),
                    module_name=module_name,
                    job_instance_name=job.job_name+'_job',
                    keys=keys,
                    force_recompute=force_recompute,
                )
                f.write(slurm_script)
                os.fchmod(f.fileno(), mode = stat.S_IRWXU)
            # _slurm_script

#             cpus_per_task = engine_kargs.get('cpus_per_task', 1)
#             partition = engine_kargs.get('partition', 'shared-cpu') #shared-gpu
#             gpus = engine_kargs.get('gpus', 0)   #turing:1
#             mem = engine_kargs.get('mem', '1G')
#             # matlab = engine_kargs.get('matlab', 1)
#             # mem = engine_kargs.get('mem', '1G')
#             list = []
#
# ## TODO condition on cluster/local and module load things
#
#             cmd = ['sbatch',
#                             f'--cpus-per-task={cpus_per_task}',
#                             f'--mem={mem}',
#                             # f'--licenses=matlab@matlablm.unige.ch',
#                             f'--job-name="{job.job_name}"',
#                             f'--partition={partition}',
#                             f'--gpus={gpus}',
#                             f'--time=6:00:00',
#                             f'--output="{output_name}"',
#                             str(script_name),
#                             ]
#
            slurm_params = engine_kargs['slurm_params']
            if not slurm_params:
                slurm_params = {'partition':'shared-cpu', 'cpus-per-task':'1', 'mem':'1G'}

            # cmd = ['sbatch',
            #                 f'--cpus-per-task={cpus_per_task}',
            #                 f'--mem={mem}',
            #                 f'--exclude={exclude}',
            #                 f'--output="{output_name}"',
            #                 str(script_name),
            #                 ]
            cmd = ['sbatch']
            cmd += [f'--{key}={value}' for key, value in slurm_params.items()]
            cmd += [f'--output="{output_name}"',
                    str(script_name),]

            cmd2 = ' '.join(cmd)
            print(cmd2)
            os.system(cmd2)


            # process = subprocess.Popen(cmd,
            #                     stdout=subprocess.PIPE,
            #                     stderr=subprocess.PIPE)
            # stdout, stderr = process.communicate()
            # print('stdout', stdout)
            # print('stderr', stderr)


    t1 = time.perf_counter()

    print(job.job_name, 'Total time {:.3f}'.format(t1-t0))




class Job:
    def __init__(self, base_folder, job_name, params, func):
        self.base_folder = base_folder
        self.job_name = job_name
        self.params = params
        self.save_path = get_path(base_folder, job_name, params)
        self.func = func

    def _make_keys(self, *args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                keys = (arg, )
            elif isinstance(arg, tuple):
                keys = arg
            elif isinstance(arg, list):
                keys = tuple(arg)
            else:
                raise(ValueError('need keys'))
        elif len(args) > 1:
            keys = args
        else:
            raise(ValueError('need keys'))
        return keys

    def get_filename(self, *args):
        keys = self._make_keys(*args)
        keys = tuple(str(k) for k in keys)
        filename = self.save_path / ('_'.join(keys) + '.nc')
        return filename

    def get(self, *args, compute=True):
        filename = self.get_filename(*args)
        if not os.path.exists(filename) and compute:
            ds = self.compute(*args)
            return ds
        ds = xr.open_dataset(filename)
        return ds

    def compute(self, *args, force_recompute=False):
        keys = self._make_keys(*args)
        output_filename = self.get_filename(*args)
        if not force_recompute and os.path.exists(output_filename):
            print(self.job_name , 'already processed',keys)
            return

        print(self.job_name, 'is processing' , keys)
        try:
            ds = self.func(*keys, **self.params)
        except:
            print('Erreur processing', self.job_name, keys)
            return None

        if ds is not None:
            try :
                ds.to_netcdf(output_filename)
            except PermissionError:
                # 2 job are computed in parralel
                pass
                print('erreur write')


        return ds
