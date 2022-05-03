# import os


from configuration import *

import datetime

from dataio import dataio
# from key_selection import *

#~ import spikeextractors as se
#~ import spikesorters as ss
#~ from spikecomparison.studytools import load_probe_file_inplace
# import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss


from probeinterface.plotting import plot_probe
import probeinterface as pi

import json
import shutil

from spikeinterface.sorters import Kilosort2Sorter, Kilosort2_5Sorter, Kilosort3Sorter, TridesclousSorter
import tridesclous as tdc

sorter_names = ['kilosort2', 'kilosort2_5', 'kilosort3', 'tridesclous']
# sorter_names = ['kilosort2_5']



if platform.system() == 'Linux' and getpass.getuser() == 'juventin':

    kilosort2_path = '/home/users/j/juventin/spike_sorters/Kilosort2'
    os.environ["KILOSORT2_PATH"] = kilosort2_path
    Kilosort2Sorter.set_kilosort2_path(kilosort2_path)

    kilosort2_5_path = '/home/users/j/juventin/spike_sorters/Kilosort2.5'
    os.environ["KILOSORT2_5_PATH"] = kilosort2_5_path
    Kilosort2_5Sorter.set_kilosort2_5_path(kilosort2_5_path)

    kilosort3_path = '/home/users/j/juventin/spike_sorters/Kilosort3'
    os.environ["KILOSORT3_PATH"] = kilosort3_path
    Kilosort3Sorter.set_kilosort3_path(kilosort3_path)

elif platform.system() == 'Windows' and getpass.getuser() == 'juventin' :

    kilosort2_path = 'N:/GCarleton/JUVENTIN_Maxime/code_library/spike_sorters/Kilosort2'
    os.environ["KILOSORT2_PATH"] = kilosort2_path
    Kilosort2Sorter.set_kilosort2_path(kilosort2_path)

    kilosort2_5_path = 'N:/GCarleton/JUVENTIN_Maxime/code_library/spike_sorters/Kilosort2_5'
    os.environ["KILOSORT2_5_PATH"] = kilosort2_5_path
    Kilosort2_5Sorter.set_kilosort2_5_path(kilosort2_5_path)

    kilosort3_path = 'N:/GCarleton/JUVENTIN_Maxime/code_library/spike_sorters/Kilosort3'
    os.environ["KILOSORT3_PATH"] = kilosort3_path
    Kilosort3Sorter.set_kilosort3_path(kilosort3_path)

def run_sorting_one_sorting(run_key, sorter):
    recording_dict = {}
    print(ss.installed_sorters())
    # for run_key, probe_index in sorting_list:
        # rec = dataio.get_subrecording(run_key, probe_index, cached=True)
    recording = dataio.get_recordingextractor(run_key)

    # if rec is None:
    #     continue

    # name = f'{run_key} # probe{probe_index}'
    name = run_key
    # recording_dict[name] = rec

    # pprint(recording_dict)


    working_folder = Path(f'{sortingdir}/{run_key}/{sorter}/')
    # sorter_list = ['tridesclous', ]
    # sorter_list = ['tridesclous','kilosort2' ]
    # sorter = 'kilosort2_5'
    # sorter_list = ['kilosort2', ]
    #~ sorter_list = ['spykingcircus']

    # TODO EXPORT
    sorter_params = {
        'tridesclous':{
            'freq_min': 300.,
            'freq_max': 6000.,
            'detect_threshold' : 6,
            'common_ref_removal': True,
            'nested_params' : {
                'duration' : 60000.,
                'peak_detector': {'adjacency_radius_um': 100},
                'clean_peaks': {'alien_value_threshold': 100.},
                'peak_sampler' : {'mode': 'rand_by_channel', 'nb_max_by_channel': 10000},
                },
        },
        'spykingcircus': {'num_workers' : 20},
        'kilosort2': {},
        'kilosort3': {},
        'kilosort2_5': {}

    }

    engine='loop'
    engine_kwargs={}

    #~ engine='joblib'
    #~ engine_kwargs={'n_jobs': max(4, len(sorter_list))}

    # engine='dask'
    # from dask.distributed import Client
    # from dask_jobqueue import SLURMCluster
    # python = '/home/samuel.garcia/.virtualenvs/py36/bin/python3.6'
    # cluster = SLURMCluster(cores=20, memory="64GB", python=python)
    # cluster.scale(10)
    # client = Client(cluster)
    # engine_kwargs={'client': client}

    # engine='slurm'
    # engine_kwargs={
    #     'cpus_per_task' : 1,
    #     'partition' : 'shared-gpu',
    #     'gpus' : 'turing:1',
    #     'mem' : '30G',
    #     'module_name':'run_spikesorting'
    #
    #     }



    # ss.run_sorters(sorter, recording_dict, working_folder,
    #                     mode_if_folder_exists='overwrite',
    #                     sorter_params=sorter_params,
    #                     verbose=True,
    #                     engine=engine, engine_kwargs=engine_kwargs)



    ss.run_sorter(sorter, recording, output_folder=working_folder,
                   remove_existing_folder=True, delete_output_folder=False,
                   verbose=False, raise_error=True,
                   docker_image=None, singularity_image=None,
                   with_output=True, **sorter_params[sorter])


    # jobtools.compute_job_list(preprocess_and_peak_recording_job, tasks, force_recompute=True,
    #      engine='slurm', cpus_per_task=20, mem='40G', module_name='sorting_pipeline')



def copy_raw_sortings():
    # move evrything in a sub folder
    now = datetime.datetime.now()
    now_txt = f'{now.year}_{now.month}_{now.day} {now.hour}h{now.minute}m{now.second}'
    backup_folder = Path(sortingresultsdir) / 'backups_raw' / now_txt
    backup_folder.mkdir(exist_ok=True, parents=True)
    for file in Path(sortingresultsdir).iterdir():
        if file.is_dir():
            continue
        if file.suffix == '.npz':
            print(file)
            print(file.name)
            shutil.move(file, backup_folder / file.name)

    sorting_folder = Path(sortingdir)
    for folder_name in sorting_folder.iterdir():
        name = folder_name.stem
        for sorter_name in sorter_names:
            output_folder =  folder_name / sorter_name

            if not output_folder.is_dir():
                print()
                print(output_folder)
                continue

            sorter_class = si.sorter_dict[sorter_name]
            print()
            print(name)
            print(folder_name)
            print(sorter_name)
            print(sorter_class)
            print(output_folder)
            try:
                sorting = sorter_class.get_result_from_folder(output_folder)
            except:
                print('Sorting error : ',folder_name)
                continue
            print(sorting)
            print(type(sorting))
            # sorting = sorting.remove_empty_units()
            tokeep = []
            for u,unit_id in enumerate(sorting.get_unit_ids()):
                spikeframes = sorting.get_unit_spike_train(unit_id)
                if spikeframes.size >5:
                    tokeep.append(unit_id)
            sorting = sorting.select_units(tokeep)

            # copy locally
            npz_filename = folder_name / f'{name} # {sorter_name}.npz'
            se.NpzSortingExtractor.write_sorting(sorting, npz_filename)
            # and in CRNLDATA
            npz_filename = Path(sortingresultsdir)  / 'raw' / f'{name} # {sorter_name}.npz'
            npz_filename.parent.mkdir(exist_ok=True, parents=True)
            se.NpzSortingExtractor.write_sorting(sorting, npz_filename)




if __name__ == '__main__':

    # run_sorting_one_sorting()
    copy_raw_sortings()
