import time
import shutil

from configuration import *
import subprocess as sp

from dataio import dataio #, get_main_index, get_annotations

import spikeinterface.full as si
import spikeinterface
# from key_selection import *

job_wargs = dict(n_jobs=20, chunk_size=30000, progress_bar=True)

def extract_waveforms(run_key, sorter_name, ms_before=1., ms_after=2., overwrite=True, clean = False):
    waveform_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_waveforms'
    if waveform_folder.is_dir():
        if overwrite:
            shutil.rmtree(waveform_folder)
        else:
            return

    recording = dataio.get_recordingextractor(run_key)
    sorting = dataio.get_sorting(run_key, sorter_name=sorter_name, clean=clean)

    spike_times = dataio.get_spikes(run_key=run_key, sorter_name=sorter_name, clean=False)
    # ids_to_keep=[]
    # # print(spike_times)
    # for cluster in spike_times:
    #     st = spike_times[cluster]
    #     a = st.size
    #     if a >100:
    #         ids_to_keep.append(cluster)
    # sorting = sorting.select_units(ids_to_keep)
    #         print(a)
    # exit()

    recording_filtered = si.bandpass_filter(recording,  freq_min=300., freq_max=6000., margin_ms=5.0)


    waveform_extractor = si.extract_waveforms(recording_filtered, sorting, waveform_folder, ms_before=ms_before, ms_after=ms_after, **job_wargs)

def test_extract_waveforms():

    run_key = 'sample'

    #~ probe_index = 1
    # sorter_name = 'tridesclous'
    sorter_name = 'kilosort3'

    extract_waveforms(run_key, sorter_name)

def compute_pca(run_key, sorter_name):
    t0 = time.perf_counter()
    waveform_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_waveforms'
    we = si.WaveformExtractor.load_from_folder(waveform_folder)
    pca = si.compute_principal_components(we, load_if_exists=True,
                                 n_components=5, mode='by_channel_local')
    t1 = time.perf_counter()

    print('Total time {:.3f}'.format(t1-t0))
    return pca


def test_compute_pca():
    run_key = 'SD1548_5_S1'
    probe_index = 0
    #~ probe_index = 1
    sorter_name = 'tridesclous'
    # sorter_name = 'kilosort2'
    pca = compute_pca(run_key, probe_index, sorter_name)

def compute_spike_amplitudes(run_key, sorter_name):
    waveform_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_waveforms'
    we = si.WaveformExtractor.load_from_folder(waveform_folder)

    amplitude_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_spike_amplitudes'
    amplitude_folder.mkdir(exist_ok=True)


    amplitudes = spikeinterface.toolkit.compute_spike_amplitudes(we,  peak_sign='neg', outputs='concatenated', **job_wargs)

    for i, amps in enumerate(amplitudes):
        np.save(amplitude_folder / f'spike_amplitudes_{i}.npy' , amps)

    return amplitudes



def test_compute_spike_amplitudes():
    run_key = 'SD1548_5_S1'
    probe_index = 0
    #~ probe_index = 1
    # sorter_name = 'tridesclous'
    sorter_name = 'kilosort2'

    compute_spike_amplitudes(run_key, probe_index, sorter_name)

def compute_quality_metrics(run_key, sorter_name):

    metric_names = ['snr', 'num_spikes', 'isi_violation', 'firing_rate', 'presence_ratio', 'amplitude_cutoff']

    waveform_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_waveforms'
    if not waveform_folder.is_dir():
        extract_waveforms(run_key, sorter_name, overwrite=True)

    we = si.WaveformExtractor.load_from_folder(waveform_folder)

    metrics = spikeinterface.toolkit.compute_quality_metrics(we, metric_names=metric_names)
    return metrics



def test_compute_quality_metrics():
    run_key = 'SD1548_5_S1'
    run_key = 'SD1854_3_S1'

    probe_index = 0
    #~ probe_index = 1
    sorter_name = 'tridesclous'
    # sorter_name = 'kilosort2'

    metrics = compute_quality_metrics(run_key, probe_index, sorter_name)
    print(metrics)


def compute_all_quality():

    sorter_name = 'tridesclous'
    # sorter_name = 'kilosort'

    # run_keys = get_run_keys()
    run_keys = get_all_valid_run_keys()
    for run_key in run_keys:
        print(run_key)
        for probe_index in (0, 1):
            metrics = compute_quality_metrics(run_key, probe_index, sorter_name)


def compute_all_waveform():

    sorter_name = 'tridesclous'
    # sorter_name = 'kilosort'

    # run_keys = get_run_keys()
    run_keys = get_all_valid_run_keys()
    for run_key in run_keys:
        print(run_key)
        for probe_index in (0, 1):
            # metrics = compute_quality_metrics(run_key, probe_index, sorter_name)
            extract_waveforms(run_key, probe_index, sorter_name, overwrite=True)





if __name__ == '__main__':
    # test_extract_waveforms()
    # test_compute_pca()

    # test_compute_spike_amplitudes()
    # test_compute_quality_metrics()
    # compute_all_quality()
    # compute_all_waveform()


    # exit()

        #######CLUSTER COMPUTATION###############
    python = sp.getoutput('which python')
    print(python)
    run_key = 'test'
    # run_key = 'sample'
    # sorter = 'kilosort2_5'
    sorter = 'kilosort3'
    # sorter = 'kilosort2'
    # for sorter in ['kilosort2', 'kilosort2_5']:
    for sorter in ['kilosort2', 'kilosort2_5','kilosort3']:
        slurm_chara = '--partition=shared-cpu --mem=30G --time=3:00:00 --cpus-per-task=20'
        # slurm_chara = '--partition=shared-cpu --mem=5G --time=0:30:00 --cpus-per-task=20'

        # python = '~/yggdrasil_python_envs/py395/bin/python'
        module = 'compute_sorting_quality_metrics'
        function = 'extract_waveforms'
        # function = 'compute_spike_amplitudes'
        # function = 'compute_quality_metrics'
        # function = 'compute_pca'

        # compute_pca(run_key, sorter)
        cmd = f"""srun {slurm_chara} {python} -c "import {module}; {module}.{function}('{run_key}', '{sorter}')" &"""
        print(cmd)
        os.system(cmd)

    # cmd = f"""srun --partition=shared-cpu --mem=30G --time=3:00:00 --cpus-per-task=20 ~/yggdrasil_python_envs/py396/bin/python -c "import compute_sorting_quality_metrics; compute_sorting_quality_metrics.extract_waveforms('{run_key}', '{sorter}')" """
    # print(cmd)
    # os.system(cmd)
    # for function in ['compute_pca', 'compute_spike_amplitudes', 'compute_quality_metrics']:
    #     cmd = f"""srun --partition=shared-cpu --mem=10G --time=3:00:00 --cpus-per-task=20 ~/yggdrasil_python_envs/py396/bin/python -c "import compute_sorting_quality_metrics; compute_sorting_quality_metrics.{function}('{run_key}', '{sorter}')" &"""
    #     os.system(cmd)
