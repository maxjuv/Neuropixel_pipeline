import time
import shutil
import subprocess as sp

from spikeinterface.core.waveform_extractor import extract_waveforms
# from compute_sorting_quality_metrics import compute_quality_metrics

from configuration import *

import matplotlib as mpl


from dataio import dataio


from spikeinterface.exporters import export_report



def export_sorting_report(run_key, sorter_name,  overwrite=True):
    #~ recording = dataio.get_subrecording(run_key, probe_index, cached=True)
    #~ sorting = dataio.get_sorting(run_key, probe_index, sorter_name=sorter_name)
    #~ recording_filtered = si.bandpass_filter(recording,  freq_min=300., freq_max=6000., margin_ms=5.0)


    waveform_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_waveforms'

    if not waveform_folder.is_dir():
        print(f'export_sorting_report need waveform first!!!! {run_key} {sorter_name}')
        return

    we = si.WaveformExtractor.load_from_folder(waveform_folder)
    print('waveform extracted')

    #~ sorting = we.sorting


    #~ amplitude_folder = Path(sortingdir) / f'{run_key} # probe{probe_index}' / f'{sorter_name}_spike_amplitudes'
    #~ amplitudes0 = np.load(amplitude_folder / 'spike_amplitudes_0.npy') # only one segment
    #~ amplitudes_by_dict = {}
    #~ for unit_id in

    #~ amplitudes =  [amplitudes_by_dict]
    # amplitudes = None

    # job_wargs = dict(n_jobs=20, chunk_size=30000, progress_bar=True)
    # amplitudes = si.get_spike_amplitudes(we,  peak_sign='neg', outputs='by_units', **job_wargs)


    #~ job_wargs = dict(n_jobs=8, chunk_size=30000, progress_bar=True)
    #~ waveform_extractor = si.extract_waveforms(recording_filtered, sorting, waveform_folder, **job_wargs)


    # DEBUG
    # waveform_extractor = si.extract_waveforms(recording_filtered, sorting, waveform_folder, load_if_exists=True)

    # metrics = compute_quality_metrics(run_key, probe_index, sorter_name)
    # print(metrics)


    report_folder = Path(workdir) / 'spike_sorting_report' / f'{run_key}#{sorter_name}'

    if report_folder.is_dir():
        if overwrite:
            shutil.rmtree(report_folder)
        else:
            print(f'already report folder {report_folder}')
            return
    print('ready for report')
    print(we)
    job_wargs = dict(n_jobs=20, chunk_size=30000, progress_bar=True)
    export_report(we, report_folder,
        # amplitudes=amplitudes,
        # metrics=metrics,
        **job_wargs)
    print('done')



def test_export_sorting_report():

    run_key = 'SD1548_6_S1'
    # run_key = 'SD1548_5_S1'
    probe_index = 0
    #~ probe_index = 1
    sorter_name = 'tridesclous'
    # sorter_name = 'kilosort2'

    export_sorting_report(run_key, probe_index, sorter_name)

def export_some_report():

    run_keys = get_run_keys_test()
    # run_keys = get_all_valid_run_keys()

    for run_key in run_keys:
        for probe_index in (0, 1):
            for sorter_name in ['tridesclous','kilosort2']:
                print(f'export_sorting_report {run_key} {probe_index}')

                report_folder = Path(workdir) / 'spike sorting report' / f' {run_key} {probe_index}'
                if report_folder.is_dir():
                    continue


                try:
                    export_sorting_report(run_key, probe_index,sorter_name)
                except:
                    print(f'ERREUR export_sorting_report {run_key} {probe_index}')



if __name__ == '__main__':


    # test_export_sorting_report()

    # export_some_report()

    # run_key = 'test'
    # sorter_name = 'kilosort3'
    # export_sorting_report(run_key, sorter_name)


    python = sp.getoutput('which python')
    print(python)
    run_key = 'test'
    sorter = 'kilosort2_5'
    # sorter = 'kilosort3'
    # sorter = 'kilosort2'
    # for sorter in ['kilosort2', 'kilosort2_5']:
    slurm_chara = '--partition=shared-cpu --mem=30G --time=3:00:00 --cpus-per-task=20'
    # python = '~/yggdrasil_python_envs/py395/bin/python'
    module = 'my_export_report'
    function = 'export_sorting_report'
    # function = 'compute_pca'
    # function = 'compute_spike_amplitudes'
    # function = 'compute_quality_metrics'
    # compute_pca(run_key, sorter)
    cmd = f"""srun {slurm_chara} {python} -c "import {module}; {module}.{function}('{run_key}', '{sorter}')" &"""
    print(cmd)
    os.system(cmd)
