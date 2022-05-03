import shutil

from configuration import *
import jobtools
from dataio import dataio

from params import (detect_peaks_params, filter_params,
    job_kwargs, preprocess_and_peak_recording_params, localize_peaks_params,
    estimate_motion_params)


import spikeinterface.full as si
from spikeinterface.sortingcomponents import detect_peaks
from spikeinterface.sortingcomponents import localize_peaks
from spikeinterface.sortingcomponents import estimate_motion



def preprocess_and_peak_recording(run_key, **p):

    name = run_key
    folder = Path(sortingdir) / name
    folder.mkdir(parents=True, exist_ok=True)

    rec_raw = dataio.get_recordingextractor(run_key)

    rec_filtred = si.bandpass_filter(rec_raw, **filter_params)
    rec_preprocessed = si.common_reference(rec_filtred, reference='global', operator='median')
    #~ print(rec_filtred)

    preprocess_folder = folder / 'preprocessed'
    if preprocess_folder.is_dir():
        shutil.rmtree(preprocess_folder)
    rec_preprocessed = rec_preprocessed.save(folder=preprocess_folder, **job_kwargs)

    noise_levels = si.get_noise_levels(rec_preprocessed, return_scaled=False)
    np.save(folder / 'noise_levels.npy', noise_levels)

    peaks = detect_peaks(rec_preprocessed, noise_levels=noise_levels, **detect_peaks_params, **job_kwargs)
    np.save(folder / 'peaks.npy', peaks)

    peak_locations = localize_peaks(rec_preprocessed, peaks, **localize_peaks_params, **job_kwargs)
    np.save(folder / 'peak_locations.npy', peak_locations)

    # empty dataset
    ds = xr.Dataset()
    return ds

preprocess_and_peak_recording_job = jobtools.Job(precomputedir, 'preprocess_and_peak_recording', preprocess_and_peak_recording_params, preprocess_and_peak_recording)
jobtools.register_job(preprocess_and_peak_recording_job)



def test_preprocess_and_peak_recording():
    run_key = 'test'
    # run_key = 'SD1548_6_S1'
    # probe_index = 0
    # probe_index = 1
    ds = preprocess_and_peak_recording(run_key)
    print(ds)


def motion_estimation(run_key, **p):
    name = run_key
    folder = Path(sortingdir) / name

    rec_preprocessed = si.load_extractor(folder / 'preprocessed')
    peaks = np.load(folder / 'peaks.npy')
    peak_locations = np.load(folder / 'peak_locations.npy')


    motion, temporal_bins, spatial_bins, extra_check = estimate_motion(rec_preprocessed, peaks,
            peak_locations=peak_locations,
            output_extra_check=True,
            progress_bar=True,
            verbose=False,
            **estimate_motion_params
    )
    np.savez(folder / 'motion_estimated.npz', motion=motion, temporal_bins=temporal_bins, spatial_bins=spatial_bins)

motion_estimation_job = jobtools.Job(precomputedir, 'motion_estimation', estimate_motion_params, motion_estimation)
jobtools.register_job(motion_estimation_job)

def test_motion_estimation():
    run_key = 'test'
    # run_key = 'SD1548_6_S1'
    probe_index = 0
    # probe_index = 1
    ds = motion_estimation(run_key)
    print(ds)







def compute(run_key):


    run_keys = [tuple([run_key])]
    print(run_keys)

    #~ run_keys = ['SD1548_2_S1']
    # print(run_keys)
    # print('run_keys', len(run_keys))

    #~ jobtools.compute_job_list(detect_cycle_job, tasks, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(preprocess_and_peak_recording_job, tasks, force_recompute=False, engine='joblib', n_jobs=4)


    # slurm_params = { 'partition':'shared-cpu', \
    #                 'cpus-per-task':'20', 'mem':'40G', 'time':'3:00:00' }
    # jobtools.compute_job_list(preprocess_and_peak_recording_job, run_keys, force_recompute=True,
    #      engine='slurm', slurm_params=slurm_params, module_name='sorting_pipeline')

    slurm_params = {'partition':'shared-cpu', \
                    'cpus-per-task':'5', 'mem':'10G', 'time':'3:00:00' }
    # jobtools.compute_job_list(motion_estimation_job, tasks, force_recompute=True, engine='loop')
    jobtools.compute_job_list(motion_estimation_job, run_keys, force_recompute=True,
         engine='slurm', slurm_params=slurm_params, module_name='sorting_pipeline')




def compute_all():


    run_keys = get_run_keys()
    print(run_keys)

    #~ run_keys = ['SD1548_2_S1']
    # print(run_keys)
    # print('run_keys', len(run_keys))
    tasks = [(run_key, probe_index) for run_key in run_keys for probe_index in (0,1)]
    print(tasks )

    #~ jobtools.compute_job_list(detect_cycle_job, tasks, force_recompute=False, engine='loop')
    # jobtools.compute_job_list(preprocess_and_peak_recording_job, tasks, force_recompute=False, engine='joblib', n_jobs=4)

    jobtools.compute_job_list(preprocess_and_peak_recording_job, tasks, force_recompute=True,
         engine='slurm', cpus_per_task=20, mem='40G', module_name='sorting_pipeline')



    # jobtools.compute_job_list(motion_estimation_job, tasks, force_recompute=True, engine='loop')
    # jobtools.compute_job_list(motion_estimation_job, tasks, force_recompute=True,
    #      engine='slurm', cpus_per_task=5, mem='10G', module_name='sorting_pipeline')




if __name__ == '__main__':
    # test_preprocess_and_peak_recording()
    test_motion_estimation()

    # compute(run_key='test')
    # compute_all()
