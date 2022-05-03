import time
import shutil

from spikeinterface.core.waveform_extractor import extract_waveforms

from configuration import *

import matplotlib as mpl


from dataio import dataio



from probeinterface.plotting import plot_probe
import probeinterface as pi
import spikeinterface.full as si


# import tridesclous as tdc
# import pyqtgraph as pg


# from tools_tridesclous import get_tridesclous_dataio

# from precompute_peaks import compute_peaks_job, dataset_to_peaks



# use_cache_recording = True
use_cache_recording = False


def plot_drift_before_sorting(run_key):

    rec = dataio.get_recordingextractor()
    # print(rec)

    folder = Path(sortingdir) / run_key
    if not (folder / 'peaks.npy').exists() or not (folder / 'peak_locations.npy').exists():
        print('not computed peak or peak_location', run_key)
        return

    peaks = np.load(folder / 'peaks.npy')
    peak_locations = np.load(folder / 'peak_locations.npy')

    gridspec_kw = dict(width_ratios=[4, 1])
    fig, axs = plt.subplots(ncols=2, figsize=(25, 10), sharey=True, gridspec_kw=gridspec_kw)

    ax = axs[0]
    x = peaks['sample_ind'] / rec.get_sampling_frequency()
    y = peak_locations['y']
    ax.scatter(x, y, s=1, color='k', alpha=0.005)

    ax = axs[1]
    probe = rec.get_probe()
    plot_probe(probe, ax=ax)
    ax.scatter(peak_locations['x'], peak_locations['y'], color='k', s=1, alpha=0.002)


    zile = np.load(folder / 'motion_estimated.npz')
    motion = zile['motion']
    temporal_bins = zile['temporal_bins']
    # spatial_bins = zile['spatial_bins']
    ax = axs[0]
    mid_probe = np.median(rec.get_channel_locations()[:, 1])
    ax.plot(temporal_bins, motion[:, 0] + mid_probe, color='r')



    # si.plot_displacement(motion, temporal_bins, spatial_bins, extra_check, with_histogram=False, ax=ax)


    # si.plot_drift_over_time(rec, peaks=peaks, mode='heatmap',
    #         bin_duration_s=10., weight_with_amplitudes=False, ax=ax)


    # imgs = [obj for obj in ax.get_children() if isinstance(obj, mpl.image.AxesImage)]
    # im = imgs[0]
    # im.set_clim(0, 1000)

    title = f'{run_key} {probe_index}'
    ax.set_title(title)

    fig_path = Path(figuredir) / 'drift_before_sorting'
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / (title + '.png'))






def plot_probe_activity_by_bin(run_key, probe_index):
    rec = dataio.get_subrecording(run_key, probe_index, cached=use_cache_recording)
    #~ print(rec)

    #~ rec_filtred = si.bandpass_filter(rec, freq_min=300., freq_max=6000., margin_ms=5.0)
    #~ print(rec_filtred)


    #~ peaks = detect_peaks(
        #~ rec_filtred, method='locally_exclusive',
        #~ peak_sign='neg', detect_threshold=6, n_shifts=7,
        #~ local_radius_um=100,
        #~ noise_levels=None,
        #~ random_chunk_kwargs={},
        #~ chunk_memory='10M', n_jobs=16, progress_bar=True)
    title = f'{run_key} {probe_index}'

    ds = compute_peaks_job.get(run_key, probe_index)
    peaks = dataset_to_peaks(ds)
    #~ print(peaks)

    fig, ax = plt.subplots()
    w = si.plot_peak_activity_map(rec, peaks=peaks, bin_duration_s=60.,
            with_contact_color=True,
            with_interpolated_map=False,
            figure=fig)

    fig_path = Path(figuredir) / 'probe_activity_by_bins'
    fig_path.mkdir(parents=True, exist_ok=True)
    w.animation.save(fig_path / (title + '.gif'))

    fig, ax = plt.subplots()
    w = si.plot_peak_activity_map(rec, peaks=peaks,
                    with_contact_color=True,
                    with_interpolated_map=False,
                    figure=fig)

    fig.suptitle(title)




def plot_unit_localisation(run_key, probe_index):
    rec = dataio.get_subrecording(run_key, probe_index, cached=use_cache_recording)
    print(rec)
    sorting = dataio.get_sorting(run_key, probe_index)
    print(sorting)

    rec_filtred = si.bandpass_filter(rec, freq_min=300., freq_max=6000., margin_ms=5.0)
    print(rec_filtred)

    folder = precomputedir + 'waveform_extractor/' + f'{run_key} # {probe_index}'

    # we = si.extract_waveforms(rec_filtred, sorting, folder, load_if_exists=True, n_jobs=16, total_memory='8G')
    we = si.extract_waveforms(rec_filtred, sorting, folder, load_if_exists=True,
                ms_before=1.5, ms_after=2., max_spikes_per_unit=500, dtype=None,
                chunk_memory='10M', n_jobs=8)

    #~ print(we)
    unit_ids = sorting.get_unit_ids()
    template0 = we.get_template(unit_ids[0])
    wf0 = we.get_waveforms(unit_ids[0])

    si.plot_unit_localization(we)
    #~ si.plot_unit_templates(we)


def plot_unit_map(run_key, probe_index):

    rec = dataio.get_subrecording(run_key, probe_index, cached=use_cache_recording)
    print(rec)
    sorting = dataio.get_sorting(run_key, probe_index)
    print(sorting)

    rec_filtred = si.bandpass_filter(rec, freq_min=300., freq_max=6000., margin_ms=5.0)
    print(rec_filtred)

    folder = precomputedir + 'waveform_extractor/' + f'{run_key} # {probe_index}'

    we = si.extract_waveforms(rec_filtred, sorting, folder, load_if_exists=True,
                ms_before=1.5, ms_after=2., max_spikes_per_unit=500, dtype=None,
                chunk_memory='10M', n_jobs=8)
    print(we)


    # unit_ids = sorting.unit_ids[:4]
    unit_ids = sorting.unit_ids

    si.plot_unit_map(we, animated=False, unit_ids=unit_ids)
    si.plot_unit_map(we, animated=True, unit_ids=unit_ids)


def plot_scatter_time_vs_depth_by_label(run_key, probe_index):
    tdc_dataio = get_tridesclous_dataio(run_key, probe_index)
    cc = tdc.CatalogueConstructor(dataio=tdc_dataio, chan_grp=0)

    y = cc.geometry[:, 1]
    x = cc.geometry[:, 0]
    peak_times = cc.all_peaks['index'] / tdc_dataio.sample_rate
    channels = cc.all_peaks['channel']

    peak_depth = y[channels]

    peak_depth += np.random.randn(peak_depth.size) * 2


    cc.refresh_colors(reset=False)

    fig, ax = plt.subplots()
    for label in cc.positive_cluster_labels:
        mask = cc.all_peaks['cluster_label'] == label
        color = cc.colors[label]

        ax.scatter(peak_times[mask], peak_depth[mask], s=2, alpha=0.1, color=color)

    title = f'{run_key} {probe_index}'
    ax.set_title(title)


def plot_time_vs_amplitude(run_key, probe_index):


    rec = dataio.get_subrecording(run_key, probe_index, cached=use_cache_recording)
    print(rec)
    sorting = dataio.get_sorting(run_key, probe_index)
    print(sorting)

    rec_filtred = si.bandpass_filter(rec, freq_min=300., freq_max=6000., margin_ms=5.0)
    print(rec_filtred)

    folder = precomputedir + 'waveform_extractor/' + f'{run_key} # {probe_index}'

    we = si.extract_waveforms(rec_filtred, sorting, folder, load_if_exists=True,
                ms_before=1.5, ms_after=2., max_spikes_per_unit=500, dtype=None,
                chunk_memory='10M', n_jobs=8)

    amplitudes = si.get_unit_amplitudes(we,  peak_sign='neg', outputs='by_units',
        chunk_memory='10M', n_jobs=16, progress_bar=True)

    fig, ax = plt.subplots()
    si.plot_amplitudes_timeseries(we, amplitudes=amplitudes, ax=ax)

    title = f'{run_key} {probe_index}'
    ax.set_title(title)

    fig_path = Path(figuredir) / 'time_vs_amplitude'
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / (title + '.png'))


def walk_check_drift():

    run_keys = get_run_keys()

    for run_key in run_keys:
        print(run_key)
        for probe_index in (0, 1):
            print(probe_index)
            rec = dataio.get_subrecording(run_key, probe_index, cached=use_cache_recording)
            # print(rec)
            if rec is None:
                continue

            plot_drift_before_sorting(run_key, probe_index)
            #~ plot_probe_activity_by_bin(run_key, probe_index)
            #~ plot_time_vs_amplitude(run_key, probe_index)
            #~ plot_unit_map(run_key, probe_index)
            # plot_scatter_time_vs_depth_by_label(run_key, probe_index)
            # plt.show()
        # plt.show()



if __name__ == '__main__':
    run_key = 'SD1548_2_S1'
    # run_key = 'SD939_7_S3'
    # run_key = 'SD940_1_S1'

    #~ run_key = 'SD1548_2_S1'
    # run_key = 'SD1548_3_S1'
    # run_key = 'SD1688_8_S1'
    probe_index = 0
    # probe_index = 1


    # with spikeinterface
    # plot_drift_before_sorting(run_key, probe_index)
    #~ plot_probe_activity_by_bin(run_key, probe_index)
    #~ plot_unit_localisation(run_key, probe_index)
    #~ plot_unit_map(run_key, probe_index)
    #~ plot_time_vs_amplitude(run_key, probe_index)

    # with TDC catalogue : TODO remove this
    # plot_scatter_time_vs_depth_by_label(run_key, probe_index)


    # plt.show()

    # walk_check_drift()
    plot_drift_before_sorting(run_key='test')
