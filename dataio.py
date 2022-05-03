

from configuration import *

import json
import dateutil

import spikeinterface.extractors as se
import spikeinterface.toolkit as tk
import probeinterface as pi




# def get_main_index():
#     ds = xr.open_dataset(main_index_filename)
#     main_index = ds.to_dataframe()
#     return main_index


#
# def get_annotations(nls_dirname):
#     nls_dirname = Path(nls_dirname)
#     with open(nls_dirname /  'annotations.json', 'r', encoding='utf8') as f:
#         annotations = json.load(f)
#     return annotations

#
# def open_nls(nls_dirname):
#     nls_dirname = Path(nls_dirname)
#
#     annotations = get_annotations(nls_dirname)
#
#     with open(nls_dirname / 'stream_properties.json', 'r', encoding='utf8') as f:
#         stream_properties = json.load(f)
#
#     input0 = stream_properties['input0']
#     sr = input0['sample_rate']
#     dtype = input0['dtype']
#     shape = list(input0['shape'])
#     shape[0] = -1
#     all_sigs = np.memmap(os.path.join(nls_dirname, 'input0.raw'), dtype=dtype, mode='r').reshape(shape)
#
#     return annotations, sr, all_sigs
#
#
# def get_rec_datetime(annotations):
#     if 'rec_datetime' in annotations:
#         return dateutil.parser.parse(annotations['rec_datetime'])
#
#
#
# def get_nls_signal(nls_dirname, by_channel_name=None,
#                     by_channel_index=None, copy=True):
#     annotations, sr, all_sigs = open_nls(nls_dirname)
#
#
#     if by_channel_name is not None:
#         raise(NotImplementedError)
#     elif by_channel_index is not None:
#         ind = by_channel_index
#     else:
#         raise(Exception('channel name or index!!!'))
#
#     sig = all_sigs[:, ind]
#     if copy:
#         sig = sig.copy()
#     return sig, sr
#


#~ channel_names = [p[0] for p in channel_params]
#~ channel_gains = [p[1] for p in channel_params]




class DataIO:
    def __init__(self):
        pass

    # def get_nls_dirname(self, run_key):
    #     nls_dirname =Path(workdir) / get_main_index().loc[run_key, 'nls_dirname']
    #     return nls_dirname
    #
    # def get_annotations(self, run_key):
    #     return get_annotations(self.get_nls_dirname(run_key))

    # def get_sampling_rate(self, run_key):
        # nls_dirname =self.get_nls_dirname(run_key)
        # annotations, sr, all_sigs = open_nls(nls_dirname)
        # return sr

    def get_sampling_rate(self, run_key):
        return 30000


    def get_recordingextractor(self, run_key = 'TOCHANGE'):
        # nls_dirname = self.get_nls_dirname(run_key)
        # nls_dirname = '/Volumes/m-GCarleton/GCarleton/JUVENTIN_Maxime/project/Neuropixel_pipeline/data'
        # with open(nls_dirname / 'stream_properties.json', 'r', encoding='utf8') as f:
        #     stream_properties = json.load(f)

        # input0 = stream_properties['input0']
        # sr = input0['sample_rate']
        # dtype = input0['dtype']
        # shape = list(input0['shape'])
        # shape[0] = -1
        # numchan = shape[1]
        nls_dirname = f'{datadir}/{run_key}'


        wired_channels = np.load(nls_dirname + '/channel_map.npy')
        positions = np.load(nls_dirname + '/channel_positions.npy')


        positions = np.zeros((385, 2))
        x = [43,11,59, 27]*int(385)
        print(len(x), positions.shape)
        x = x[:385]
        for i in range(int(385/2)):
            positions[2*i, 1] = 20*i +20
            positions[2*i+1, 1] = 20*i +20
        positions[:,0]=x
        channel_indices = np.arange(385)
        for i in range(385):
            if i not in wired_channels:
                channel_indices[i] = -1

        probe = pi.Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=positions, shapes='square', shape_params={'width': 12})
        probe.set_device_channel_indices(channel_indices)


        sr = self.get_sampling_rate(run_key)
        dtype = 'int16'
        numchan = 385  #374

        # rec = se.BinaryRecordingExtractor([Path(nls_dirname) / 'input0.raw'], sr, numchan, dtype)
        rec = se.BinaryRecordingExtractor([Path(nls_dirname) / 'Hopkins_20160722_g0_t0.imec.ap_CAR.bin'], sr, numchan, dtype)
        # rec = se.BinaryRecordingExtractor([Path(datadir) / 'rawDataSample.bin'], sr, numchan, dtype)
        # TODO gain, channel_ids

        rec = rec.set_probe(probe)

        return rec

    # def get_subrecording(self, run_key, probe_index, cached=False):
    #     if cached:
    #         name = f'{run_key} # probe{probe_index}'
    #         folder = Path(sortingdir) / name / 'subrecording'
    #         if not folder.is_dir():
    #             return
    #         sub_rec = si.load_extractor(folder)
    #     else:
    #         from run_spikesorting import make_subrecording
    #         sub_rec = make_subrecording(run_key, probe_index)
    #
    #     return sub_rec

    # def get_signals(self, run_key, copy=False):
    #     # nls_dirname = self.get_nls_dirname(run_key)
    #     # annotations, sr, all_sigs = open_nls(nls_dirname)
    #     t_start = 0.
    #     return all_sigs, sr, t_start



    def get_sorting(self, run_key, sorter_name=None, clean=True):
        if sorter_name is None:
            print('NO SORTER')
            # sorter_name =get_default_sorter_name()

        # fname = f'{run_key} # probe{probe_index} # {sorter_name}.npz'
        fname = f'{run_key} # {sorter_name}.npz'
        ####TODO FINISH
        if clean== True:
            file_name = Path(sortingresultsdir) / 'clean' / fname
        elif clean == False or clean =='auto':
            file_name = Path(sortingresultsdir) / 'raw' / fname
        if not file_name.is_file():
            return None
        sorting = se.NpzSortingExtractor(file_name)

        #######AVOID PROBLEM
        ids_to_keep=[]
        for unit_id in sorting.get_unit_ids():
            spikeframes = sorting.get_unit_spike_train(unit_id)
            if spikeframes.size >100:
                ids_to_keep.append(unit_id)
        sorting = sorting.select_units(ids_to_keep)

        if clean =='auto' :
            metric_names = ['snr', 'num_spikes', 'isi_violation']
            waveform_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_waveforms'
            if not waveform_folder.is_dir():
                print('compute waveforms first !')
                exit()
            we = si.WaveformExtractor.load_from_folder(waveform_folder)
            metrics = tk.compute_quality_metrics(we, metric_names=metric_names)
            # print(metrics.columns)
            # fig, ax = plt.subplots(nrows =4)
            # count,bin = np.histogram(metrics['snr'],bins = 30)
            # ax[0].plot(bin[:-1], count)
            # count,bin = np.histogram(metrics['num_spikes'],bins = 30)
            # ax[1].plot(bin[:-1], count)
            # count,bin = np.histogram(metrics['isi_violations_ratio'],bins = np.arange(0,0.1,.005))
            # ax[2].plot(bin[:-1], count)
            # count,bin = np.histogram(metrics['isi_violations_rate'],bins = np.arange(0,0.1,.005))
            # ax[3].plot(bin[:-1], count)
            # plt.show()
            mask = (metrics['snr']>5)&(metrics['isi_violations_rate']<0.05)&(metrics['isi_violations_ratio']<0.05)&(metrics['num_spikes']>500)
            # mask = (metrics['snr']>3)&(metrics['isi_violations_rate']<0.1)&(metrics['isi_violations_ratio']<0.1)&(metrics['num_spikes']>1000)
            # mask = (metrics['snr']>3)&(metrics['isi_violations_rate']<0.1)&(metrics['isi_violations_ratio']<0.1)&(metrics['num_spikes']>1000)

            print(100*np.sum(mask)/mask.size)
            tokeep = metrics.index.values[mask]
            before = len(sorting.get_unit_ids())
            sorting = sorting.select_units(tokeep)
            after = len(sorting.get_unit_ids())
            print(f'auto cleaning removed {before-after} units')

        return sorting


    def get_spikes(self, run_key, sorter_name=None, clean=True):
        sorting = self.get_sorting(run_key, sorter_name=sorter_name, clean=clean)
        if sorting is None:
            return
        print(run_key)
        spike_times = {}
        sr = self.get_sampling_rate(run_key)
        t_start = 0.
        for unit_id in sorting.get_unit_ids():
            spikeframes = sorting.get_unit_spike_train(unit_id)
            spike_times[unit_id] = spikeframes / sr + t_start

        return spike_times


    def get_neo_spiketrains(self, run_key, sorter_name=None, clean=True):

        recording = self.get_recordingextractor(run_key)
        times = recording.get_times()
        t_start =times[0] * pq.s
        t_stop =times[-1] * pq.s

        all_spike_times = self.get_spikes(run_key, sorter_name=sorter_name, clean=clean)

        spiketrains = []

        for unit_id, spike_times in all_spike_times.items():
            st = neo.SpikeTrain(spike_times, units='s', t_start=t_start, t_stop=t_stop)
            st.annotate(unit_id=unit_id)
            spiketrains.append(st)

        return spiketrains



# import warnings
# def get_default_sorter_name():
#     from params import default_sorter_name
#     sorter_name = default_sorter_name
#     warnings.warn(f'you are using {sorter_name}')
#     return sorter_name


dataio = DataIO()


# run_key_exception ={
#     }







if __name__ == '__main__':

    run_key = 'test'
    sorter_name = 'kilosort3'
    clean = 'auto'

    sorting = dataio.get_sorting(run_key, sorter_name, clean)

    # rec = dataio.get_recordingextractor(run_key)

    # trace = rec.get_traces(channel_ids=[50], end_frame = int(30000*500))
    # times = np.arange(trace.size)/30000
    # fig, ax = plt.subplots()
    # ax.plot(times, trace)
    # plt.show()
    #~ print(rec)
    #
    # sorting = dataio.get_sorting(run_key, 0)
    # print(sorting)
    # sorting = dataio.get_sorting(run_key, 0, sorter_name='tridesclous')
    # print(sorting)
    # sorting = dataio.get_sorting(run_key, 0, sorter_name='kilosort2')
    # print(sorting)

    #~ spike_times = dataio.get_spikes(run_key, 1)
    #~ print(spike_times)

    #~ all_sigs, sr, t_start = dataio.get_signals(run_key)
    #~ print(all_sigs.shape, sr, t_start)

    # probe_index = 0
    # spiketrains = dataio.get_neo_spiketrains(run_key, probe_index)




    #~ print(get_default_sorter_name())
