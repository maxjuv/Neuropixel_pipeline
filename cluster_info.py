
from configuration import *


#####GO TO TOOLS.PY
def get_clusters(dir):
    clusters = np.load(f'{dir}/spike_clusters.npy')
    clusters = np.unique(clusters)
    return clusters

def get_spike_times_one_cluster(dir, cluster):
    cluster_mask = np.load(f'{dir}/spike_clusters.npy')
    cluster_mask = cluster_mask == cluster
    spike_times = np.load(f'{dir}/spike_times.npy')
    spike_times = spike_times[cluster_mask]

    return spike_times

# def get_all_spike_times(dir):
#     clusters = get_clusters(dir)
#     all_spike_times = []
#     for cluster in clusters:
#         spike_times = get_spike_times_one_cluster(dir, cluster)
#         all_spike_times.append = {cluster : spike_times}

def make_si_sorting(dir):
    clusters = get_clusters(dir)
    units_dict_list = []
    seg = {}
    for cluster in clusters[:30]:
        # print(cluster)
        spike_times = get_spike_times_one_cluster(dir, cluster)
        mask =np.where(np.diff(np.diff(spike_times)==0))
        spike_times = np.delete(spike_times, mask)
        print(np.sum(np.diff(spike_times)==0), np.sum(np.diff(spike_times)<0), np.sum(np.diff(spike_times)>0), spike_times.shape)
        # print(np.diff(np.array([0,1,32,33,34,34,39,100])))
        # print(np.diff(np.diff(np.array([0,1,32,33,34,34,39,100]))))
        # exit()

        # dict = {cluster : spike_times.astype('int16')}
        seg[cluster] = spike_times.astype('int16')
        # units_dict_list.append(dict)
    units_dict_list.append(seg)
    sorting = si.numpyextractors.NumpySorting.from_dict(units_dict_list, sr)
    return sorting

def compute_metrics(dir = raw_dir):
    # all_spike_times = get_all_spike_times(dir)
    sorting  = make_si_sorting(dir)

    fs = sorting.get_sampling_frequency()

    isi_threshold_ms=1.5
    min_isi_ms=0
    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000
    isi_threshold_samples = int(isi_threshold_s * fs)

    isi_violations_rate = {}
    isi_violations_count = {}
    isi_violations_ratio = {}

    unit_ids = sorting.get_unit_ids()
    num_segs = sorting.get_num_segments()
    print(num_segs)
    total_duration = 3600
    print(unit_ids)

    for unit_id in unit_ids:
        num_violations = 0
        num_spikes = 0
        for segment_index in range(num_segs):
            spike_train = sorting.get_unit_spike_train(unit_id=unit_id, segment_index=segment_index)
            isis = np.diff(spike_train)
            # print(np.sum(isis<0))
            print(spike_train.size)
            num_spikes += len(spike_train)
            num_violations += np.sum(isis < isi_threshold_samples)
        violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)
        total_rate = num_spikes / total_duration
        violation_rate = num_violations / violation_time

        isi_violations_ratio[unit_id] = violation_rate / total_rate
        isi_violations_rate[unit_id] = num_violations / total_duration
        isi_violations_count[unit_id] = num_violations

    # print(isi_violations_ratio)
    print(isi_violations_rate)
    # print(isi_violations_count)

    # clusters = get_clusters(dir)
    # spike_clusters = np.load(f'{dir}/spike_clusters.npy')
    #
    # spike_times = np.load(f'{dir}/spike_times.npy')
    # mask = spike_clusters == clusters[3]
    # print(spike_times[mask].size)

    return

def select_clusters():
    return

def putative_neuron_population():
    return

def stim_tagged_neurons():
    return









if __name__ == '__main__':
    compute_metrics()
