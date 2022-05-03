from configuration import *
from dataio import dataio
import spikeinterface.toolkit as tk
import elephant
import scipy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from umap import UMAP


def plot_PSTH(run_key, sorter_name, clean =False):

    stim_positions = np.load(f'{workdir}/whitePositions.npy')
    stim_times = np.load(f'{workdir}/whiteTimes.npy')[0,:]

    stim_positions = np.load(f'{workdir}/blackPositions.npy')
    stim_times = np.load(f'{workdir}/blackTimes.npy')[0,:]
    print(stim_positions[:,0])
    print(stim_positions[:,1])
    # print(np.unique(stim_positions[:,1]))
######
    # sorting = dataio.get_sorting(run_key=run_key, sorter_name=sorter_name, clean=False)
    # spike_times = dataio.get_spikes(run_key=run_key, sorter_name=sorter_name, clean=False)
    # for cluster in spike_times:
    #     st = spike_times[cluster]
    #     a = st.size
    #     if a <2:
    #         print(a)
    # exit()

    sorting = dataio.get_sorting(run_key=run_key, sorter_name=sorter_name, clean=clean)

    # correlograms, correl_bins = tk.compute_correlograms(sorting,
    #                      window_ms=100.0, bin_ms=1.0,
    #                      symmetrize=False)

    t1 = -.05
    t2 = .1

    spike_times = dataio.get_spikes(run_key=run_key, sorter_name=sorter_name, clean=clean)

    for cluster in spike_times:
        st = spike_times[cluster]
        raster = []
        step = 0.002
        psth = np.zeros(np.arange(t1,t2, step).size-1)

        sub_sorting = sorting.select_units([cluster])
        auto_correlogram, correl_bins = tk.compute_correlograms(sub_sorting,
                             window_ms=200.0, bin_ms=1.0,
                             symmetrize=True)
        fig,ax = plt.subplots()
        ax.plot(correl_bins[:-1], auto_correlogram[0, 0])

        # for i in np.unique(stim_positions[:,0]):
        #     for j in np.unique(stim_positions[:,1]):
        #         mask = (stim_positions[:,0]==i) & (stim_positions[:,1]==j)
        #         # mask = (stim_positions[:,0]==i)
        for t in stim_times:
            # print(t)
            mask = (st>t+t1) & (st<t+t2)
            spikes = st[mask] -t
            raster.append(spikes)
            count, bin = np.histogram(spikes, np.arange(t1,t2, step))
            psth += count


        fig, ax = plt.subplots(nrows=2)
        ax[0].bar(bin[:-1], psth, width = step)
        ax[0].axhline(np.median(psth), color = 'r')
        ax[1].bar(bin[:-1], (psth-np.mean(psth))/np.std(psth), width = step)

            # fig, ax = plt.subplots()
        plt.show()


def plot_max_peak_waveform(run_key, sortername, clean=False):

    waveform_folder = Path(sortingdir) / f'{run_key}' / f'{sorter_name}_waveforms'

    if not waveform_folder.is_dir():
        print(f'export_sorting_report need waveform first!!!! {run_key} {sorter_name}')
        return
    we = si.WaveformExtractor.load_from_folder(waveform_folder)


    sorting = dataio.get_sorting(run_key, sorter_name=sorter_name, clean=clean)
    spike_times = dataio.get_spikes(run_key=run_key, sorter_name=sorter_name, clean=clean)
    print()
    N_clusters = len(list(spike_times.keys()))
    print(N_clusters)
    firing_rates = tk.compute_firing_rate(we)
    features = np.zeros((N_clusters,6))
    waveforms = np.zeros((N_clusters,50))
    for c, cluster in enumerate(spike_times):
        template = we.get_template(unit_id=cluster, mode='average')
        max_mask = np.max(template, axis = 0)-np.min(template, axis = 0)
        max_peak_ind = max_mask==np.max(max_mask)
        peak_waveform = template[:,max_peak_ind]
        # if np.sum(max_peak_ind)>1:
        peak_waveform = peak_waveform[:,0]
        peak_waveform = peak_waveform[10:60]
        peak_waveform = peak_waveform/abs(np.min(peak_waveform))
        # peak_waveform = peak_waveform[10:60]

        waveforms[c,:] = peak_waveform

        indice = np.arange(10,60)

        thresh = .2
        max1 = np.max(peak_waveform[(indice<18) & (indice>5)])
        if max1 >thresh:
            h1 = max1
        else :
            h1 =0
        max2 = np.max(peak_waveform[(indice>20) & (indice<50)])
        if max2 >thresh:
            h2 = max2
        else :
            h2 =0

        m1 = (indice<18) & (indice>5)
        d1 = np.sum(peak_waveform[m1])
        m2 = (indice>20) & (indice<35)
        d2 = np.sum(peak_waveform[m2])



        mini = np.min(peak_waveform)/4
        mask = peak_waveform<=mini
        di = indice[mask]
        dt = (di[-1]-di[0])*1000/30000 ### ms
        dv = np.sum(abs(peak_waveform<mini))
        features[c,0] = dt
        features[c,1] = dv
        features[c,2] = firing_rates[cluster]
        features[c,3] = (d1-d2)/(d1+d2)
        features[c,4] = h1
        features[c,5] = h2


        # fig, ax = plt.subplots()
        # ax.plot(peak_waveform)
        # ax.axvspan(indice[m1][0],indice[m1][-1],color = 'r', alpha = .2)
        # ax.axvspan(indice[m2][0],indice[m2][-1],color = 'r', alpha = .2)
        # ax.axhline(mini)
        # ax.scatter(pos_spike_inds,peak_waveform[pos_spike_inds], color = 'green')
        # plt.show()

        # exit()

    print(features.shape)
    features = (features-np.mean(features,axis =0))/np.std(features,axis=0)
    mcs = 50
    ms =5

    fig, ax = plt.subplots()
    fig.suptitle('TSNE then HDBSCAN')
    figW,axW = plt.subplots()
    figW.suptitle('UMAP then HDBSCAN-Waveforms')
    tsne = TSNE(n_components = 2).fit_transform(features)
    tsne_dbscan = HDBSCAN(min_cluster_size=mcs, min_samples = ms).fit(tsne)
    labels = tsne_dbscan.labels_
    for lab in np.unique(labels):
        mask = labels == lab
        inf, med, sup = np.percentile(waveforms[mask], q= [5,50,95], axis =0)
        axW.plot(indice/(30000/1000), med)
        axW.fill_between(indice/(30000/1000), inf, sup, alpha = .1)
        if lab == -1:
            ax.scatter(tsne[:,0][mask], tsne[:,1][mask], alpha = .3, marker = 'x')
        else :
            ax.scatter(tsne[:,0][mask], tsne[:,1][mask], alpha = .3)

    fig, ax = plt.subplots()
    fig.suptitle('UMAP then HDBSCAN')
    figW,axW = plt.subplots()
    figW.suptitle('TSNE then HDBSCAN-Waveforms')

    umap = UMAP(n_components = 2).fit_transform(features)
    umap_dbscan = HDBSCAN(min_cluster_size=mcs, min_samples = ms).fit(umap)
    labels = umap_dbscan.labels_
    for lab in np.unique(labels):
        mask = labels == lab
        inf, med, sup = np.percentile(waveforms[mask], q= [5,50,95], axis =0)
        axW.plot(indice/(30000/1000), med)
        axW.fill_between(indice/(30000/1000), inf, sup, alpha = .1)
        if lab == -1:
            ax.scatter(umap[:,0][mask], umap[:,1][mask], alpha = .3, marker = 'x')
        else :
            ax.scatter(umap[:,0][mask], umap[:,1][mask], alpha = .3)


    plt.show()

    exit()


    for cluster in spike_times:

        template = we.get_template(unit_id=cluster, mode='average')
        max_mask = np.max(template, axis = 0)-np.min(template, axis = 0)
        max_peak_ind = max_mask==np.max(max_mask)
        peak_waveform = template[:,max_peak_ind]
        if np.sum(max_peak_ind)>1:
            peak_waveform = peak_waveform[:,0]
            peak_waveform = peak_waveform.reshape(-1,1)
        peak_waveform = peak_waveform/abs(np.max(peak_waveform))
        # print(np.sum(max_mask==np.max(max_mask)))
        # print(template.shape)
        # print(max_mask.shape)
        # exit()
        # # ax2.plot(max_peak/abs(np.min(max_peak)), alpha = .2)
        # st = spike_times[cluster]
        # isi = np.diff(st)
        # bins = np.arange(-100,100,1)/1000
        # count, bin = np.histogram(isi, bins = bins)
        # fig, ax = plt.subplots()
        # width = np.mean(np.diff(bin))
        # ax.bar(bin[:-1],count, width = width, color = 'k', alpha =.8 )


        n_span = 4
        s = peak_waveform.size-n_span*2
        local_max = np.ones((s,), dtype='bool')
        sig_center = peak_waveform[n_span:-n_span]
        #~ print(sig_center.size)

        print(peak_waveform.shape)

        thresh = 0.2
        local_max = sig_center>thresh
        for k in range(n_span):
            local_max &= (sig_center > peak_waveform[k:k+sig_center.size])
            local_max &= (sig_center >= peak_waveform[k+n_span+1:k+n_span+1+sig_center.size])
        pos_spike_inds, _  = np.nonzero(local_max)
        pos_spike_inds += n_span

        # n_span = 8
        neg_peak_waveform = -1*peak_waveform
        s = neg_peak_waveform.size-n_span*2
        local_max = np.ones((s,), dtype='bool')
        sig_center = neg_peak_waveform[n_span:-n_span]
        #~ print(sig_center.size)
        # thresh = 0.2
        local_max = sig_center>thresh
        for k in range(n_span):
            local_max &= (sig_center > neg_peak_waveform[k:k+sig_center.size])
            local_max &= (sig_center >= neg_peak_waveform[k+n_span+1:k+n_span+1+sig_center.size])
        neg_spike_inds, _  = np.nonzero(local_max)
        neg_spike_inds += n_span


        sub_sorting = sorting.select_units([cluster])
        auto_correlogram, correl_bins = tk.compute_correlograms(sub_sorting,
                             window_ms=200.0, bin_ms=1.0,
                             symmetrize=True)


        width = np.mean(np.diff(correl_bins))
        fig, ax = plt.subplots(ncols =2)
        ax[0].plot(peak_waveform, color = 'k', alpha =.8)
        ax[0].axhline(0, color = 'r')
        ax[0].scatter(pos_spike_inds,peak_waveform[pos_spike_inds], color = 'green')
        ax[0].scatter(neg_spike_inds,peak_waveform[neg_spike_inds], color = 'orange')
        ax[1].bar(correl_bins[:-1], auto_correlogram[0,0], width = width, color = 'k', alpha =.8)

        plt.show()
        # exit()






def binarize_spike_times(spike_times, bin_size=20*pq.ms, smooth=True, kernel_half_with=5, zscore =True):

    sampling_rate = (1. / bin_size).rescale('Hz')


    sigs = []
    # count = 0
    for st in spike_times:
        # count+=1
        # if count//5:
        #     rng = int(10*np.random.random())
        #     print(rng)
        #     st = spike_times[0][rng::50]
        sig, times = elephant.conversion.binarize(st, sampling_rate=sampling_rate,
        t_start=None, t_stop=None, return_times=True)

        times = times.magnitude[:-1]
        sig = sig.astype('float32')
        # sig = scipy.signal.convolve(sig , kernel, mode= 'same', method='direct')              #####NO NORM

        if zscore :
            sig = (sig-np.mean(sig))/np.std(sig)
                                               #####NORM to reduce attraction from high firing rate cells

        if smooth:
            kernel = scipy.signal.windows.hann(kernel_half_with * 2 + 1)
            kernel = kernel / np.sum(kernel)
            sig = scipy.signal.convolve(sig , kernel, mode= 'same', method='direct')

        sigs.append(sig[:, None])
    sigs = np.concatenate(sigs, axis=1)

    return sigs, times

def detect_assemblies(run_key, sorter_name, clean=False,bin_size=5, zscore = True, smooth = True):

    spiketrains = dataio.get_neo_spiketrains(run_key=run_key, sorter_name=sorter_name, clean=clean)
    sigs, times = binarize_spike_times(spiketrains, bin_size=bin_size*pq.ms, smooth=smooth, zscore=zscore, kernel_half_with=1)

    N_neurons = sigs.shape[1]
    N_time_bins = sigs.shape[0]
    neurons = np.arange(N_neurons)

    correl_matrix = np.corrcoef(sigs.T)
    w, v = np.linalg.eig(correl_matrix)

    fig, ax = plt.subplots()
    inf, sup = np.percentile(correl_matrix, q=(5,95))
    im = ax.imshow(correl_matrix, aspect ='auto',interpolation='none', vmin = inf, vmax=sup )
    fig.colorbar(im, ax = ax)


    bins = 200
    count, bin = np.histogram(w, bins=bins)
    width = np.mean(np.diff(bin))

    q = sigs.shape[0]/sigs.shape[1]
    lambda_max = (1+(1/q)**.5)**2
    lambda_min = (1-(1/q)**.5)**2
    x = np.arange(lambda_min,lambda_max,0.0001)
    denom = 2*np.pi*x
    marchenko_pastur_eigenvalue_distribution =(q/denom)*np.sqrt((lambda_max-x)*(x-lambda_min))

    fig,ax =plt.subplots()
    ax.plot(x, marchenko_pastur_eigenvalue_distribution, color = 'k')
    ax.bar(bin[:-1], count, width = width)

    mask_assemblies = w>lambda_max
    mask_neurons = w<lambda_min
    N_signif_assemblies = np.sum(mask_assemblies)
    N_signif_neurons = np.sum(mask_neurons)
    print(f'{N_signif_assemblies} significant assemblies are detected')
    print(f'{N_signif_neurons} putative neurons involved in assemblies')
    #

    significant_PCS = v[:,mask_assemblies]        #### TRUE cf numpy.linalg.eig
    print(significant_PCS.shape)
    # significant_PCS = v[mask_assemblies,:].T        ###FALSE
    # print(significant_PCS.shape)
    ##################### CAUTION #########################
    ####    eigenvalues i goes with eigenvectors [:,i]
    #######################################################



    ##### LOPEZ
    neurons_vectors_in_assembly_spaces = significant_PCS
    print(neurons_vectors_in_assembly_spaces.shape)
    vector_lengths = np.linalg.norm(neurons_vectors_in_assembly_spaces, axis = 1)
    length_thresh = np.percentile(vector_lengths, q = 100*(1-N_signif_neurons/N_neurons))

    length_thresh = 0.455

    mask_larger_lentgh_neurons = vector_lengths>length_thresh
    cells_involved_in_assemblies = neurons[mask_larger_lentgh_neurons]
    # cells_involved_in_assemblies = neurons[vector_lengths>.3]
    order = np.argsort(vector_lengths)
    vector_lengths_to_plot = vector_lengths.copy()[order]


    # for i in range(N_signif_assemblies):
    #     fig, ax = plt.subplots()
    #     ax.scatter(np.arange(N_neurons), significant_PCS[:,i], color = 'green')
    #     for j in range(N_neurons):
    #         ax.plot((j,j), (0,significant_PCS[j,i]), color = 'k')
    fig, ax = plt.subplots(nrows=3)
    for i in range(vector_lengths.size):
        length = vector_lengths_to_plot[i]
        if length>length_thresh:
            ax[0].bar(i, length, color='green')
        else:
            ax[0].bar(i, length, color='k')
    ax[0].axhline(length_thresh, color = 'r')
    mini,maxi = vector_lengths_to_plot[0],vector_lengths_to_plot[-1]
    x = np.arange(0,N_neurons-1, 0.01)
    interp = scipy.interpolate.interp1d(np.arange(N_neurons),vector_lengths_to_plot, kind='cubic')
    y = interp(x)
    kernel_half_with = 4
    kernel = scipy.signal.windows.hann(kernel_half_with * 2 + 1)
    kernel = kernel / np.sum(kernel)
    y = scipy.signal.convolve(vector_lengths_to_plot , kernel, mode= 'same', method='direct')

    ax[0].plot(np.arange(y.size),y,color='k')
    ax[1].plot(np.diff(y))
    count,bin = np.histogram(vector_lengths_to_plot, bins = int(vector_lengths_to_plot.size/2))
    width = np.mean(np.diff(bin))
    ax[2].bar(bin[:-1], count, width=width, color ='k')

    plt.show()
    N_long_vector_neurons = np.sum(mask_larger_lentgh_neurons)
    print(N_long_vector_neurons, 'cells involved in assemblies are the following', cells_involved_in_assemblies)
    interaction_matrix = np.ones((cells_involved_in_assemblies.size, cells_involved_in_assemblies.size))
    for i,cell1 in enumerate(cells_involved_in_assemblies):
        for j, cell2 in enumerate(cells_involved_in_assemblies):
            a1 = neurons_vectors_in_assembly_spaces[cell1,:]
            a2 = neurons_vectors_in_assembly_spaces[cell2,:]
            # p = np.dot(ai,aj)/np.dot(aj,aj)
            p = np.inner(a1,a2)
            interaction_matrix[i,j] = p

    fig, ax = plt.subplots()
    inf, sup = np.percentile(interaction_matrix, q=(5,95))
    im = ax.imshow(interaction_matrix, aspect ='auto',interpolation='none', vmin = inf, vmax=sup )
    fig.colorbar(im, ax = ax)

    ######### Determine threshold for binary matrix ##############
    kmeans = KMeans(init="k-means++", n_clusters=2)
    flat_data = interaction_matrix.flatten().reshape(-1,1)
    kmeans.fit(flat_data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    count, bin = np.histogram(interaction_matrix, bins = int(interaction_matrix.size/2))
    h = 0.001
    x_min, x_max = bin.min() - .1, bin.max() + .1
    x = np.arange(x_min,x_max, h)
    prediction = kmeans.predict(x.reshape(-1,1))
    mask_binary_thresh = abs(np.diff(prediction))==1
    binary_thresh = x[:-1][mask_binary_thresh]

    fig,ax = plt.subplots()
    width = np.mean(np.diff(bin))
    ax.bar(bin[:-1], count, width=width, color ='k')
    colors = ['blue','orange']
    for label in labels :
        span = prediction == label
        frontiers = x[span]
        ax.axvspan(frontiers[0], frontiers[-1], color = colors[label], alpha = .3, zorder = 1)
    ax.scatter(centroids[0],x_max/2, marker='x', color = 'w', zorder = 3)
    ax.scatter(centroids[1],x_max/2, marker='x', color = 'w', zorder = 3)
    ax.bar(bin[:-1], count, width=width,color ='k', zorder = 2)

    plt.show()

    fig, ax = plt.subplots()
    binary = interaction_matrix>binary_thresh
    ax.imshow(binary, aspect ='auto',interpolation='none' )

    indice_to_remove = np.zeros(binary.shape[0])
    ####remove assembly of 1 cell
    for i in range(binary.shape[0]):
        line1 = binary[:,i]
        if np.sum(line1)<=1:
            indice_to_remove[i] = 1
    ####remove assembly of identical assemblies
    for i in range(binary.shape[0]):
        for j in np.arange(i,binary.shape[0]):
            line1 = binary[:,i]
            line2 = binary[:,j]
            if i != j and np.array_equal(line1,line2):
                indice_to_remove[j] = 1

    print(f'{np.sum(indice_to_remove)} removed neurons because identical')
    ordered_binary = binary.copy()
    ordered_binary = ordered_binary[indice_to_remove==0]
    conserved_indices = np.where(indice_to_remove==0)[0]
    # conserved_cells = cells_involved_in_assemblies[indice_to_remove==0]

    fig, ax = plt.subplots()
    ax.imshow(ordered_binary, aspect ='auto',interpolation='none' )

    order = np.argsort(np.sum(ordered_binary, axis= 1))
    ordered_binary = ordered_binary[order]
    conserved_indices = conserved_indices[order]
    fig, ax = plt.subplots()
    ax.imshow(ordered_binary, aspect ='auto',interpolation='none')

    plt.show()

    ###ordered_binary -->
    ### rows = assemblies
    ### columns = neurons
    print(conserved_indices)
    assemblies_activation = np.zeros((ordered_binary.shape[0], N_time_bins))
    for assembly in range(ordered_binary.shape[0]):
        indice = np.where(ordered_binary[assembly,:])[0]
        cells_in_the_assembly = cells_involved_in_assemblies[indice]
        print(cells_in_the_assembly)
        optimal_assembly_vector = np.sum(neurons_vectors_in_assembly_spaces[cells_in_the_assembly,:], axis =0 )
        optimal_assembly_vector /= np.linalg.norm(optimal_assembly_vector)

        optimal_assembly_vector_in_eigenvectors_space = np.zeros(N_long_vector_neurons)
        for i in range(N_signif_assemblies):
            optimal_assembly_vector_in_eigenvectors_space += optimal_assembly_vector[i]*neurons_vectors_in_assembly_spaces[mask_larger_lentgh_neurons,i]
            alpha = optimal_assembly_vector_in_eigenvectors_space

        fig, ax = plt.subplots()
        ax.scatter(np.arange(N_long_vector_neurons), optimal_assembly_vector_in_eigenvectors_space, color = 'green')
        for j in range(N_long_vector_neurons):
            ax.plot((j,j), (0,optimal_assembly_vector_in_eigenvectors_space[j]), color = 'k')
        ax.set_xticks(np.arange(cells_involved_in_assemblies.size))
        ax.set_xticklabels(cells_involved_in_assemblies)
            # plt.show()
        projector = np.outer(alpha, alpha)
        print(projector.shape)
        times = np.arange(N_time_bins)*bin_size
        coactivation = np.zeros(N_time_bins)
        # for b in np.arange(N_time_bins):
            # Z = sigs[b,:]
            # c = 0
        for i in range(cells_involved_in_assemblies.size):
            for j in range(cells_involved_in_assemblies.size):
                if i!=j:
                    # c += Z[cells_involved_in_assemblies[i]]*projector[i,j]*Z[cells_involved_in_assemblies[j]]
                    coactivation +=  sigs[:,cells_involved_in_assemblies[i]]*projector[i,j]*sigs[:,cells_involved_in_assemblies[j]]
        assemblies_activation[assembly] = coactivation
        fig, ax = plt.subplots(nrows=2, sharex = True)
        ax[1].plot(times, assemblies_activation[assembly,:])
        ax[0].imshow(sigs[:,cells_in_the_assembly].T, extent =[times[0], times[-1], 0, indice.size], aspect = 'auto')
        ax[0].set_yticks(np.arange(indice.size))
        ax[0].set_yticklabels(cells_involved_in_assemblies[indice])





if __name__ == '__main__':
    run_key = 'test'
    # run_key = 'sampleq'
    sorter_name = 'kilosort3'
    # sorter_name = 'kilosort2_5'
    # plot_PSTH(run_key, sorter_name, clean = 'auto')
    detect_assemblies(run_key, sorter_name, clean='auto',bin_size=10, zscore = False, smooth = True)
    # detect_assemblies(run_key, sorter_name, clean='auto',bin_size=1, smooth = True)
    # detect_assemblies(run_key, sorter_name, clean='auto', smooth = False)
    # plot_max_peak_waveform(run_key, sorter_name, clean = 'auto')
    plt.show()
