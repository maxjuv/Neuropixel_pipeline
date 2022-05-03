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



def binarize_spike_times(spike_times, bin_size=20*pq.ms, smooth=True, kernel_half_with=2, zscore =True, AP_coding='rate'):

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
        print(np.unique(sig))
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

def lopez_2011_method(run_key, sorter_name, clean=False,bin_size=5, zscore = True, smooth = True):

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

    # plt.show()
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

    # plt.show()

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

    # plt.show()

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
