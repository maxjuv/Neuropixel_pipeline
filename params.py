channel_index_respi = 67


sorter_names = ['tridesclous', 'kilosort2', 'spykingcircus']

default_sorter_name = 'kilosort2'



job_kwargs = dict(
    n_jobs=20,
    # chunk_duration='1s',
    chunk_size=30000,
    progress_bar=True,
)

filter_params = dict(
    freq_min=300.,
    freq_max=6000.,
    margin_ms=5.0,
)

detect_peaks_params = dict(
    method='locally_exclusive',
    peak_sign='neg',
    detect_threshold=5,
    n_shifts=10,
    local_radius_um=100,
)

localize_peaks_params = dict(
    ms_before=0.3,
    ms_after=0.6,
    method='monopolar_triangulation',
    method_kwargs={'local_radius_um': 100., 'max_distance_um': 1000.},
)

estimate_motion_params = dict(
    direction='y',
    bin_duration_s=5.,
    bin_um=5.,
    method='decentralized_registration',
    method_kwargs = dict(
        pairwise_displacement_method='conv2d',
        convergence_method='gradient_descent',
    ),

    # method='decentralized_registration'
    # method_kwargs = dict(
    #     pairwise_displacement_method='phase_cross_correlation',
    #     convergence_method='gradient_descent_robust',
    # ),

    non_rigid_kwargs=None,
    # non_rigid_kwargs=dict(bin_step_um=200),

)

preprocess_and_peak_recording_params = dict(
    filter_params=filter_params,
    detect_peaks_params=detect_peaks_params,
    localize_peaks_params=localize_peaks_params,
    estimate_motion_params=estimate_motion_params,
)
