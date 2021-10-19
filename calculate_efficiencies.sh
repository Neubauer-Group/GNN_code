#!/bin/#!/usr/bin/env bash

# python track_efficiency.py --group='test_5gev_no_edge' --n_events=10 |& tee test10_no_edge.log
# python track_efficiency.py --group='test_5gev_empty_edge' --n_events=10 |& tee test10_empty_edge.log
# python track_efficiency.py --group='test_5gev_non_empty_edge' --n_events=10 |& tee test10_non_empty_edge.log
# python track_efficiency.py --group='test_5gev_hough' --n_events=10 |& tee test10_hough.log

# python track_efficiency.py --group='test_no_edge' --n_events=10 |& tee test10_no_edge.log
# python track_efficiency.py --group='test_empty_edge' --n_events=10 |& tee test10_empty_edge.log
# python track_efficiency.py --group='test_non_empty_edge' --n_events=10 |& tee test10_non_empty_edge.log
# python track_efficiency.py --group='test_hough' --n_events=10 |& tee test10_hough.log
#
# python track_efficiency.py --group='test_no_edge' --n_events=200 |& tee test200_no_edge.log
# python track_efficiency.py --group='test_empty_edge' --n_events=200 |& tee test200_empty_edge.log
# python track_efficiency.py --group='test_non_empty_edge' --n_events=200 |& tee test200_non_empty_edge.log
# python track_efficiency.py --group='test_hough' --n_events=200 |& tee test200_hough.log


# python track_efficiency.py --group='test_1.5gev_edge' --n_events=2 |& tee test2_1.5gev_edge.log


# python track_efficiency.py --group='test_1.5gev_edge' --n_events=200 |& tee test200_1.5gev_edge.log
# python track_efficiency.py --group='test_1gev_edge' --n_events=200 |& tee test200_1gev_edge.log
# python track_efficiency.py --group='test_5gev_non_empty_edge' --n_events=20 |& tee test20_.5gev_edge.log


# python track_efficiency.py --group='test_.75gev_edge' --n_events=50 |& tee test50_.75gev_edge.log
# python track_efficiency.py --group='test_.6gev_edge' --n_events=25 |& tee test25_.6gev_edge.log


# python track_efficiency.py --group='test_noise' --n_events=200 |& tee test_noise.log

# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=0.5 --pt_max=0.6 |& tee test_noise_0.5_0.6.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=0.6 --pt_max=0.7 |& tee test_noise_0.6_0.7.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=0.7 --pt_max=0.8 |& tee test_noise_0.7_0.8.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=0.8 --pt_max=0.9 |& tee test_noise_0.8_0.9.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=0.9 --pt_max=1.0 |& tee test_noise_0.9_1.0.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=1.0 --pt_max=1.2 |& tee test_noise_1.0_1.2.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=1.2 --pt_max=1.4 |& tee test_noise_1.2_1.4.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=1.4 --pt_max=1.6 |& tee test_noise_1.4_1.6.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=1.6 --pt_max=1.8 |& tee test_noise_1.6_1.8.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=1.8 --pt_max=2.0 |& tee test_noise_1.8_2.0.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=2.0 --pt_max=2.5 |& tee test_noise_2.0_2.5.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=2.5 --pt_max=3.0 |& tee test_noise_2.5_3.0.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=3.0 --pt_max=3.5 |& tee test_noise_3.0_3.5.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=3.5 --pt_max=4.0 |& tee test_noise_3.5_4.0.log
# python track_efficiency.py --group='test_noise' --n_events=200 --pt_min=4.0 --pt_max=5.0 |& tee test_noise_4.0_5.0.log
# python track_efficiency.py --group='test_noise' --n_events=5 --pt_min=0.0 --pt_max=5000.0 |& tee test_noise_0.5_5.0.log



# python track_efficiency.py --group='test_noise' --model='test_5gev_no_phi' |& tee test_no_phi.log
# python track_efficiency.py --group='test_noise' --model='test_5gev_phi' |& tee test_phi.log
# python track_efficiency.py --group='test_noise_2' --model='test_5gev_no_phi_no_edge' |& tee test_no_phi_no_edge.log
# python track_efficiency.py --group='test_noise_2' --model='test_5gev_phi_no_edge' |& tee test_phi_no_edge.log

# python track_efficiency.py --group='test_noise' --model='test_5gev_phi' |& tee test_no_phi.log

# python track_efficiency.py --group='edge_classifier' --model='edge_classifier' --type='edge_classifier' |& tee edge_classifier.log
# python track_efficiency.py --group='edge_classifier_ext_barrel' --model='edge_classifier_ext_barrel' --type='edge_classifier' |& tee edge_classifier_ext_barrel.log
#
# python track_efficiency.py --group='interaction_network' --model='interaction_network' --type='interaction_network' |& tee interaction_network.log
# python track_efficiency.py --group='interaction_network_ext_barrel' --model='interaction_network_ext_barrel' --type='interaction_network' |& tee interaction_network_ext_barrel.log

# python track_efficiency.py --group='atlas_EC' --model='atlas_EC' --type='edge_classifier' |& tee edge_classifier_atlas.log
# python track_efficiency.py --group='atlas_IN' --model='atlas_IN' --type='interaction_network' |& tee interaction_network_atlas.log

# python track_efficiency.py --group='char_EC' --model='char_EC' --type='edge_classifier' |& tee edge_classifier_char.log
# python track_efficiency.py --group='char_IN' --model='char_IN' --type='interaction_network' |& tee interaction_network_char.log


python scripts/track_efficiency.py --group='test_run' --model='test_run' --type='interaction_network' |& tee track_efficiecny_test_run.log
