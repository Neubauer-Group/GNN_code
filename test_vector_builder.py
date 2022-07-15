from torch_geometric.data import Data, Dataset
from datasets.particle import TrackMLParticleTrackingDataset
from models.EdgeNetWithCategories import EdgeNetWithCategories
from models.InteractionNetwork import InteractionNetwork
# from princeton_gnn_tracking.models.EC1.ec1 import GNNSegmentClassifier

import pickle
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from unionfind import unionfind
from matplotlib import pyplot as plt


def main(args):
    """Main function"""
    # Parse the command line
    # args = parse_args()

    group = args.group
    model_group = args.model
    type = args.type
    config_file = '/data/gnn_code/training_data/' + group + '/config.yaml'
    # Nevents = args.n_events
    # pt_range = [args.pt_min, args.pt_max]

    with open(config_file) as f:
        config = yaml.load(f)
        selection = config['selection']
        # n_events = config['n_files']
        # n_events = 100
        n_events = 8850
        # n_events = 1960
        # n_events = 500
        # n_events = 4000


    trackml_data = TrackMLParticleTrackingDataset('/data/gnn_code/training_data/' + group,
    # trackml_data = TrackMLParticleTrackingDataset('/data/gnn_code/training_data/test_geometric',
                                                  volume_layer_ids=selection['volume_layer_ids'],
                                                  layer_pairs=selection['layer_pairs'],
                                                  layer_pairs_plus=selection['layer_pairs_plus'],
                                                  pt_range=selection['pt_range'],
                                                  # pt_range=pt_range,
                                                  eta_range=selection['eta_range'],
                                                  phi_slope_max=selection['phi_slope_max'],
                                                  # phi_slope_max=.001,
                                                  z0_max=selection['z0_max'],
                                                  diff_phi=selection['diff_phi'],
                                                  diff_z=selection['diff_z'],
                                                  # z0_max=150,
                                                  n_phi_sections=selection['n_phi_sections'],
                                                  n_eta_sections=selection['n_eta_sections'],
                                                  # n_phi_sections=4,
                                                  # n_eta_sections=2,
                                                  # augments=selection['construct_augmented_graphs'],
                                                  augments=False,
                                                  intersect=selection['remove_intersecting_edges'],
                                                  hough=selection['hough_transform'],
                                                  noise=selection['noise'],
                                                  duplicates=selection['duplicates'],
                                                  secondaries=selection['secondaries'],
                                                  tracking=True,
                                                  # n_workers=24,
                                                  n_tasks=1,
                                                  n_events=n_events,
                                                  data_type=config['data_type'],
                                                  test_vectors=selection['test_vectors'],
                                                  mmap=selection['module_map'],
                                                  N_modules=selection['N_modules']
                                                  # n_events=Nevents,
                                                  # directed=True,
                                                  # layer_pairs_plus=True,
                                                  )

    if (args.reprocess):
        trackml_data.process(True)
        print("done")
        # trackml_data.draw(0)

    trackml_data.construct_module_map()
    # trackml_data.construct_module_map_atlas()


    # Print properties of the data set

    # print()
    # print("Graph Construction Features")
    # print("number_of_graphs = " + str(n_events))
    # print("number_of_phi_sections = " + str(selection['n_phi_sections']))
    # print("number_of_eta_sections = " + str(selection['n_eta_sections']))
    # print("edge_features_used = " + str(selection['hough_transform']))
    # print("layer_pairs_plus_used = " + str(selection['layer_pairs_plus']))
    # print()
    # print("Truth Level Cuts Applied")
    # print("pt_range = " + str(selection['pt_range']))
    # print("noise_hits_present = " + str(selection['noise']))
    # print("duplicate_hits_from_same_paticle_within_layer_present = " + str(selection['duplicates']))
    # print()
    # print("Geometric Cuts Applied")
    # print("eta_range = " + str(selection['eta_range']))
    # print("phi_slope_max = " + str(selection['phi_slope_max']))
    # print("z0_max = " + str(selection['z0_max']))
    # print("intersecting_line_cut_used = " + str(selection['remove_intersecting_edges']))
    # print()
    # print("Inference Graph Node Properties")
    print("Average_Graph_Node_Count = " + str(trackml_data.average_node_count))
    # print("Average_Total_Pixel_Node_Count = " + str(trackml_data.average_total_pixel_node_count))
    # print("Average_Total_Global_Node_Count = " + str(trackml_data.average_total_node_count))
    # print()
    print("Average_Graph_Edge_Count = " + str(trackml_data.average_edge_count))
    print("Average_Graph_True_Edge_Count = " + str(trackml_data.average_true_edge_count))
    # print("Inference Graph Edge Properties")
    # print("Average_Pruned_Pixel_True_Edge_Count = " + str(trackml_data.average_pruned_pixel_true_edge_count))
    # print("Average_Total_Pixel_True_Edge_Count = " + str(trackml_data.average_total_pixel_true_edge_count))
    # print("Average_Total_Global_True_Edge_Count = " + str(trackml_data.average_total_true_edge_count))
    # print()
    # print("Inference Graph Track Properties")
    # print("Average_Pixel_Track_Threshold_Count = " + str(trackml_data.average_pixel_track_threshold_count))
    # print("Average_Pixel_Track_Count = " + str(trackml_data.average_pixel_track_count))
    # print("Average_Global_Track_Count = " + str(trackml_data.average_total_track_count))
    # print()



    # for i in tqdm(range(n_events)):
    #     if (i == 0):
    #         z0 = trackml_data[i].edge_attr[:,1].numpy()
    #         dp = trackml_data[i].edge_attr[:,2].numpy()
    #         y  = trackml_data[i].y.numpy()
    #     else:
    #         z0 = np.concatenate((z0, trackml_data[i].edge_attr[:,1].numpy()))
    #         dp = np.concatenate((dp, trackml_data[i].edge_attr[:,2].numpy()))
    #         y  = np.concatenate((y,  trackml_data[i].y.numpy()))
    #
    # # true_y  = y[y==1]
    # true_z0 = z0[y==1]
    # true_dp = dp[y==1]
    # # false_y = y[y==0]
    # false_z0 = z0[y==0]
    # false_dp = dp[y==0]
    #
    # print(np.amin(true_z0), np.amax(true_z0))
    #
    #
    # fig2, (ax2) = plt.subplots(1,1, figsize=(6,6))
    # ax2.set_xlabel('Edge z0 [mm]')
    # ax2.set_ylabel('Frequency')
    # ax2.hist(false_z0, bins=200, range=(-1000000,1000000), histtype='step', edgecolor='blue',   fill=False, label='False Edges')
    # ax2.hist(true_z0, bins=200, range=(-1000000,1000000), histtype='step', edgecolor='orange',   fill=False, label='True Edges')
    # ax2.legend(loc='upper right')
    # ax2.set_yscale("log")
    # fig2.savefig("z0_distribution.png")
    #
    # fig3, (ax3) = plt.subplots(1,1, figsize=(6,6))
    # ax3.set_xlabel('Edge phi slope [mm]')
    # ax3.set_ylabel('Frequency')
    # ax3.hist(false_dp, bins=200, range=(-10,10), histtype='step', edgecolor='blue',   fill=False, label='False Edges')
    # ax3.hist(true_dp, bins=200, range=(-10,10), histtype='step', edgecolor='orange',   fill=False, label='True Edges')
    # ax3.legend(loc='upper right')
    # ax3.set_yscale("log")
    # fig3.savefig("phi_distribution.png")
    #
    #
    # for i in tqdm(range(5000)):
    #     # z0cut = 50 - .01 * i
    #     z0cut = 2500 - .5 * i
    #     true_z0_cut = true_z0[np.absolute(true_z0) <= z0cut]
    #     false_z0_cut = false_z0[np.absolute(false_z0) <= z0cut]
    #     if (i == 0):
    #         cut    = z0cut
    #         purity = true_z0_cut.shape[0] / (true_z0_cut.shape[0] + false_z0_cut.shape[0])
    #         efficiency = true_z0_cut.shape[0] / true_z0.shape[0]
    #     else:
    #         cut = np.append(cut, z0cut)
    #         purity = np.append(purity, true_z0_cut.shape[0] / (true_z0_cut.shape[0] + false_z0_cut.shape[0]))
    #         efficiency = np.append(efficiency, true_z0_cut.shape[0] / true_z0.shape[0])
    #
    # z0_start = cut[np.absolute(efficiency - .97).argmin()]
    # print("99.95% efficient", cut[np.absolute(efficiency - .9995).argmin()],  purity[np.absolute(efficiency - .9995).argmin()])
    # print("99.9% efficient", cut[np.absolute(efficiency - .999).argmin()],  purity[np.absolute(efficiency - .999).argmin()])
    # print("99.8% efficient", cut[np.absolute(efficiency - .998).argmin()],  purity[np.absolute(efficiency - .998).argmin()])
    # print("99.7% efficient", cut[np.absolute(efficiency - .997).argmin()],  purity[np.absolute(efficiency - .997).argmin()])
    # print("99.6% efficient", cut[np.absolute(efficiency - .996).argmin()],  purity[np.absolute(efficiency - .996).argmin()])
    # print("99.5% efficient", cut[np.absolute(efficiency - .995).argmin()],  purity[np.absolute(efficiency - .995).argmin()])
    # print("99% efficient", cut[np.absolute(efficiency - .99).argmin()],  purity[np.absolute(efficiency - .99).argmin()])
    # print("98% efficient", cut[np.absolute(efficiency - .98).argmin()],  purity[np.absolute(efficiency - .98).argmin()])
    # print("97% efficient", cut[np.absolute(efficiency - .97).argmin()],  purity[np.absolute(efficiency - .97).argmin()])
    # print("max purity", cut[purity.argmax()], purity[purity.argmax()], efficiency[purity.argmax()])
    #
    #
    # fig0, (ax0) = plt.subplots(1,1, figsize=(6,6))
    # ax0.set_xlabel('max Z0 cut [mm]')
    # ax0.scatter(cut, purity, c='blue', s=.25, label='Purity')
    # ax0.scatter(cut, efficiency, c='black', s=.25, label='Efficiency')
    # ax0.legend(loc='upper right')
    # fig0.savefig("z0_cuts.png")
    #
    #
    # for i in tqdm(range(5000)):
    #     dpcut = .005 - .000001 * i
    #     true_dp_cut = true_dp[np.absolute(true_dp) <= dpcut]
    #     false_dp_cut = false_dp[np.absolute(false_dp) <= dpcut]
    #     if (i == 0):
    #         cut    = dpcut
    #         purity = true_dp_cut.shape[0] / (true_dp_cut.shape[0] + false_dp_cut.shape[0])
    #         efficiency = true_dp_cut.shape[0] / true_dp.shape[0]
    #     else:
    #         cut = np.append(cut, dpcut)
    #         purity = np.append(purity, true_dp_cut.shape[0] / (true_dp_cut.shape[0] + false_dp_cut.shape[0]))
    #         efficiency = np.append(efficiency, true_dp_cut.shape[0] / true_dp.shape[0])
    #
    # dp_start = cut[np.absolute(efficiency - .97).argmin()]
    # print("99.95% efficient", cut[np.absolute(efficiency - .9995).argmin()],  purity[np.absolute(efficiency - .9995).argmin()])
    # print("99.9% efficient", cut[np.absolute(efficiency - .999).argmin()],  purity[np.absolute(efficiency - .999).argmin()])
    # print("99.8% efficient", cut[np.absolute(efficiency - .998).argmin()],  purity[np.absolute(efficiency - .998).argmin()])
    # print("99.7% efficient", cut[np.absolute(efficiency - .997).argmin()],  purity[np.absolute(efficiency - .997).argmin()])
    # print("99.6% efficient", cut[np.absolute(efficiency - .996).argmin()],  purity[np.absolute(efficiency - .996).argmin()])
    # print("99.5% efficient", cut[np.absolute(efficiency - .995).argmin()],  purity[np.absolute(efficiency - .995).argmin()])
    # print("99% efficient", cut[np.absolute(efficiency - .99).argmin()],  purity[np.absolute(efficiency - .99).argmin()])
    # print("98% efficient", cut[np.absolute(efficiency - .98).argmin()],  purity[np.absolute(efficiency - .98).argmin()])
    # print("97% efficient", cut[np.absolute(efficiency - .97).argmin()],  purity[np.absolute(efficiency - .97).argmin()])
    # print("max purity", cut[purity.argmax()], purity[purity.argmax()], efficiency[purity.argmax()])
    #
    # fig1, (ax1) = plt.subplots(1,1, figsize=(6,6))
    # ax1.set_xlabel('max phi slope cut')
    # ax1.scatter(cut, purity, c='blue', s=.25, label='Purity')
    # ax1.scatter(cut, efficiency, c='black', s=.25, label='Efficiency')
    # ax3.legend(loc='upper right')
    # fig1.savefig("phi_slope_cuts.png")
    #
    # for i in tqdm(range(5000)):
    #     # z0cut = z0_start + .01 * i
    #     z0cut = z0_start + .1 * i
    #     dpcut = dp_start + .000001 * i
    #
    #     true_dp_cutz0 = true_dp[np.absolute(true_z0) <= z0cut]
    #     false_dp_cutz0 = false_dp[np.absolute(false_z0) <= z0cut]
    #     true_cut = true_dp_cutz0[np.absolute(true_dp_cutz0) <= dpcut]
    #     false_cut = false_dp_cutz0[np.absolute(false_dp_cutz0) <= dpcut]
    #
    #     purity = true_cut.shape[0] / (true_cut.shape[0] + false_cut.shape[0])
    #     efficiency = true_cut.shape[0] / true_dp.shape[0]
    #
    #     if (purity <= .5):
    #         print("50% purity", z0cut, dpcut, purity, efficiency)
    #         break

def get_particle_pt_eta(particle_id, track_attr):
    particle = track_attr[track_attr[:,0] == particle_id]
    # print(particle[0,1], particle[0,2])
    return particle[0,1], particle[0,2]


def get_node_eta(pos):
    r = pos[0]
    z = pos[2]
    return np.arcsinh(z/r)


def div(a,b):
    return np.divide(a,b, out=np.zeros_like(a, dtype=float), where=b!=0)


def div_err(c, a_e, a, b_e, b):
    return c * np.sqrt(div(a_e,a)**2 + div(b_e,b)**2)


def sum_err(a, b, c=0, d=0):
    return np.sqrt(a**2 + b**2 + c**2 + d**2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', '-g', default='test_no_edge', help='Group name of the inference graphs')
    parser.add_argument('--model', '-m', default='test_no_edge', help='Group name of the model to run on')
    parser.add_argument('--type', '-t', default='edge_classifier', help='Type of model to load')
    parser.add_argument('--reprocess', '-r', action='store_true', help='toggle reprocessing inference graphs')
    # parser.add_argument('--n_events', default=10, type=int, help='How many events to average over?')
    # parser.add_argument('--pt_min', default=1.5, type=float, help='lower pt range')
    # parser.add_argument('--pt_max', default=2.0, type=float, help='upper pt range')

    args = parser.parse_args()
    main(args)
