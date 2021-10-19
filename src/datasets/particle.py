import os.path as osp
import glob

import copy as copy
import multiprocessing as mp
from tqdm import tqdm
import random
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import is_undirected
from torch_geometric.data import Data, Dataset


class TrackMLParticleTrackingDataset(Dataset):
    r"""The `TrackML Particle Tracking Challenge
    <https://www.kaggle.com/c/trackml-particle-identification>`_ dataset to
    reconstruct particle tracks from 3D points left in the silicon detectors.

    Args:
        root (string): Root directory where the dataset should be saved.

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)

        n_events (int): Number of events in the raw folder to process



    GRAPH CONSTRUCTION PARAMETERS
    ###########################################################################

        volume_layer_ids (List): List of the volume and layer ids to be included
            in the graph. Layers get indexed by increasing volume and layer id.
            Refer to the following map for the layer indices, and compare them
            to the chart at https://www.kaggle.com/c/trackml-particle-identification/data

                                            41
                        34 --- 39            |        42 --- 47
                                            40

                                            27
                        18 --- 23            |        28 --- 33
                                            24

                                            10
                         0 ---  6            |        11 --- 17
                                             7

        layer_pairs (List): List of which pairs of layers can have edges between them.
            Uses the layer indices described above to reference layers.
            Example for Barrel Only:
            [[7,8],[8,9],[9,10],[10,24],[24,25],[25,26],[26,27],[27,40],[40,41]]

        pt_range ([min, max]): A truth cut applied to reduce the number of nodes in the graph.
            Only nodes associated with particles in this momentum range are included.

        eta_range ([min, max]): A cut applied to nodes to select a specific eta

        phi_slope_max (float32): A cut applied to edges to limit the change in phi between
            the two nodes.

        z0_max (float32): A cut applied to edges that limits how far from the center of
            the detector the particle edge can originate from.

        n_phi_sections (int): Break the graph into multiple segments in the phi direction.

        n_eta_sections (int): Break the graph into multiple segments in the eta direction.

        augments (bool): Toggle for turning data augmentation on and off

        intersect (bool): Toggle for interseting lines cut. When connecting Barrel
            edges to the inner most endcap layer, sometimes the edge passes through
            the layer above, this cut removes those edges.

        hough (bool): Toggle for using a hough transform to construct an edge weight.
            Each node in the graph casts votes into an accumulator for a linear
            parameter space. The edge is then used to address this accumulator and
            extract the vote count.

        noise (bool): Toggle if you want noise hits in the graph

        tracking (bool): Toggle for building truth tracks. Track data is a tensor with
            dimensions (Nx5) with the following columns:
            [r coord, phi coord, z coord, layer index, track number]

        directed (bool): Edges are directed, for an undirected graph, edges are
            duplicated and in reverse direction.

        layer_pairs_plus (bool): Allows for edge connections within the same layer


    MULTIPROCESSING PARAMETERS
    ###########################################################################

        n_workers (int): Number of worker nodes for multiprocessing

        n_tasks (int): Break the processing into a number of tasks

    """

    url = 'https://www.kaggle.com/c/trackml-particle-identification'

    def __init__(self, root, transform=None, n_events=0,
                 directed=False, layer_pairs_plus=False,
                 volume_layer_ids=[[8, 2], [8, 4], [8, 6], [8, 8]], #Layers Selected
                 layer_pairs=[[7, 8], [8, 9], [9, 10]],             #Connected Layers
                 pt_range=[1.5, 2], eta_range=[-5, 5],              #Node Cuts
                 phi_slope_max=0.0006, z0_max=150,                  #Edge Cuts
                 n_phi_sections=1, n_eta_sections=1,                #N Sections
                 augments=False, intersect=False,                   #Toggle Switches
                 hough=False, tracking=False,                       #Toggle Switches
                 noise=False, duplicates=False,                     #Truth Toggle Switches
                 n_workers=mp.cpu_count(), n_tasks=1,               #multiprocessing
                 mmap=False,                                        #module map
                 data_type="TrackML"                                #Other Detectors
                 ):
        events = glob.glob(osp.join(osp.join(root, 'raw'), 'event*-truth.csv'))
        events = [e.split(osp.sep)[-1].split('-')[0][5:] for e in events]
        self.events = sorted(events)
        if (n_events > 0):
            self.events = self.events[:n_events]

        self.data_type        = data_type
        self.mmap             = mmap

        self.directed         = directed
        self.layer_pairs_plus = layer_pairs_plus
        self.volume_layer_ids = torch.tensor(volume_layer_ids)
        self.layer_pairs      = torch.tensor(layer_pairs)
        self.pt_range         = pt_range
        self.eta_range        = eta_range
        self.phi_slope_max    = phi_slope_max
        self.z0_max           = z0_max
        self.n_phi_sections   = n_phi_sections
        self.n_eta_sections   = n_eta_sections
        self.augments         = augments
        self.intersect        = intersect
        self.hough            = hough
        self.noise            = noise
        self.duplicates       = duplicates
        self.tracking         = tracking
        self.n_workers        = n_workers
        self.n_tasks          = n_tasks

        self.accum0_m          = [-30, 30, 2000]          # cot(theta) [eta]
        self.accum0_b          = [-20, 20, 2000]          # z0
        self.accum1_m          = [-.0003, .0003, 2000]    # phi-slope  [qA/pT]
        self.accum1_b          = [-3.3, 3.3, 2000]        # phi0

        # bin = 2000
        # m = torch.cot(2*torch.atan(torch.e^(-eta_range)))
        # self.accum0_m          = [m[0], m[1], bin]                      # cot(theta) [eta]
        # # self.accum0_b          = [-z0_max, z0_max, bin]                 # z0
        # self.accum0_b          = [-20, 20, bin]                 # z0
        # self.accum1_m          = [-phi_slope_max, phi_slope_max, bin]   # phi-slope  [qA/pT]
        # self.accum1_b          = [-np.pi, np.pi, bin]                   # phi0

        super(TrackMLParticleTrackingDataset, self).__init__(root, transform)


    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = sorted(glob.glob(self.raw_dir+'/*.csv'))
        return [f.split('/')[-1] for f in self.input_files]


    @property
    def processed_file_names(self):
        N_sections = self.n_phi_sections*self.n_eta_sections
        if not hasattr(self,'processed_files'):
            proc_names = ['event{}_section{}.pt'.format(idx, i) for idx in self.events for i in range(N_sections)]
            if(self.augments):
                proc_names_aug = ['event{}_section{}_aug.pt'.format(idx, i) for idx in self.events for i in range(N_sections)]
                proc_names = [x for y in zip(proc_names, proc_names_aug) for x in y]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files


    @property
    def average_node_count(self):
        if not hasattr(self,'node_avg'):
            N_nodes = np.asarray([self[idx].x.shape[0] for idx in range(len(self.events))])
            self.node_avg = N_nodes.mean()

            fig0, (ax0) = plt.subplots(1, 1, dpi=500, figsize=(6, 6))
            ax0.hist(N_nodes)
            ax0.set_xlabel('Nodes')
            ax0.set_ylabel('Count')
            # ax0.set_xlim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
            # ax0.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
            fig0.savefig('Nodes_distribution.pdf', dpi=500)


        return self.node_avg


    @property
    def maximum_node_count(self):
        if not hasattr(self,'node_max'):
            N_nodes = np.asarray([self[idx].x.shape[0] for idx in range(len(self.events))])
            self.node_max = N_nodes.max()
        return self.node_max


    @property
    def average_total_node_count(self):
        if not hasattr(self,'total_node_avg'):
            N_total_nodes = np.asarray([self[idx].tracks.shape[0] for idx in range(len(self.events))])
            self.total_node_avg = N_total_nodes.mean()
        return self.total_node_avg


    @property
    def average_total_pixel_node_count(self):
        if not hasattr(self,'total_pixel_node_avg'):
            N_total_nodes = np.asarray([self[idx].tracks[self[idx].tracks[:,3] < 18].shape[0] for idx in range(len(self.events))])
            self.total_pixel_node_avg = N_total_nodes.mean()
        return self.total_pixel_node_avg


    @property
    def average_edge_count(self):
        if not hasattr(self,'edge_avg'):
            N_edges = np.asarray([self[idx].y.shape[0] for idx in range(len(self.events))])
            self.edge_avg = N_edges.mean()

            fig0, (ax0) = plt.subplots(1, 1, dpi=500, figsize=(6, 6))
            ax0.hist(N_edges)
            ax0.set_xlabel('Edges')
            ax0.set_ylabel('Count')
            # ax0.set_xlim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
            # ax0.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
            fig0.savefig('Edges_distribution.pdf', dpi=500)

        return self.edge_avg


    @property
    def maximum_edge_count(self):
        if not hasattr(self,'edge_max'):
            N_edges = np.asarray([self[idx].y.shape[0] for idx in range(len(self.events))])
            self.edge_max = N_edges.max()
        return self.edge_max


    @property
    def average_true_edge_count(self):
        if not hasattr(self,'true_edge_avg'):
            N_true_edges = np.asarray([torch.sum(self[idx].y) for idx in range(len(self.events))])
            self.true_edge_avg = N_true_edges.mean()

            fig0, (ax0) = plt.subplots(1, 1, dpi=500, figsize=(6, 6))
            ax0.hist(N_true_edges)
            ax0.set_xlabel('True Edges')
            ax0.set_ylabel('Count')
            # ax0.set_xlim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
            # ax0.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
            fig0.savefig('True_edges_distribution.pdf', dpi=500)

        return self.true_edge_avg


    @property
    def maximum_true_edge_count(self):
        if not hasattr(self,'true_edge_max'):
            N_true_edges = np.asarray([torch.sum(self[idx].y) for idx in range(len(self.events))])
            self.true_edge_max = N_true_edges.max()
        return self.true_edge_max


    @property
    def average_total_true_edge_count(self):
        if not hasattr(self,'total_true_edge_avg'):
            true_edges = np.asarray([torch.sum(self[idx].track_attr[:,3])-self[idx].track_attr.shape[0] for idx in range(len(self.events))])
            if not self.directed:
                self.total_true_edge_avg = 2*true_edges.mean()
            else:
                self.total_true_edge_avg = true_edges.mean()
        return self.total_true_edge_avg


    @property
    def average_total_pixel_true_edge_count(self):
        if not hasattr(self,'total_pixel_true_edge_avg'):
            true_edges = np.asarray([torch.sum(self[idx].track_attr_pix[:,3])-self[idx].track_attr_pix.shape[0] for idx in range(len(self.events))])
            if not self.directed:
                self.total_pixel_true_edge_avg = 2*true_edges.mean()
            else:
                self.total_pixel_true_edge_avg = true_edges.mean()
        return self.total_pixel_true_edge_avg


    @property
    def average_pruned_pixel_true_edge_count(self):
        if not hasattr(self,'pruned_pixel_true_edge_avg'):
            true_edges = np.asarray([torch.sum(self[idx].track_attr_pruned[:,3])-self[idx].track_attr_pruned.shape[0] for idx in range(len(self.events))])
            if not self.directed:
                self.pruned_pixel_true_edge_avg = 2*true_edges.mean()
            else:
                self.pruned_pixel_true_edge_avg = true_edges.mean()
        return self.pruned_pixel_true_edge_avg


    @property
    def average_total_track_count(self):
        if not hasattr(self,'total_track_avg'):
            N_tracks = np.asarray([self[idx].track_attr.shape[0] for idx in range(len(self.events))])
            self.total_track_avg = N_tracks.mean()
        return self.total_track_avg


    @property
    def average_pixel_track_count(self):
        if not hasattr(self,'pixel_track_avg'):
            N_tracks = np.asarray([self[idx].track_attr_pix.shape[0] for idx in range(len(self.events))])
            self.pixel_track_avg = N_tracks.mean()
        return self.pixel_track_avg


    @property
    def average_pixel_track_threshold_count(self):
        if not hasattr(self,'pixel_track_threshold_avg'):
            N_tracks = np.asarray([self[idx].track_attr_pruned[self[idx].track_attr_pruned[:,3] > 2].shape[0] for idx in range(len(self.events))])
            self.pixel_track_threshold_avg = N_tracks.mean()
        return self.pixel_track_threshold_avg


    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.csv files to {}'.format(self.url, self.raw_dir))


    def len(self):
        N_events = len(self.events)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments


    def __len__(self):
        N_events = len(self.events)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments


    def read_hits(self, idx):
        hits_filename = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        hits = pandas.read_csv(
            hits_filename, usecols=['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id'],
            dtype={
                'hit_id': np.int64,
                'x': np.float32,
                'y': np.float32,
                'z': np.float32,
                'volume_id': np.int64,
                'layer_id': np.int64,
                'module_id': np.int64
            })
        return hits


    def read_cells(self, idx):
        cells_filename = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        cells = pandas.read_csv(
            cells_filename, usecols=['hit_id', 'ch0', 'ch1', 'value'],
            dtype={
                'hit_id': np.int64,
                'ch0': np.int64,
                'ch1': np.int64,
                'value': np.float32
            })
        return cells


    def read_particles(self, idx):
        particles_filename = osp.join(self.raw_dir, f'event{idx}-particles.csv')

        if self.data_type == "TrackML":
            particles = pandas.read_csv(
                particles_filename, usecols=['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits'],
                dtype={
                    'particle_id': np.int64,
                    'vx': np.float32,
                    'vy': np.float32,
                    'vz': np.float32,
                    'px': np.float32,
                    'py': np.float32,
                    'pz': np.float32,
                    'q': np.int64,
                    'nhits': np.int64
                })
        elif self.data_type == "ATLAS":
            particles = pandas.read_csv(
                particles_filename, usecols=['particle_id', 'px', 'py', 'pz', 'pt', 'eta', 'vx', 'vy', 'vz', 'radius', 'status', 'charge', 'pdgId', 'pass'],
                dtype={
                    'particle_id': np.int64,
                    'px': np.float32,
                    'py': np.float32,
                    'pz': np.float32,
                    'pt': np.float32,
                    'eta': np.float32,
                    'vx': np.float32,
                    'vy': np.float32,
                    'vz': np.float32,
                    'radius': np.float32,
                    'status': np.int64,
                    'charge': np.float32,
                    'pdgId': np.int64,
                    'pass': str
                })
        return particles


    def read_truth(self, idx):
        truth_filename = osp.join(self.raw_dir, f'event{idx}-truth.csv')

        if self.data_type == "TrackML":
            truth = pandas.read_csv(
                truth_filename, usecols=['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight'],
                dtype={
                    'hit_id': np.int64,
                    'particle_id': np.int64,
                    'tx': np.float32,
                    'ty': np.float32,
                    'tz': np.float32,
                    'tpx': np.float32,
                    'tpy': np.float32,
                    'tpz': np.float32,
                    'weight': np.float32
                })
        elif self.data_type == "ATLAS":
            truth = pandas.read_csv(
                # truth_filename, usecols=['hit_id', 'x', 'y', 'z', 'cluster_index_1', 'cluster_index_2', 'particle_id', 'hardware', 'cluster_x', 'cluster_y', 'cluster_z', 'barrel_endcap', 'layer_disk', 'eta_module', 'phi_module', 'eta_angle', 'phi_angle', 'norm_x', 'norm_y', 'norm_z'],
                truth_filename, usecols=['hit_id', 'x', 'y', 'z', 'cluster_index_1', 'cluster_index_2', 'particle_id', 'hardware', 'barrel_endcap', 'layer_disk', 'eta_module', 'phi_module'],
                dtype={
                    'hit_id': np.int64,
                    'x': np.float32,
                    'y': np.float32,
                    'z': np.float32,
                    'cluster_index_1': np.int64,
                    'cluster_index_2': np.int64,
                    'particle_id': np.int64,
                    'hardware': str,
                    # 'cluster_x': np.float32,
                    # 'cluster_y': np.float32,
                    # 'cluster_z': np.float32,
                    'barrel_endcap': np.int64,
                    'layer_disk': np.int64,
                    'eta_module': np.int64,
                    'phi_module': np.int64
                    # 'eta_angle': np.float32,
                    # 'phi_angle': np.float32,
                    # 'norm_x': np.float32,
                    # 'norm_y': np.float32,
                    # 'norm_z': np.float32
                })
        return truth


    def build_module_map(self, hits, particles, truth):
        return 1

    def select_hits(self, hits, particles, truth):
        # print('Selecting Hits')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        n_det_layers = len(valid_layer)

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        index = layer.unique(return_inverse=True)[1]
        hits = hits[['hit_id', 'x', 'y', 'z', 'module_id']].assign(layer=layer, index=index)


        valid_groups = hits.groupby(['layer'])
        hits = pandas.concat([valid_groups.get_group(valid_layer.numpy()[i]) for i in range(n_det_layers)])

        pt = np.sqrt(particles['px'].values**2 + particles['py'].values**2)
        particles = particles[np.bitwise_and(pt > self.pt_range[0], pt < self.pt_range[1])]

        # Manually creates the noise particle
        if self.noise:
            particles.loc[len(particles)] = [0,0,0,0,0,0,0,0,0]

        hits = (hits[['hit_id', 'x', 'y', 'z', 'module_id', 'index']].merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits = (hits.merge(particles[['particle_id']], on='particle_id'))



        r = np.sqrt(hits['x'].values**2 + hits['y'].values**2)
        phi = np.arctan2(hits['y'].values, hits['x'].values)
        theta = np.arctan2(r,hits['z'].values)
        eta = -1*np.log(np.tan(theta/2))
        hits = hits[['z', 'index', 'particle_id', 'module_id']].assign(r=r, phi=phi, eta=eta)


        # Splits out the noise hits from the true hits
        if self.noise:
            noise = hits.groupby(['particle_id']).get_group(0)
            hits = hits.drop(hits.groupby(['particle_id']).get_group(0).index)

        # Remove duplicate true hits within same layer
        if not self.duplicates:
            # hits = hits.loc[hits.groupby(['particle_id', 'index'], as_index=False).r.idxmin()]
            hits = hits.loc[hits.groupby(['particle_id', 'index'], as_index=False).r.idxmin().r.values.tolist()]

        # Append the noise hits back to the list
        if self.noise:
            hits = pandas.concat([noise, hits])


        r = torch.from_numpy(hits['r'].values)
        phi = torch.from_numpy(hits['phi'].values)
        z = torch.from_numpy(hits['z'].values)
        eta = torch.from_numpy(hits['eta'].values)
        layer = torch.from_numpy(hits['index'].values)
        particle = torch.from_numpy(hits['particle_id'].values)
        module = torch.from_numpy(hits['module_id'].values)
        pos = torch.stack([r, phi, z], 1)


        return pos, layer, particle, eta


    def select_hits_atlas(self, particles, truth):
        # print('Selecting Hits')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        n_det_layers = len(valid_layer)

        truth.loc[truth['hardware'] == 'STRIP','barrel_endcap'] = truth.loc[truth['hardware'] == 'STRIP','barrel_endcap'] + 100

        layer = torch.from_numpy(20 * truth['barrel_endcap'].values + truth['layer_disk'].values)
        index = layer.unique(return_inverse=True)[1]
        truth = truth[['hit_id', 'x', 'y', 'z', 'particle_id']].assign(layer=layer, index=index)

        valid_groups = truth.groupby(['layer'])
        truth = pandas.concat([valid_groups.get_group(valid_layer.numpy()[i]) for i in range(n_det_layers)])

        pt = particles['pt'].values/1000
        particles = particles[np.bitwise_and(pt > self.pt_range[0], pt < self.pt_range[1])]

        # Manually creates the noise particle
        if self.noise:
            particles.loc[len(particles)] = [0,0,0,0,0,0,0,0,0]

        hits = (truth.merge(particles[['particle_id']], on='particle_id'))


        r = np.sqrt(hits['x'].values**2 + hits['y'].values**2)
        phi = np.arctan2(hits['y'].values, hits['x'].values)
        theta = np.arctan2(r,hits['z'].values)
        eta = -1*np.log(np.tan(theta/2))
        hits = hits[['z', 'index', 'particle_id']].assign(r=r, phi=phi, eta=eta)

        # Splits out the noise hits from the true hits
        if self.noise:
            noise = hits.groupby(['particle_id']).get_group(0)
            hits = hits.drop(hits.groupby(['particle_id']).get_group(0).index)

        # Remove duplicate true hits within same layer
        if not self.duplicates:
            hits = hits.loc[hits.groupby(['particle_id', 'index'], as_index=False).r.idxmin()]

        # Append the noise hits back to the list
        if self.noise:
            hits = pandas.concat([noise, hits])


        r = torch.from_numpy(hits['r'].values)
        phi = torch.from_numpy(hits['phi'].values)
        z = torch.from_numpy(hits['z'].values)
        eta = torch.from_numpy(hits['eta'].values)
        layer = torch.from_numpy(hits['index'].values)
        particle = torch.from_numpy(hits['particle_id'].values)
        pos = torch.stack([r, phi, z], 1)

        return pos, layer, particle, eta


    def compute_edge_index(self, pos, layer):
        # print("Constructing Edge Index")
        edge_indices = torch.empty(2,0, dtype=torch.long)

        layer_pairs = self.layer_pairs
        if self.layer_pairs_plus:
            layers = layer.unique()
            layer_pairs_plus = torch.tensor([[layers[i],layers[i]] for i in range(layers.shape[0])])
            layer_pairs = torch.cat((layer_pairs, layer_pairs_plus), 0)

        for (layer1, layer2) in layer_pairs:
            mask1 = layer == layer1
            mask2 = layer == layer2
            nnz1 = mask1.nonzero().flatten()
            nnz2 = mask2.nonzero().flatten()

            dr   = pos[:, 0][mask2].view(1, -1) - pos[:, 0][mask1].view(-1, 1)
            dphi = pos[:, 1][mask2].view(1, -1) - pos[:, 1][mask1].view(-1, 1)
            dz   = pos[:, 2][mask2].view(1, -1) - pos[:, 2][mask1].view(-1, 1)
            dphi[dphi > np.pi] -= 2 * np.pi
            dphi[dphi < -np.pi] += 2 * np.pi

            # Calculate phi_slope and z0 which will be cut on
            phi_slope = dphi / dr
            z0 = pos[:, 2][mask1].view(-1, 1) - pos[:, 0][mask1].view(-1, 1) * dz / dr

            # Check for intersecting edges between barrel and endcap connections
            intersected_layer = dr.abs() < -1
            if (self.intersect and self.data_type == "TrackML"):
                if((layer1 == 7 and (layer2 == 6 or layer2 == 11)) or
                   (layer2 == 7 and (layer1 == 6 or layer1 == 11))):
                    z_int =  71.56298065185547 * dz / dr + z0
                    intersected_layer = z_int.abs() < 490.975
                elif((layer1 == 8 and (layer2 == 6 or layer2 == 11)) or
                     (layer2 == 8 and (layer1 == 6 or layer1 == 11))):
                    z_int = 115.37811279296875 * dz / dr + z0
                    intersected_layer = z_int.abs() < 490.975

            elif (self.intersect and self.data_type == "ATLAS"):
                if((layer1 == 21 and (layer2 == 15 or layer2 == 25)) or
                   (layer2 == 21 and (layer1 == 15 or layer1 == 25))):
                    z_int =  562 * dz / dr + z0
                    intersected_layer = z_int.abs() < 1400
                elif((layer1 == 22 and (layer2 == 15 or layer2 == 25)) or
                     (layer2 == 22 and (layer1 == 15 or layer1 == 25))):
                    z_int = 762 * dz / dr + z0
                    intersected_layer = z_int.abs() < 1400

                elif((layer1 == 4 and layer2 == 15) or (layer1 == 14 and layer2 == 25) or
                     (layer2 == 4 and layer1 == 15) or (layer2 == 14 and layer1 == 25)):
                    z_int = 405 * dz / dr + z0
                    intersected_layer = z_int.abs() < 1400

                elif((layer1 == 4 and layer2 == 20) or (layer1 == 14 and layer2 == 30) or
                     (layer2 == 4 and layer1 == 20) or (layer2 == 14 and layer1 == 30)):
                    r0 = pos[:, 0][mask1].view(-1, 1) - pos[:, 2][mask1].view(-1, 1) * dr / dz
                    r_int = 2602 * dr / dz + r0
                    intersected_layer = r_int.abs() > 384.5
                    # intersected_layer = r_int > 384.5 & r_int < 967.8
                elif((layer1 == 4 and layer2 == 19) or (layer1 == 14 and layer2 == 29) or
                     (layer2 == 4 and layer1 == 19) or (layer2 == 14 and layer1 == 29)):
                    r0 = pos[:, 0][mask1].view(-1, 1) - pos[:, 2][mask1].view(-1, 1) * dr / dz
                    r_int = 2252 * dr / dz + r0
                    intersected_layer = r_int.abs() > 384.5
                    # intersected_layer = r_int > 384.5 & r_int < 967.8
                elif((layer1 == 4 and layer2 == 18) or (layer1 == 14 and layer2 == 28) or
                     (layer2 == 4 and layer1 == 18) or (layer2 == 14 and layer1 == 28)):
                    r0 = pos[:, 0][mask1].view(-1, 1) - pos[:, 2][mask1].view(-1, 1) * dr / dz
                    r_int = 1952 * dr / dz + r0
                    intersected_layer = r_int.abs() > 384.5
                    # intersected_layer = r_int > 384.5 & r_int < 967.8
                elif((layer1 == 4 and layer2 == 17) or (layer1 == 14 and layer2 == 27) or
                     (layer2 == 4 and layer1 == 17) or (layer2 == 14 and layer1 == 27)):
                    r0 = pos[:, 0][mask1].view(-1, 1) - pos[:, 2][mask1].view(-1, 1) * dr / dz
                    r_int = 1702 * dr / dz + r0
                    intersected_layer = r_int.abs() > 384.5
                    # intersected_layer = r_int > 384.5 & r_int < 967.8
                elif((layer1 == 4 and layer2 == 16) or (layer1 == 14 and layer2 == 26) or
                     (layer2 == 4 and layer1 == 16) or (layer2 == 14 and layer1 == 26)):
                    r0 = pos[:, 0][mask1].view(-1, 1) - pos[:, 2][mask1].view(-1, 1) * dr / dz
                    r_int = 1512 * dr / dz + r0
                    intersected_layer = r_int.abs() > 384.5
                    # intersected_layer = r_int > 384.5 & r_int < 967.8

            adj = (phi_slope.abs() < self.phi_slope_max) & (z0.abs() < self.z0_max) & (intersected_layer == False)

            row, col = adj.nonzero().t()
            row = nnz1[row]
            col = nnz2[col]
            edge_index = torch.stack([row, col], dim=0)

            edge_indices = torch.cat((edge_indices, edge_index), 1)

        return edge_indices


    def compute_y_index(self, edge_indices, particle):
        # print("Constructing y Index")
        pid1 = [ particle[i].item() for i in edge_indices[0] ]
        pid2 = [ particle[i].item() for i in edge_indices[1] ]
        # print(pid1)
        # print(pid2)
        y = np.zeros(edge_indices.shape[1], dtype=np.int64)
        for i in range(edge_indices.shape[1]):
            if pid1[i] == pid2[i] and pid1[i] != 0:
                y[i] = 1

        return torch.from_numpy(y)



    def split_detector_sections(self, pos, layer, particle, eta, phi_edges, eta_edges):
        pos_sect, layer_sect, particle_sect = [], [], []

        for i in range(len(phi_edges) - 1):
            phi_mask1 = pos[:,1] > phi_edges[i]
            phi_mask2 = pos[:,1] < phi_edges[i+1]
            phi_mask  = phi_mask1 & phi_mask2
            phi_pos      = pos[phi_mask]
            phi_layer    = layer[phi_mask]
            phi_particle = particle[phi_mask]
            phi_eta      = eta[phi_mask]

            for j in range(len(eta_edges) - 1):
                eta_mask1 = phi_eta > eta_edges[j]
                eta_mask2 = phi_eta < eta_edges[j+1]
                eta_mask  = eta_mask1 & eta_mask2
                phi_eta_pos = phi_pos[eta_mask]
                phi_eta_layer = phi_layer[eta_mask]
                phi_eta_particle = phi_particle[eta_mask]
                pos_sect.append(phi_eta_pos)
                layer_sect.append(phi_eta_layer)
                particle_sect.append(phi_eta_particle)

        return pos_sect, layer_sect, particle_sect


    def read_event(self, idx):
        if self.data_type == "TrackML":
            hits      = self.read_hits(idx)
            particles = self.read_particles(idx)
            truth     = self.read_truth(idx)
        elif self.data_type == "ATLAS":
            hits = 0
            particles = self.read_particles(idx)
            truth     = self.read_truth(idx)

        return hits, particles, truth


    def process(self, reprocess=False):
        print('Constructing Graphs using n_workers = ' + str(self.n_workers))
        task_paths = np.array_split(self.processed_paths, self.n_tasks)
        for i in range(self.n_tasks):
            if reprocess or not self.files_exist(task_paths[i]):
                self.process_task(i)


    def process_task(self, idx):
        print('Running task ' + str(idx))
        task_events = np.array_split(self.events, self.n_tasks)
        with mp.Pool(processes = self.n_workers) as pool:
            pool.map(self.process_event, tqdm(task_events[idx]))


    def process_event(self, idx):
        hits, particles, truth = self.read_event(idx)

        if (self.mmap):
            module_map = self.build_module_map(hits, particles, truth)

        if self.data_type == "TrackML":
            pos, layer, particle, eta = self.select_hits(hits, particles, truth)
        elif self.data_type == "ATLAS":
            pos, layer, particle, eta = self.select_hits_atlas(particles, truth)

        tracks            = torch.empty(0, dtype=torch.long)
        track_attr        = torch.empty(0, dtype=torch.long)
        track_attr_pix    = torch.empty(0, dtype=torch.long)
        track_attr_pruned = torch.empty(0, dtype=torch.long)
        if(self.tracking):
            if self.data_type == "TrackML":
                tracks, track_attr, track_attr_pix, track_attr_pruned = self.build_tracks(hits, particles, truth)
            elif self.data_type == "ATLAS":
                tracks, track_attr, track_attr_pix, track_attr_pruned = self.build_tracks_atlas(particles, truth)

        phi_edges = np.linspace(*(-np.pi, np.pi), num=self.n_phi_sections+1)
        eta_edges = np.linspace(*self.eta_range, num=self.n_eta_sections+1)
        pos_sect, layer_sect, particle_sect = self.split_detector_sections(pos, layer, particle, eta, phi_edges, eta_edges)

        for i in range(len(pos_sect)):
            edge_index = self.compute_edge_index(pos_sect[i], layer_sect[i])
            y = self.compute_y_index(edge_index, particle_sect[i])

            edge_votes = torch.zeros(edge_index.shape[1], 0, dtype=torch.long)
            # edge_votes = torch.zeros(edge_index.shape[1], 2, dtype=torch.long)
            if(self.hough):
                # accumulator0, accumulator1 = self.build_accumulator(pos_sect[i])
                # edge_votes  = self.extract_votes(accumulator0, accumulator1, pos_sect[i], edge_index)
                edge_votes  = self.extract_votes(pos_sect[i], edge_index)

            data = Data(x=pos_sect[i], edge_index=edge_index, edge_attr=edge_votes, y=y, tracks=tracks, track_attr=track_attr, track_attr_pix=track_attr_pix, track_attr_pruned=track_attr_pruned, particles=particle_sect[i])

            if not self.directed and not data.is_undirected():
                rows,cols = data.edge_index
                temp = torch.stack((cols,rows))
                data.edge_index = torch.cat([data.edge_index,temp],dim=-1)
                data.y = torch.cat([data.y,data.y])
                data.edge_attr = torch.cat([data.edge_attr,data.edge_attr])


            if (self.augments):
                data_a = copy.deepcopy(data)
                data_a.x[:,1]= -data_a.x[:,1]
                torch.save(data_a, osp.join(self.processed_dir, 'event{}_section{}_aug.pt'.format(idx, i)))

            torch.save(data, osp.join(self.processed_dir, 'event{}_section{}.pt'.format(idx, i)))

        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue
        #
        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)


    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data


    def draw(self, idx, dpi=500):
        # print("Making plots for " + str(self.processed_files[idx]))
        width1 = .1   #blue edge (false)
        width2 = .2   #black edge (true)
        points = .25  #hit points
        dpi   = 500

        X = self[idx].x.cpu().numpy()
        index = self[idx].edge_index.cpu().numpy()
        y = self[idx].y.cpu().numpy()
        true_index = index[:,y > 0]

        r_co = X[:,0]
        phi_co = X[:,1]
        z_co = X[:,2]
        x_co = X[:,0]*np.cos(X[:,1])
        y_co = X[:,0]*np.sin(X[:,1])

        # scale = 12*z_co.max()/r_co.max()
        fig0, (ax0) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
        fig1, (ax1) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
        fig2, (ax2) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
        fig3, (ax3) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))

        # Adjust axes
        ax0.set_xlabel('Z [mm]')
        ax0.set_ylabel('R [mm]')
        ax0.set_xlim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
        ax0.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_xlim(-1.1*r_co.max(), 1.1*r_co.max())
        ax1.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
        ax2.set_xlabel('R [mm]')
        ax2.set_ylabel('Z [mm]')
        ax2.set_xlim(0, 1.1*r_co.max())
        ax2.set_ylim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
        ax3.set_xlabel('R [mm]')
        ax3.set_ylabel('Phi [mm]')
        ax3.set_xlim(0, 1.1*r_co.max())
        ax3.set_ylim(-np.pi, np.pi)


        #plot points
        r_co[X[:,1] < 0] *= -1
        ax0.scatter(z_co, r_co, s=points, c='k')
        ax0.plot([z_co[index[0]], z_co[index[1]]],
                 [r_co[index[0]], r_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax0.plot([z_co[true_index[0]], z_co[true_index[1]]],
                 [r_co[true_index[0]], r_co[true_index[1]]],
                 '-', c='black', linewidth=width2)
        r_co[X[:,1] < 0] *= -1

        ax1.scatter(x_co, y_co, s=points, c='k')
        ax1.plot([x_co[index[0]], x_co[index[1]]],
                 [y_co[index[0]], y_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax1.plot([x_co[true_index[0]], x_co[true_index[1]]],
                 [y_co[true_index[0]], y_co[true_index[1]]],
                 '-', c='black', linewidth=width2)

        ax2.scatter(r_co, z_co, s=points, c='k')
        ax2.plot([r_co[index[0]], r_co[index[1]]],
                 [z_co[index[0]], z_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax2.plot([r_co[true_index[0]], r_co[true_index[1]]],
                 [z_co[true_index[0]], z_co[true_index[1]]],
                 '-', c='black', linewidth=width2)

        ax3.scatter(r_co, phi_co, s=points, c='k')
        ax3.plot([r_co[index[0]], r_co[index[1]]],
                 [phi_co[index[0]], phi_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax3.plot([r_co[true_index[0]], r_co[true_index[1]]],
                 [phi_co[true_index[0]], phi_co[true_index[1]]],
                 '-', c='black', linewidth=width2)


        fig0_name = self.processed_files[idx].split('.')[0] + '_zr_signed.png'
        fig1_name = self.processed_files[idx].split('.')[0] + '_xy.png'
        fig2_name = self.processed_files[idx].split('.')[0] + '_rz.png'
        fig3_name = self.processed_files[idx].split('.')[0] + '_rphi.png'
        fig0.savefig(fig0_name, dpi=dpi)
        fig1.savefig(fig1_name, dpi=dpi)
        fig2.savefig(fig2_name, dpi=dpi)
        fig3.savefig(fig3_name, dpi=dpi)

        fig0_name = self.processed_files[idx].split('.')[0] + '_zr_signed.pdf'
        fig1_name = self.processed_files[idx].split('.')[0] + '_xy.pdf'
        fig2_name = self.processed_files[idx].split('.')[0] + '_rz.pdf'
        fig3_name = self.processed_files[idx].split('.')[0] + '_rphi.pdf'
        fig0.savefig(fig0_name, dpi=dpi)
        fig1.savefig(fig1_name, dpi=dpi)
        fig2.savefig(fig2_name, dpi=dpi)
        fig3.savefig(fig3_name, dpi=dpi)


    def build_tracks(self, hits, particles, truth):
        # print('Building Tracks')
        # valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id']]
                .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'particle_id']]
                .merge(particles[['particle_id', 'px', 'py', 'pz']], on='particle_id'))

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        r = torch.from_numpy(np.sqrt(hits['x'].values**2 + hits['y'].values**2))
        phi = torch.from_numpy(np.arctan2(hits['y'].values, hits['x'].values))
        z = torch.from_numpy(hits['z'].values)
        pt = torch.from_numpy(np.sqrt(hits['px'].values**2 + hits['py'].values**2))
        eta = torch.from_numpy(np.arcsinh(hits['pz'].values/(np.sqrt(hits['px'].values**2 + hits['py'].values**2))))
        particle = torch.from_numpy(hits['particle_id'].values)

        # layer_mask = torch.from_numpy(np.isin(layer, valid_layer))
        pt_mask_lo = pt > self.pt_range[0]
        pt_mask_hi = pt < self.pt_range[1]
        eta_mask_lo = eta > self.eta_range[0]
        eta_mask_hi = eta < self.eta_range[1]

        # mask = layer_mask & pt_mask_lo & pt_mask_hi & eta_mask_lo & eta_mask_hi
        mask = pt_mask_lo & pt_mask_hi & eta_mask_lo & eta_mask_hi

        layer = layer.unique(return_inverse=True)[1]
        r = r[mask]
        phi = phi[mask]
        z = z[mask]
        pos = torch.stack([r, phi, z], 1)
        particle = particle[mask]
        layer = layer[mask]
        pt = pt[mask]
        eta = eta[mask]

        particle, indices = torch.sort(particle)
        particle_i = particle.unique(return_inverse=True)[1]
        pos = pos[indices]
        layer = layer[indices]
        pt = pt[indices]
        eta = eta[indices]

        track_attr = torch.cat((particle[:,None].type(torch.float64), pt[:,None].type(torch.float64), eta[:,None].type(torch.float64)), 1)
        track_attr, hit_count = track_attr.unique(dim=0, return_counts=True)
        track_attr = torch.cat((track_attr, hit_count[:,None].type(torch.float64)),1)

        track_attr_pix = torch.cat((particle[layer<18][:,None].type(torch.float64), pt[layer<18][:,None].type(torch.float64), eta[layer<18][:,None].type(torch.float64)), 1)
        track_attr_pix, hit_count_pix = track_attr_pix.unique(dim=0, return_counts=True)
        track_attr_pix = torch.cat((track_attr_pix, hit_count_pix[:,None].type(torch.float64)),1)

        track_attr_pruned = torch.cat((particle[layer<18][:,None].type(torch.float64), pt[layer<18][:,None].type(torch.float64), eta[layer<18][:,None].type(torch.float64), layer[layer<18][:,None].type(torch.float64)), 1)
        track_attr_pruned = track_attr_pruned.unique(dim=0)
        track_attr_pruned = torch.index_select(track_attr_pruned, 1, torch.tensor([0,1,2]))
        track_attr_pruned, hit_count_pruned = track_attr_pruned.unique(dim=0, return_counts=True)
        track_attr_pruned = torch.cat((track_attr_pruned, hit_count_pruned[:,None].type(torch.float64)),1)

        tracks = torch.empty(0,6, dtype=torch.float64)
        for i in range(particle_i.max()+1):
            track_pos   = pos[particle_i == i]
            track_layer = layer[particle_i == i]
            track_particle = particle[particle_i == i]
            track_particle_i = particle_i[particle_i == i]
            track_layer, indices = torch.sort(track_layer)
            track_pos = track_pos[indices]
            track_layer = track_layer[:, None]
            track_particle = track_particle[:, None]
            track_particle_i = track_particle_i[:, None]
            track = torch.cat((track_pos.type(torch.float64), track_layer.type(torch.float64)), 1)
            track = torch.cat((track, track_particle_i.type(torch.float64)), 1)
            track = torch.cat((track, track_particle.type(torch.float64)), 1)
            tracks = torch.cat((tracks, track), 0)

        return tracks, track_attr, track_attr_pix, track_attr_pruned


    def build_tracks_atlas(self, particles, truth):
        # print('Building Tracks')
        # valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]

        truth.loc[truth['hardware'] == 'STRIP','barrel_endcap'] = truth.loc[truth['hardware'] == 'STRIP','barrel_endcap'] + 100

        hits = (truth[['hit_id', 'x', 'y', 'z', 'barrel_endcap', 'layer_disk', 'particle_id']]
                .merge(particles[['particle_id', 'pt', 'eta']], on='particle_id'))

        layer = torch.from_numpy(20 * hits['barrel_endcap'].values + hits['layer_disk'].values)
        r = torch.from_numpy(np.sqrt(hits['x'].values**2 + hits['y'].values**2))
        phi = torch.from_numpy(np.arctan2(hits['y'].values, hits['x'].values))
        z = torch.from_numpy(hits['z'].values)
        pt = torch.from_numpy(hits['pt'].values/1000)
        eta = torch.from_numpy(hits['eta'].values)
        particle = torch.from_numpy(hits['particle_id'].values)

        # layer_mask = torch.from_numpy(np.isin(layer, valid_layer))
        pt_mask_lo = pt > self.pt_range[0]
        pt_mask_hi = pt < self.pt_range[1]
        eta_mask_lo = eta > self.eta_range[0]
        eta_mask_hi = eta < self.eta_range[1]

        # mask = layer_mask & pt_mask_lo & pt_mask_hi & eta_mask_lo & eta_mask_hi
        mask = pt_mask_lo & pt_mask_hi & eta_mask_lo & eta_mask_hi

        layer = layer.unique(return_inverse=True)[1]
        r = r[mask]
        phi = phi[mask]
        z = z[mask]
        pos = torch.stack([r, phi, z], 1)
        particle = particle[mask]
        layer = layer[mask]
        pt = pt[mask]
        eta = eta[mask]

        particle, indices = torch.sort(particle)
        particle_i = particle.unique(return_inverse=True)[1]
        pos = pos[indices]
        layer = layer[indices]
        pt = pt[indices]
        eta = eta[indices]

        track_attr = torch.cat((particle[:,None].type(torch.float64), pt[:,None].type(torch.float64), eta[:,None].type(torch.float64)), 1)
        track_attr, hit_count = track_attr.unique(dim=0, return_counts=True)
        track_attr = torch.cat((track_attr, hit_count[:,None].type(torch.float64)),1)

        track_attr_pix = torch.cat((particle[layer<15][:,None].type(torch.float64), pt[layer<15][:,None].type(torch.float64), eta[layer<15][:,None].type(torch.float64)), 1)
        track_attr_pix, hit_count_pix = track_attr_pix.unique(dim=0, return_counts=True)
        track_attr_pix = torch.cat((track_attr_pix, hit_count_pix[:,None].type(torch.float64)),1)

        track_attr_pruned = torch.cat((particle[layer<15][:,None].type(torch.float64), pt[layer<15][:,None].type(torch.float64), eta[layer<15][:,None].type(torch.float64), layer[layer<15][:,None].type(torch.float64)), 1)
        track_attr_pruned = track_attr_pruned.unique(dim=0)
        track_attr_pruned = torch.index_select(track_attr_pruned, 1, torch.tensor([0,1,2]))
        track_attr_pruned, hit_count_pruned = track_attr_pruned.unique(dim=0, return_counts=True)
        track_attr_pruned = torch.cat((track_attr_pruned, hit_count_pruned[:,None].type(torch.float64)),1)

        tracks = torch.empty(0,6, dtype=torch.float64)
        for i in range(particle_i.max()+1):
            track_pos   = pos[particle_i == i]
            track_layer = layer[particle_i == i]
            track_particle = particle[particle_i == i]
            track_particle_i = particle_i[particle_i == i]
            track_layer, indices = torch.sort(track_layer)
            track_pos = track_pos[indices]
            track_layer = track_layer[:, None]
            track_particle = track_particle[:, None]
            track_particle_i = track_particle_i[:, None]
            track = torch.cat((track_pos.type(torch.float64), track_layer.type(torch.float64)), 1)
            track = torch.cat((track, track_particle_i.type(torch.float64)), 1)
            track = torch.cat((track, track_particle.type(torch.float64)), 1)
            tracks = torch.cat((tracks, track), 0)

        return tracks, track_attr, track_attr_pix, track_attr_pruned


    def files_exist(self, files):
        return len(files) != 0 and all([osp.exists(f) for f in files])


    def shuffle(self):
        random.shuffle(self.processed_files)


    def sort(self):
        self.processed_files.sort()


    def build_accumulator(self, pos):
        # print("build_accumulator")
        accumulator0 = torch.zeros(self.accum0_b[2] , self.accum0_m[2], dtype=torch.long)
        accumulator1 = torch.zeros(self.accum1_b[2] , self.accum1_m[2], dtype=torch.long)

        for i in tqdm(range(pos.shape[0])):
        # for i in range(pos.shape[0]):
            self.cast_vote(accumulator0, pos[i,0], pos[i,2], 0) #R-Z   Plane
            self.cast_vote(accumulator1, pos[i,0], pos[i,1], 1) #R-Phi Plane

        # accumulator = torch.stack([self.cast_vote(pos[i,0], pos[i,2]) for i in range(pos.shape[0])], dim=0).sum(dim=0)
        # self.draw_accumulator(accumulator0, accumulator1)
        return accumulator0, accumulator1


    def draw_accumulator(self, accumulator0, accumulator1):
        fig0, ax0 = plt.subplots()
        img0 = ax0.imshow(accumulator0, cmap="hot", extent=[self.accum0_m[0],self.accum0_m[1],self.accum0_b[0],self.accum0_b[1]], aspect="auto")
        ax0.set_xlabel(r"$m$")
        ax0.set_ylabel(r"$b$")
        fig0.colorbar(img0)
        plt.title("Hough Transform Accumulator (RZ)")
        fig0.savefig("accumulator_rz.pdf", dpi=600)

        fig1, ax1 = plt.subplots()
        img1 = ax1.imshow(accumulator1, cmap="hot", extent=[self.accum1_m[0],self.accum1_m[1],self.accum1_b[0],self.accum1_b[1]], aspect="auto")
        ax1.set_xlabel(r"$m$")
        ax1.set_ylabel(r"$b$")
        fig1.colorbar(img1)
        plt.title("Hough Transform Accumulator (RPhi)")
        fig1.savefig("accumulator_rphi.pdf", dpi=600)



    def cast_vote(self, accumulator, x_co, y_co, switch=0):
    # def cast_vote(self, x_co, y_co):
        # accumulator = torch.zeros(self.accum_b[2] , self.accum_m[2], dtype=torch.long)

        # print(switch)
        # print(accumulator)

        if switch == 0:
            b_min = self.accum0_b[0]
            b_max = self.accum0_b[1]
            b_bin = self.accum0_b[2]
            m_min = self.accum0_m[0]
            m_max = self.accum0_m[1]
            m_bin = self.accum0_m[2]
        elif switch == 1:
            b_min = self.accum1_b[0]
            b_max = self.accum1_b[1]
            b_bin = self.accum1_b[2]
            m_min = self.accum1_m[0]
            m_max = self.accum1_m[1]
            m_bin = self.accum1_m[2]

        m_step = (m_max - m_min) / m_bin
        b_step = (b_max - b_min) / b_bin
        m_lo = torch.tensor([m_min +  i   *m_step for i in range(m_bin)])
        m_hi = torch.tensor([m_min + (i+1)*m_step for i in range(m_bin)])
        b_lo = y_co - m_lo * x_co
        b_hi = y_co - m_hi * x_co
        j_lo = torch.floor(b_bin * (b_lo - b_max) / (b_min - b_max))
        j_hi = torch.floor(b_bin * (b_hi - b_max) / (b_min - b_max))
        j_min = torch.min(j_lo, j_hi)
        j_max = torch.max(j_lo, j_hi)

        for i in range(m_bin):
            min = int(j_min[i].item())
            max = int(j_max[i].item())

            if min < 0 and max >= 0 and max < b_bin:
                accumulator[:,i][:max+1] = accumulator[:,i][:max+1] + 1
                # accumulator[:,i][:max+1] = 1
            elif min >= 0 and max < b_bin:
                accumulator[:,i][min:max+1] = accumulator[:,i][min:max+1] + 1
                # accumulator[:,i][min:max+1] = 1
            elif min >= 0 and min < b_bin and max >= b_bin:
                accumulator[:,i][min:] = accumulator[:,i][min:] + 1
                # accumulator[:,i][min:] = 1
            elif min < 0 and max >= b_bin:
                accumulator[:,i] = accumulator[:,i] + 1
                # accumulator[:,i] = 1

        return accumulator
        # for i in range(self.accum_m[2]):
        #     m_lo = self.accum_m[0] +  i   *m_step
        #     m_hi = self.accum_m[0] + (i+1)*m_step
        #     b_lo = y_co - m_lo * x_co
        #     b_hi = y_co - m_hi * x_co
        #
        #     if ((b_lo >= b_min and b_lo < b_max) or (b_hi >= b_min and b_hi < b_max)):
        #         j_lo = torch.floor(b_bin * (b_lo - b_max) / (b_min - b_max))
        #         j_hi = torch.floor(b_bin * (b_hi - b_max) / (b_min - b_max))
        #         if j_lo > j_hi:
        #             a = j_lo
        #             j_lo = j_hi
        #             j_hi = a
        #
        #         for k in range(torch.tensor(j_hi-j_lo+1, dtype=torch.int64)):
        #             j = torch.tensor(j_lo + k, dtype=torch.int64)
        #             if (j >= 0 and j < b_bin):
        #                 accumulator[j,i] = accumulator[j,i] + 1


    # def extract_votes(self, accumulator0, accumulator1, pos, edge_index):
    def extract_votes(self, pos, edge_index):
        # print("extract_votes")

        r_in = pos[edge_index[0], 0]
        p_in = pos[edge_index[0], 1]
        z_in = pos[edge_index[0], 2]
        r_ot = pos[edge_index[1], 0]
        p_ot = pos[edge_index[1], 1]
        z_ot = pos[edge_index[1], 2]

        dr = r_ot-r_in
        dp = p_ot-p_in
        dz = z_ot-z_in
        dp[dp >  np.pi] -= 2 * np.pi
        dp[dp < -np.pi] += 2 * np.pi

        m0 = dz/dr
        b0 = z_in - m0*r_in
        m1 = dp/dr
        b1 = p_in - m1*r_in

        return torch.stack([m0, b0, m1, b1], 1)


        # i0 = torch.floor(self.accum0_m[2] * (m0 - self.accum0_m[0]) / (self.accum0_m[1] - self.accum0_m[0]))
        # j0 = torch.floor(self.accum0_b[2] * (b0 - self.accum0_b[1]) / (self.accum0_b[0] - self.accum0_b[1]))
        # i1 = torch.floor(self.accum1_m[2] * (m1 - self.accum1_m[0]) / (self.accum1_m[1] - self.accum1_m[0]))
        # j1 = torch.floor(self.accum1_b[2] * (b1 - self.accum1_b[1]) / (self.accum1_b[0] - self.accum1_b[1]))
        #
        # # plt.hist(j0, 100, [-100, 2100])
        # # plt.savefig('debug_hist.pdf', dpi=600)
        #
        # edge_votes = torch.empty(2,0, dtype=torch.long)
        # for i in range(edge_index.shape[1]):
        #     i0_int = int(i0[i].item())
        #     j0_int = int(j0[i].item())
        #     i1_int = int(i1[i].item())
        #     j1_int = int(j1[i].item())
        #
        #     if (i0_int >= 0 and i0_int < self.accum0_m[2] and j0_int >= 0 and j0_int < self.accum0_b[2]):
        #         vote0 = accumulator0[j0_int, i0_int]
        #     else:
        #         vote0 = 0
        #
        #     if (i1_int >= 0 and i1_int < self.accum1_m[2] and j1_int >= 0 and j1_int < self.accum1_b[2]):
        #         vote1 = accumulator1[j1_int,i1_int]
        #     else:
        #         vote1 = 0
        #
        #     votes = torch.tensor([[vote0], [vote1]])
        #     edge_votes = torch.cat((edge_votes, votes), 1)
        #
        # return edge_votes.T
