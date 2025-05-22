import os
from typing import Union, List, Tuple, Optional, Callable

import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, download_url, DataLoader
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import rdmolops, Draw

class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ba2motif_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save(self.collate(data_list), self.processed_paths[0])


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        data_list.append(Data(x=torch.from_numpy(node_features[graph_idx]).float(),
                              edge_index=dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0],
                              y=torch.from_numpy(np.where(graph_labels[graph_idx])[0])))
    return data_list


class MUTAGDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise NotImplementedError

    def process(self):
        with open(self.raw_dir + '/Mutagenicity.pkl', 'rb') as fin:
            _, original_features, original_labels = pickle.load(fin)

        edge_lists, graph_labels, edge_label_lists, node_type_lists, edge_type_lists = self.get_graph_data()  # node type, not node label

        data_list = []
        for i in range(original_labels.shape[0]):
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(original_features[i][:num_nodes]).float()
            assert original_features[i][num_nodes:].sum() == 0
            edge_label = torch.tensor(edge_label_lists[i]).float()
            if y.item() != 0:
                edge_label = torch.zeros_like(edge_label).float()

            node_label = torch.zeros(x.shape[0])
            signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
            if y.item() == 0:
                node_label[signal_nodes] = 1

            if len(signal_nodes) != 0:
                node_type = torch.tensor(node_type_lists[i])
                node_type = set(node_type[signal_nodes].tolist())
                assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

            '''gsat filter some samples while pgexplainer not'''
            if y.item() == 0 and len(signal_nodes) == 0:
                continue

            # print(x.shape)
            # print(y)
            # print(edge_index)
            # print(node_label.shape)
            # print(edge_label.shape)

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label,
                                  node_type=torch.tensor(node_type_lists[i]),
                                  edge_type=torch.tensor(edge_type_lists[i])))
        # print(len(data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = self.raw_dir + '/Mutagenicity_'

        file_edges = pri + 'A.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        try:
            edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use edge label 0')
            edge_labels = np.zeros(edges.shape[0]).astype(np.int32)

        edge_types = np.loadtxt(os.path.join(self.raw_dir, 'Mutagenicity_edge_labels.txt'), delimiter=',').astype(
            np.int32)

        print(len(edge_types))

        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)

        try:
            node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)
        except Exception as e:
            print(e)
            print('use node label 0')
            node_labels = np.zeros(graph_indicator.shape[0]).astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i + 1)
            node2graph[i + 1] = len(starts) - 1
        # print(starts)
        # print(node2graph)
        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_type_lists = []  # add
        edge_list = []
        edge_label_list = []
        edge_type_list = []  # add
        for (s, t), l, et in list(zip(edges, edge_labels, edge_types)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_type_lists.append(edge_type_list)  # add
                edge_list = []
                edge_label_list = []
                edge_type_list = []  # add
                graphid = gid
            start = starts[gid]
            edge_list.append((s - start, t - start))
            edge_label_list.append(l)
            edge_type_list.append(et + 1)  # add

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)
        edge_type_lists.append(edge_type_list)  # add

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i + 1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists, edge_type_lists


class BenzeneDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['benzene.npz']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data = np.load(self.raw_paths[0], allow_pickle=True)
        att, X, y, df = data['attr'], data['X'], data['y'], data['smiles']
        X = X[0]
        ylist = [y[i] for i in range(y.shape[0])]
        data_list = []
        for i in range(len(X)):
            x = torch.from_numpy(X[i]['nodes'])
            edge_attr = torch.from_numpy(X[i]['edges'])
            y = torch.tensor([ylist[i]], dtype=torch.float)

            # get edge_index:
            e1 = torch.from_numpy(X[i]['receivers']).long()
            e2 = torch.from_numpy(X[i]['senders']).long()

            edge_index = torch.stack([e1, e2])

            # get ground-truth explanation:
            node_imp = torch.from_numpy(att[i][0]['nodes']).float()

            assert att[i][0]['n_edge'] == X[i]['n_edge'], 'Num: {}, Edges different sizes'.format(i)
            assert node_imp.shape[0] == x.shape[0], 'Num: {}, Shapes: {} vs. {}'.format(i, node_imp.shape[0],
                                                                                        x.shape[0]) \
                                                    + '\nExp: {} \nReal:{}'.format(att[i][0], X[i])

            node_numbers = node_imp.nonzero(as_tuple=True)[0]
            edge_imp = torch.zeros((edge_index.shape[1],))
            for e in range(edge_index.shape[1]):
                edge_imp[e] = (edge_index[0, e] in node_numbers) and (edge_index[1, e] in node_numbers)

            node_imp = node_imp.squeeze()

            # print(x.shape)
            # print(y.shape)
            # print(edge_index)
            # print(node_imp.shape)
            # print(edge_imp.shape)

            data_i = Data(
                x=x,
                y=y,
                edge_index=edge_index,
                # node_label=node_imp,
                edge_label=edge_imp,
                smile=df[i][1]
            )
            data_list.append(data_i)
        print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MRDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['attributions.npz']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        type_to_element = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
        data = np.load(self.raw_paths[0], allow_pickle=True)
        item_list = data['attributions']
        data_list = []
        for item in item_list:
            smiles = item['SMILES']
            mol = Chem.MolFromSmiles(smiles)

            num_atoms = mol.GetNumAtoms()
            print(num_atoms)
            element_to_idx = {element: i for i, element in enumerate(type_to_element)}
            x = np.zeros((num_atoms, len(type_to_element)), dtype=np.float32)
            for i in range(num_atoms):
                atom = mol.GetAtomWithIdx(i)
                element = atom.GetSymbol().capitalize()
                if element in element_to_idx:
                    idx = element_to_idx[element]
                    x[i, idx] = 1
                else:
                    print(f"Warning: Atom type {element} not in type_to_element. Ignoring.")
            x = torch.from_numpy(x)

            y = torch.tensor([[item['label']]], dtype=torch.float)

            node_labels = item['node_atts']

            adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol) 
            edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)

            node_labels = torch.tensor(node_labels, dtype=torch.float32)
            edge_label = (node_labels[edge_index[0]] * node_labels[edge_index[1]]).to(torch.long)

            data_i = Data(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_label=edge_label,
                smiles=smiles
            )
            data_list.append(data_i)
        print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # att, X, y, df = data['attr'], data['X'], data['y'], data['smiles']
        # X = X[0]
        # ylist = [y[i] for i in range(y.shape[0])]
        # data_list = []
        # for i in range(len(X)):
        #     x = torch.from_numpy(X[i]['nodes'])
        #     edge_attr = torch.from_numpy(X[i]['edges'])
        #     y = torch.tensor([ylist[i]], dtype=torch.float)
        #
        #     # get edge_index:
        #     e1 = torch.from_numpy(X[i]['receivers']).long()
        #     e2 = torch.from_numpy(X[i]['senders']).long()
        #
        #     edge_index = torch.stack([e1, e2])
        #
        #     # get ground-truth explanation:
        #     node_imp = torch.from_numpy(att[i][0]['nodes']).float()
        #
        #     assert att[i][0]['n_edge'] == X[i]['n_edge'], 'Num: {}, Edges different sizes'.format(i)
        #     assert node_imp.shape[0] == x.shape[0], 'Num: {}, Shapes: {} vs. {}'.format(i, node_imp.shape[0],
        #                                                                                 x.shape[0]) \
        #                                             + '\nExp: {} \nReal:{}'.format(att[i][0], X[i])
        #
        #     node_numbers = node_imp.nonzero(as_tuple=True)[0]
        #     edge_imp = torch.zeros((edge_index.shape[1],))
        #     for e in range(edge_index.shape[1]):
        #         edge_imp[e] = (edge_index[0, e] in node_numbers) and (edge_index[1, e] in node_numbers)
        #
        #     node_imp = node_imp.squeeze()
        #
        #     # print(x.shape)
        #     # print(y.shape)
        #     # print(edge_index)
        #     # print(node_imp.shape)
        #     # print(edge_imp.shape)
        #
        #     data_i = Data(
        #         x=x,
        #         y=y,
        #         edge_index=edge_index,
        #         # node_label=node_imp,
        #         edge_label=edge_imp,
        #         smile=df[i][1]
        #     )
        #     data_list.append(data_i)
        # print(data_list)
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])

def get_dataset(dataset_dir, dataset_name, data_split_ratio):
    np.random.seed(2025)
    dataset = None
    splited_dataset = dict()
    if dataset_name.lower() == 'BA_2Motifs'.lower():
        dataset = BA2MotifDataset(root=dataset_dir, name=dataset_name)
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        n_train, n_valid = int(data_split_ratio[0] * len(idx)), int(data_split_ratio[1] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]
        splited_dataset['all'] = dataset
        splited_dataset['train'] = dataset[train_idx]
        splited_dataset['valid'] = dataset[valid_idx]
        splited_dataset['test'] = dataset[test_idx]
    elif dataset_name.lower() == 'MUTAG'.lower():
        dataset = MUTAGDataset(root=os.path.join(dataset_dir, dataset_name))
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        n_train, n_valid = int(data_split_ratio[0] * len(idx)), int(data_split_ratio[1] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]
        splited_dataset['all'] = dataset
        splited_dataset['train'] = dataset[train_idx]
        splited_dataset['valid'] = dataset[valid_idx]
        splited_dataset['test'] = dataset[test_idx]
    elif dataset_name.lower() == 'BENZENE'.lower():
        dataset = BenzeneDataset(root=os.path.join(dataset_dir, dataset_name))
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        n_train, n_valid = int(data_split_ratio[0] * len(idx)), int(data_split_ratio[1] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]
        splited_dataset['all'] = dataset
        splited_dataset['train'] = dataset[train_idx]
        splited_dataset['valid'] = dataset[valid_idx]
        splited_dataset['test'] = dataset[test_idx]
    elif dataset_name.lower() == 'MR'.lower():
        dataset = MRDataset(root=os.path.join(dataset_dir, dataset_name))
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        n_train, n_valid = int(data_split_ratio[0] * len(idx)), int(data_split_ratio[1] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx = idx[n_train + n_valid:]
        splited_dataset['all'] = dataset
        splited_dataset['train'] = dataset[train_idx]
        splited_dataset['valid'] = dataset[valid_idx]
        splited_dataset['test'] = dataset[test_idx]

    return splited_dataset
