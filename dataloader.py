import numpy as np
from torch_geometric.loader import DataLoader


def get_dataloader(dataset, batch_size):
    batch_size = batch_size
    dataloader = dict()
    dataloader['train'] = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    dataloader['valid'] = DataLoader(dataset['valid'], batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(dataset['test'], batch_size=1, shuffle=False)
    dataloader['all_by_sample'] = DataLoader(dataset['all'], batch_size=1, shuffle=False)
    dataloader['test_by_sample'] = DataLoader(dataset['test'], batch_size=1, shuffle=False)

    # num_graphs = len(dataloader['all_by_sample'])
    # print(num_graphs)
    # edge_list = []
    # node_list = []
    # for data in dataloader['all_by_sample']:
    #     node_list.append(data.x.size(0))
    #     edge_list.append(data.edge_label.size(0))
    # print(np.mean(node_list), np.mean(edge_list)/2)
    # exit()

    return dataloader
