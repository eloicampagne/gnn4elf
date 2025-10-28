# General libraries
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import multi
import pandas as pd
from torch_geometric.nn.conv import *
from torch_geometric.nn import GraphSAGE
from tqdm import tqdm

# GraphToolbox
from graphtoolbox.data.dataset import *
from graphtoolbox.data.preprocessing import *
from graphtoolbox.utils.helper_functions import *
from graphtoolbox.models.gnn import *

# conv_classes = [GATConv, GATv2Conv, TransformerConv, ChebConv, TAGConv, APPNP]
conv_classes = [GCNConv, GraphSAGE]
num_epochs = 300
n_trials = 200
DATASET = 'weave'
OUT_CHANNELS = 48

if DATASET == 'rfrance':
    PATH_TRAIN = './data/rfrance/train2.csv'
    PATH_TEST = './data/rfrance/test2.csv'
    GRAPH_FOLDER = './graph_representations_rfrance'
    data_kwargs = {
        'node_var': 'Region',
        'dummies': ['Instant', 'JourSemaine', 'DayType', 'offset'],
        'day_inf_train': '2017-01-01',
        'day_sup_train': '2018-01-01',
        'day_inf_val': '2018-01-01',
        'day_sup_val': '2018-12-31',
        'day_inf_test': '2019-01-01',
        'day_sup_test': '2019-12-31'
    }

    dataset_kwargs = {
        'batch_size': 32,
        'adj_matrix': 'dtw',
        'features_base': ['temp', 'nebu', 'wind', 'tempMax', 'tempMin', 'Posan', 'Instant', 'JourSemaine', 'JourFerie', 'offset', 'DayType', 'Weekend', 'temp_liss_fort', 'temp_liss_faible'],
        'target_base': 'load',
    }

    data = DataClass(path_train=PATH_TRAIN, 
                     path_test=PATH_TEST, 
                     data_kwargs=data_kwargs,
                     folder_config='.')

    graph_dataset_train = GraphDataset(data=data, period='train', 
                                       graph_folder=GRAPH_FOLDER,
                                       dataset_kwargs=dataset_kwargs,
                                       out_channels=OUT_CHANNELS)
    graph_dataset_val = GraphDataset(data=data, period='val', 
                                     scalers_feat=graph_dataset_train.scalers_feat, 
                                     scalers_target=graph_dataset_train.scalers_target,
                                     graph_folder=GRAPH_FOLDER,
                                     dataset_kwargs=dataset_kwargs,
                                     out_channels=OUT_CHANNELS)
    graph_dataset_test = GraphDataset(data=data, period='test',
                                      scalers_feat=graph_dataset_train.scalers_feat, 
                                      scalers_target=graph_dataset_train.scalers_target,
                                      graph_folder=GRAPH_FOLDER,
                                      dataset_kwargs=dataset_kwargs,
                                      out_channels=OUT_CHANNELS) 
else:
    PATH_TRAIN='./data/weave/train_weave.csv'
    PATH_TEST='./data/weave/test_weave.csv'
    GRAPH_FOLDER = './graph_representations_weave'
    data_kwargs = {
        'node_var': 'id_unique',
        'dummies': ['instant', 'id_uniqueInt',],
        'day_inf_train': '2024-02-13',
        'day_sup_train': '2024-02-23',
        'day_inf_val': '2024-02-23',
        'day_sup_val': '2024-02-25',
        'day_inf_test': '2024-02-25',
        'day_sup_test': '2024-02-29'
    }

    dataset_kwargs = {
        'batch_size': 32,
        'adj_matrix': 'dtw',
        'features_base': ['instant', 'id_uniqueInt', 'weekday'] + [f'consumption_l{t}' for t in range(1, 48+1)],
        'target_base': 'consumption',
    }


    data = DataClass(path_train=PATH_TRAIN, 
                     path_test=PATH_TEST, 
                     data_kwargs=data_kwargs,
                     folder_config='.')

    graph_dataset_train = GraphDataset(data=data, period='train', 
                                       graph_folder=GRAPH_FOLDER,
                                       dataset_kwargs=dataset_kwargs,
                                       out_channels=OUT_CHANNELS)
    graph_dataset_val = GraphDataset(data=data, period='val', 
                                     scalers_feat=graph_dataset_train.scalers_feat, 
                                     scalers_target=graph_dataset_train.scalers_target,
                                     graph_folder=GRAPH_FOLDER,
                                     dataset_kwargs=dataset_kwargs,
                                     out_channels=OUT_CHANNELS)
    graph_dataset_test = GraphDataset(data=data, period='test',
                                      scalers_feat=graph_dataset_train.scalers_feat, 
                                      scalers_target=graph_dataset_train.scalers_target,
                                      graph_folder=GRAPH_FOLDER,
                                      dataset_kwargs=dataset_kwargs,
                                      out_channels=OUT_CHANNELS)

optim_base = {
    'num_layers': (1, 5),
    'hidden_channels': (32, 512),
    'lr': (1e-5, 1e-1)
}
optim_kwargs_conv = {
    'GCNConv': optim_base,
    'GraphSAGE': optim_base,
    'GATConv': optim_base | {'heads': (1, 5)}, 
    'GATv2Conv': optim_base | {'heads': (1, 5)}, 
    'TransformerConv': optim_base | {'heads': (1, 5)}, 
    'ChebConv': optim_base | {'K': (1, 10)},
    'TAGConv': optim_base | {'K': (1, 10)}, 
    'APPNP': optim_base | {'K': (1, 10), 'alpha': (.5, 1.)}
}

def main():
    max_workers = min(mp.cpu_count(), len(conv_classes))
    tasks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for conv_class in conv_classes:
            task = executor.submit(multi.worker_optuna, conv_class, optim_kwargs_conv[conv_class.__name__], num_epochs, n_trials, graph_dataset_train, graph_dataset_val, DATASET)
            tasks.append(task)
 
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing models"):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed with error: {e}")
 
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()