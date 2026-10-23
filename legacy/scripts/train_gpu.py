# General libraries
import argparse
from datetime import timedelta
import json
import multi
import os
import pandas as pd
import torch
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.conv import *

from tqdm import tqdm
from typing import Dict, Optional

# GraphToolbox
from graphtoolbox.data.dataset import *
from graphtoolbox.data.preprocessing import *
from graphtoolbox.utils.helper_functions import *
from graphtoolbox.models.gnn import *

num_epochs = 500
patience = 300
OUT_CHANNELS = 48
DATASET = 'weave'

parser = argparse.ArgumentParser()
parser.add_argument('--data_number', type=int, required=True)
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as f:
    config_dict = json.load(f)
conv_classes = [GCNConv, GraphSAGE, GATConv, GATv2Conv, TransformerConv, ChebConv, TAGConv, APPNP]
# conv_classes = [ChebConv]
config_dict2 = {conv_class: config_dict[conv_class.__name__] for conv_class in conv_classes}
job_conv = list(config_dict2.keys())[args.data_number % len(conv_classes)]
job = list(config_dict2.values())[args.data_number % len(conv_classes)]
conv_class = job_conv
params = job["params"]
adj_matrix = job["adj_matrix"]
out_channels = 48
save_dir = os.path.join(f'results_{DATASET}', conv_class.__name__)
os.makedirs(save_dir, exist_ok=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics, preds = multi.worker_train_gnn(conv_class, num_epochs, patience, params, device, graph_dataset_train, graph_dataset_val, graph_dataset_test)
torch.save(preds, os.path.join(save_dir, f'{conv_class.__name__}_{args.data_number}.pt'))
print(f"[RESULT] Job {args.data_number} | {metrics}")
with open(f'logs/output_{DATASET}.txt', 'a') as f:
    f.write(f"[RESULT] Job {args.data_number} | {adj_matrix} | {metrics}\n")