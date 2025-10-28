# General libraries
import argparse
from datetime import timedelta
import json
import numpy as np
from model import FF
import multi
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.nn.conv import *

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

if DATASET == 'rfrance':
    nodes = ['Auvergne_Rhone_Alpes',
            'Bourgogne_Franche_Comte',
            'Bretagne',
            'Centre_Val_de_Loire', 
            'Grand_Est',
            'Hauts_de_France',      
            'Ile_de_France',
            'Normandie',
            'Nouvelle_Aquitaine',
            'Occitanie',      
            'Pays_de_la_Loire',
            'Provence_Alpes_Cote_d_Azur']
else:
    nodes = ['01_010-PLOWMAN TOWER WESTLAND T1', '01_040-COURTLAND ROAD',
       '01_040-FOLLY BRIDGE', '01_040-NEW CROSS ROAD',
       '01_060-VIOLET WAY', '01_100-MAIN ROAD TOOT BALDON',
       '01_160-BROAD CLOSE TILBURY FARM', '01_180-OLD HIGH STREET RMU 1',
       '02_030-PEGASUS ROAD', '02_060-VIOLET WAY',
       '02_160-BROAD CLOSE TILBURY FARM', '02_180-OLD HIGH STREET RMU 1',
       '03_020-STOKE PLACE', '03_030-PEGASUS ROAD',
       '03_040-COURTLAND ROAD', '03_040-NEW CROSS ROAD',
       '03_060-VIOLET WAY', '03_160-BROAD CLOSE TILBURY FARM',
       '03_180-OLD HIGH STREET RMU 1', '04_010-PLOWMAN TOWER WESTLAND T1',
       '04_030-PEGASUS ROAD', '04_040-COURTLAND ROAD',
       '04_060-VIOLET WAY', '04_080-ASHMOLEON MUSEUM',
       '04_180-OLD HIGH STREET RMU 1', '05_010-PLOWMAN TOWER WESTLAND T1',
       '05_040-COURTLAND ROAD']
    #    '06-06 st andrews road n/b_020-STOKE PLACE']

config_dict2 = {node: config_dict[node] for node in nodes}
job_node = list(config_dict2.keys())[args.data_number % len(nodes)]
job = list(config_dict2.values())[args.data_number % len(nodes)]

params = job["params"]
save_dir = os.path.join(f'results_{DATASET}', 'FF', job_node.replace("/", ""))
os.makedirs(save_dir, exist_ok=True)

def create_sequences(X, y, mode='train', horizon=OUT_CHANNELS):
    if mode == 'train':
        X_seqs, y_seqs = [], []
        for i in range(len(X)-horizon+1):
            X_seqs.append(X[i])
            y_seqs.append(y[i:i+horizon])
    else:   
        X_seqs, y_seqs = [], []
        for i in range(0, len(X)-horizon+1, horizon):
            X_seqs.append(X[i])
            y_seqs.append(y[i:i+horizon])
    return np.stack(X_seqs), np.stack(y_seqs)

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
    
features = graph_dataset_train.features
target = graph_dataset_train.target
forecast_horizon = OUT_CHANNELS

nodes = data.nodes
X_train_node = {node: [] for node in nodes}
y_train_node = {node: [] for node in nodes}
X_val_node = {node: [] for node in nodes}
y_val_node = {node: [] for node in nodes}
X_test_node = {node: [] for node in nodes}
y_test_node = {node: [] for node in nodes}

X_train_all = []
y_train_all = []
for node in nodes:
    train_r = data.df_train[data.df_train[data.node_var] == node].reset_index(drop=True)[features].values
    y_train_r = data.df_train[data.df_train[data.node_var] == node].reset_index(drop=True)[[target]].values
    X_train_all.append(train_r)
    y_train_all.append(y_train_r)

X_train_all = np.concatenate(X_train_all, axis=0)
y_train_all = np.concatenate(y_train_all, axis=0)

global_scalerX = MinMaxScaler().fit(X_train_all)
global_scalerY = MinMaxScaler().fit(y_train_all)

for node in data.nodes:
    train_r = data.df_train[data.df_train[data.node_var] == node].reset_index(drop=True)[features].values
    val_r   = data.df_val[data.df_val[data.node_var] == node].reset_index(drop=True)[features].values
    test_r  = data.df_test[data.df_test[data.node_var] == node].reset_index(drop=True)[features].values

    y_train_r = data.df_train[data.df_train[data.node_var] == node].reset_index(drop=True)[[target]].values
    y_val_r   = data.df_val[data.df_val[data.node_var] == node].reset_index(drop=True)[[target]].values
    y_test_r  = data.df_test[data.df_test[data.node_var] == node].reset_index(drop=True)[[target]].values

    train_r_scaled = global_scalerX.transform(train_r)
    val_r_scaled   = global_scalerX.transform(val_r)
    test_r_scaled  = global_scalerX.transform(test_r)
    y_train_r_scaled = global_scalerY.transform(y_train_r)
    y_val_r_scaled   = global_scalerY.transform(y_val_r)
    y_test_r_scaled  = global_scalerY.transform(y_test_r)

    X_train_seq, y_train_seq = create_sequences(train_r_scaled, y_train_r_scaled, mode='train', horizon=forecast_horizon)
    X_val_seq, y_val_seq     = create_sequences(val_r_scaled, y_val_r_scaled, mode='test', horizon=forecast_horizon)
    X_test_seq, y_test_seq   = create_sequences(test_r_scaled, y_test_r_scaled, mode='test', horizon=forecast_horizon)

    X_train_node[node] = torch.tensor(X_train_seq, dtype=torch.float32).reshape(-1, len(features))
    y_train_node[node] = torch.tensor(y_train_seq, dtype=torch.float32).reshape(-1, forecast_horizon)
    X_val_node[node]   = torch.tensor(X_val_seq, dtype=torch.float32).reshape(-1, len(features))
    y_val_node[node]   = torch.tensor(y_val_seq, dtype=torch.float32).reshape(-1, forecast_horizon)
    X_test_node[node]  = torch.tensor(X_test_seq, dtype=torch.float32).reshape(-1, len(features))
    y_test_node[node]  = torch.tensor(y_test_seq, dtype=torch.float32).reshape(-1, forecast_horizon)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics, preds = multi.worker_train_ff(job_node, num_epochs, patience, params, device, X_train_node, y_train_node, X_val_node, y_val_node, X_test_node, y_test_node, global_scalerY)
torch.save(preds, os.path.join(save_dir, f'FF_{job_node.replace("/", "")}_{args.data_number}.pt'))
print(f"[RESULT] Job {args.data_number} | {metrics}")
with open(f'logs/output_{DATASET}.txt', 'a') as f:
    f.write(f"[RESULT] Job {args.data_number} | {metrics}\n")