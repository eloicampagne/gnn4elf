import datetime
import logging
from model import *
import numpy as np
import optuna
from optuna_dashboard import run_server
import os
import shutil
import sys
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric import seed_everything
from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from typing import List, Tuple, Union

from graphtoolbox.data.dataset import GraphDataset
from graphtoolbox.models.gnn import *
import graphtoolbox.training.metrics 
from graphtoolbox.training.trainer import Trainer
from graphtoolbox.utils.helper_functions import *
from graphtoolbox.utils.visualizations import *

OUT_CHANNELS = 48

class Optimizer():
    def __init__(self, model, dataset_train: GraphDataset, dataset_val: GraphDataset, optim_kwargs: Dict = None, **kwargs):
        self.model_class = model
        self.is_ff = self.model_class == FF
        self.node = kwargs.get('node', 'general')
        self.dataset = kwargs.get('dataset', '')
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        if optim_kwargs is None:
            self.optim_kwargs = load_kwargs(folder_config=dataset_train.folder_config, kwargs='optim_kwargs')
        else:
            self.optim_kwargs = optim_kwargs
        self.num_epochs = kwargs.get('num_epochs', 200)
        self.conv_class = kwargs.get('conv_class')
        self.is_optimized = False
        log_dir = './logs'
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f'optimization_{str(datetime.date.today())}.log')
        logging.basicConfig(filename=filename, level='INFO')
        self.logger = logging.getLogger()
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Init logger.')
        self.logger.info(f'Optimizer is initialized for model {self.model_class.__name__}.')

    def _run_epoch(self, optimizer, mode: str, loader: PyGDataLoader) -> float:
        """
        Runs a single training or evaluation epoch over a data loader.
 
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to update model parameters during training.
        mode : {'train', 'eval'}
            Mode indicating whether to train or evaluate.
        loader : PyGDataLoader
            DataLoader object for batch-wise iteration.
 
        Returns
        -------
        float
            Average RMSE loss over the entire epoch.
        """
 
        assert mode in ['train', 'eval']
        num_nodes = self.dataset_train.num_nodes
        self.model.train() if mode == 'train' else self.model.eval()
        total_loss, count = 0.0, 0
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE)
            batch.x = batch.x.float()
            if hasattr(batch, 'edge_weight') and batch.edge_weight is not None:
                batch.edge_weight = batch.edge_weight.float()
            out = self.model(batch.x, batch.edge_index, edge_weight=getattr(batch, 'edge_weight', None)).squeeze().view(-1, num_nodes).T           
 
            y_s = batch.y_scaled.view(-1, num_nodes).T
            loss = torch.sqrt(torch.mean((out - y_s) ** 2))
            del y_s, out
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
 
            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs
        return total_loss / count, optimizer
    
    def _run_epoch_ff(self, optimizer, mode: str, loader: PyGDataLoader) -> float:
        self.model.train() if mode == 'train' else self.model.eval()
        total_loss = 0.0
        loss_fn = torch.nn.MSELoss().to(DEVICE)
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.float().to(DEVICE), y_batch.to(DEVICE)
            y_pred = self.model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss, optimizer

    def _define_model(self, trial):
        vars = {}
        for (param, (vmin, vmax)) in zip(self.optim_kwargs.keys(), self.optim_kwargs.values()):
            if param != 'batch_size':
                if isinstance(vmin, int):
                    vars[param] = trial.suggest_int(param, vmin, vmax)
                elif isinstance(vmin, float):
                    if param != 'lr':
                        vars[param] = trial.suggest_float(param, vmin, vmax, log=False)
                    else:
                        vars[param] = trial.suggest_float(param, vmin, vmax, log=True)
                        self.lr = vars[param]
                else:
                    # trial.suggest_categorical(param, categories)
                    print(param, vmin, vmax, type(vmin))
                    raise NotImplementedError()
            else:
                pass   
        if self.is_ff:
            model = self.model_class(in_channels=self.dataset_val.tensors[0].shape[-1], out_channels=OUT_CHANNELS, **vars)
        else:     
            model = self.model_class(in_channels=self.dataset_val.num_node_features, conv_class=self.conv_class, conv_kwargs=vars, out_channels=OUT_CHANNELS, **vars)
        return model
    
    def _objective(self, trial):
        DEVICE = torch.device('cuda')
        self.model = self._define_model(trial).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.is_ff:
            if self.dataset == 'rfrance':
                batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
            else:
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        else:
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        if not(self.is_ff):
            adj_matrix = trial.suggest_categorical('adj_matrix', os.listdir(self.dataset_train.graph_folder))
            self.dataset_train._set_adj_matrix(adj_matrix=adj_matrix)
            self.dataset_val._set_adj_matrix(adj_matrix=adj_matrix)
            saving_directory = f'./checkpoints_optim/{self.model.__class__.__name__}{self.model.heads}_{self.dataset_train.adj_matrix}/batch{batch_size}_hidden{self.model.hidden_channels}_layers{self.model.num_layers}_epochs{self.num_epochs}'
            train_loader = PyGDataLoader(self.dataset_train, batch_size=batch_size, shuffle=True)
            val_loader = PyGDataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)
        else:
            saving_directory = f'./checkpoints_optim/{self.model.__class__.__name__}/{self.node}/batch{batch_size}_hidden{self.model.hidden_channels}_layers{self.model.num_layers}_epochs{self.num_epochs}'
            train_loader = DataLoader(self.dataset_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(self.dataset_val, batch_size=batch_size, shuffle=False)
        os.makedirs(saving_directory, exist_ok=True)
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        for epoch in tqdm(range(self.num_epochs)):
            params_filename = 'epoch{}.params'.format(epoch)
            if not(self.is_ff):
                train_loss, optimizer = self._run_epoch(optimizer=optimizer, mode='train', loader=train_loader)
                val_loss, _ = self._run_epoch(optimizer=optimizer, mode='eval', loader=val_loader)
            else:
                train_loss, optimizer = self._run_epoch_ff(optimizer=optimizer, mode='train', loader=train_loader)
                val_loss, _ = self._run_epoch_ff(optimizer=optimizer, mode='eval', loader=val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                clean_dir(saving_directory)
                torch.save(self.model.state_dict(), os.path.join(saving_directory, params_filename))
            trial.report(best_loss, epoch)
            if trial.should_prune():
                shutil.rmtree(saving_directory)
                raise optuna.exceptions.TrialPruned()
        return best_loss
    
    def optimize(self, **kwargs):
        self.storage = optuna.storages.InMemoryStorage()
        self.is_optimized = True
        self.study = optuna.create_study(storage=self.storage,
                                         study_name=kwargs.get('study_name', f'{self.model_class.__name__}_hpo'),
                                         direction=kwargs.get('direction', 'minimize'))
        self.logger.info('Optimization began.')
        self.study.optimize(self._objective, n_trials=kwargs.get('n_trials', 200), timeout=kwargs.get('timeout', 10000))
        self.logger.info('Optimization finished.')
        self.pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        self.complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(self.pruned_trials))
        print("  Number of complete trials: ", len(self.complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        result_dir = f'./results_optim_{self.dataset}/{self.conv_class.__name__}'
        result_file = os.path.join(result_dir, f'results_{self.node}.txt')
        os.makedirs(result_dir, exist_ok=True)

        with open(result_file, 'a') as f:
            f.write("Study statistics:\n")
            f.write(f"  Number of finished trials: {len(self.study.trials)}\n")
            f.write(f"  Number of pruned trials: {len(self.pruned_trials)}\n")
            f.write(f"  Number of complete trials: {len(self.complete_trials)}\n\n")

            f.write("Best trial:\n")
            f.write(f"  Value: {trial.value}\n\n")
            f.write("  Params:\n")
            for key, value in trial.params.items():
                f.write(f"    {key}: {value}\n")

    def run_on_server(self):
        if self.is_optimized:
            run_server(self.storage)
        else:
            print('You need to optimize your model first!')

def worker_optuna(conv_class, optim_kwargs, num_epochs, n_trials, graph_dataset_train, graph_dataset_val, dataset):
    seed_everything(42)
    worker_opt = Optimizer(model=myGNN, 
                            dataset_train=graph_dataset_train, 
                            dataset_val=graph_dataset_val, 
                            conv_class=conv_class,
                            optim_kwargs=optim_kwargs,
                            num_epochs=num_epochs,
                            dataset=dataset)
    worker_opt.optimize(n_trials=n_trials)
    print(f'Optimization finished for {conv_class.__name__}')
    torch.cuda.empty_cache()

def worker_optuna_ff(node, optim_kwargs, num_epochs, n_trials, X_train_node, y_train_node, X_val_node, y_val_node, dataset):
    seed_everything(42)
    X_train = X_train_node[node]
    y_train = y_train_node[node]
    X_val = X_val_node[node]
    y_val = y_val_node[node]
    worker_opt = Optimizer(model=FF,
                            dataset_train=TensorDataset(X_train, y_train), 
                            dataset_val=TensorDataset(X_val, y_val),
                            optim_kwargs=optim_kwargs,
                            num_epochs=num_epochs,
                            node=node,
                            dataset=dataset,
                            conv_class=FF)
    worker_opt.optimize(n_trials=n_trials)
    print(f'Optimization finished for FF')
    torch.cuda.empty_cache()

def worker_train_gnn(conv_class, num_epochs, patience, params, device, train, val, test):
    out_channels = 48
    if conv_class == GraphSAGE:
        model = GraphSAGE(
            in_channels=train.num_node_features,
            num_layers=params["num_layers"],
            hidden_channels=params["hidden_channels"],
            out_channels=out_channels
        ).to(device)
    else:
        model = myGNN(
            in_channels=train.num_node_features,
            num_layers=params["num_layers"],
            hidden_channels=params["hidden_channels"],
            out_channels=out_channels,
            conv_class=conv_class,
            conv_kwargs={k: params[k] for k in ["heads", "K", "alpha"] if k in params}
        ).to(device)

    trainer = Trainer(
        model=model,
        dataset_train=train,
        dataset_val=val,
        dataset_test=test,
        batch_size=params["batch_size"],
        return_attention=False,
        model_kwargs={'lr': params["lr"], 'num_epochs': num_epochs},
        lam_reg=params["lam_reg"]
    )

    pred_model_test, target_test, _, _ = trainer.train(
        plot_loss=False,
        force_training=True,
        save=False,
        patience=patience
    )

    preds = pred_model_test.sum(dim=0).cpu().detach()
    targets = target_test.sum(dim=0).cpu().detach()

    result = {
        "model": conv_class.__name__,
        "rmse": getattr(graphtoolbox.training.metrics, 'RMSE')(preds=preds, targets=targets).item(),
        "mape": getattr(graphtoolbox.training.metrics, 'MAPE')(preds=preds, targets=targets).item()
    }

    return result, preds

def worker_train_ff(node, num_epochs, patience, params, device, X_train_node, y_train_node, X_val_node, y_val_node, X_test_node, y_test_node, global_scalerY):
    out_channels = 48
    X_train = X_train_node[node].to(device)
    y_train = y_train_node[node].to(device)
    X_val = X_val_node[node].to(device)
    y_val = y_val_node[node].to(device)
    X_test = X_test_node[node].to(device)
    y_test = y_test_node[node].to(device)

    model = FF(
        in_channels=X_train.shape[-1],
        num_layers=params["num_layers"],
        hidden_channels=params["hidden_channels"],
        out_channels=out_channels
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = torch.nn.MSELoss().to(device)
    loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=params["batch_size"])
        
    best_val_rmse = float('inf')
    state_dict = None
    best_model_state = state_dict
    if best_model_state is None:
        for _ in tqdm(range(num_epochs)):
            model.train()
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.float().to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_val)
                val_rmse = np.sqrt(loss_fn(
                    y_pred_val,
                    y_val
                ).item())

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model_state = model.state_dict()

    model.eval()
    model.load_state_dict(best_model_state)
    y_pred_test = model(X_test)
    y_pred_test, y_true_test = global_scalerY.inverse_transform(y_pred_test.cpu().detach().numpy()), global_scalerY.inverse_transform(y_test.cpu().detach().numpy())

    result = {
        "node": node,
        "rmse": getattr(graphtoolbox.training.metrics, 'RMSE')(preds=y_pred_test, targets=y_true_test),
        "mape": getattr(graphtoolbox.training.metrics, 'MAPE')(preds=y_pred_test, targets=y_true_test)
    }

    return result, y_pred_test

