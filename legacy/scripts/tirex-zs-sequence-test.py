import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import os
from tqdm.notebook import tqdm
import warnings
import argparse
import yaml
from datetime import datetime, timedelta

from tirex import load_model, ForecastModel, TiRexZero
from tirex_util import load_tirex_from_checkpoint, plot_forecast, create_incrementing_folder
import pandas as pd

# ### Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Configuration file path')
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

### Import data
data = pd.read_csv(config['data_path'])
data['Date_local'] = pd.to_datetime(data['Date_local']).dt.tz_localize(None)
begin_train = pd.to_datetime(config['begin_train'])
begin_test = pd.to_datetime(config['begin_test'])
end_test = pd.to_datetime(config['end_test'])
folder_path = 'experiments/results/' + config['expe_name']
folder_path = create_incrementing_folder(folder_path)

#### Filter relevant data
X = data[(data['Date_local'] >= begin_train) & (data['Date_local'] < end_test)].reset_index(drop=True)
y_past = X[X['Date_local'] < begin_test]['Consommation'].values
y_test = X[(X['Date_local'] >= begin_test) & (X['Date_local'] < end_test)]['Consommation'].values

#### Load TiREX ZS
CHECKPOINT_FILE = "models/tirex.ckpt"
MODEL_ID = "TiRex"

try:
    if os.path.isfile(CHECKPOINT_FILE):
        tirex_model = load_tirex_from_checkpoint(
            checkpoint_path=CHECKPOINT_FILE,
            model_id=MODEL_ID
        )
        print("\nModèle chargé depuis le checkpoint :")
    else:
        print(f"\nCheckpoint introuvable ({CHECKPOINT_FILE}), chargement via load_model...")
        tirex_model: ForecastModel = load_model("NX-AI/TiRex")

    print("Modèle final obtenu :")
    print(tirex_model)

except Exception as e:
    print(f"\n\nUNE ERREUR EST SURVENUE : {e}")

### Perform zero-shot forecasting iteratively (could be made much more efficient)
quantiles_full = []
mean_full = []
horizon = config['horizon']
hist = X[X['Date_local'] <= begin_test]['Consommation'].values
fut = X[X['Date_local'] > begin_test]['Consommation'].values
for i in range(0, len(y_test), horizon):
    #y_train = X[X['Date_local'] <= begin_test + pd.to_timedelta(i/horizon, unit='D')]['Consommation'].values
    ctx = np.concatenate((hist, fut[0:i]))
    quantiles, mean = tirex_model.forecast(ctx, prediction_length=horizon)
    mean_full.append(mean)
    quantiles_full.append(quantiles[0])

#### Gather forecasts and save them in relevant folder
m = torch.cat(mean_full,dim=1)
q = torch.cat(quantiles_full,dim=0)

q_unbind = torch.unbind(q)
fifth_elements = [tensor[4] for tensor in q_unbind]
first_elements = [tensor[0] for tensor in q_unbind]
ninth_elements = [tensor[8] for tensor in q_unbind]
median_pred = torch.tensor(fifth_elements)
q10_pred = torch.tensor(first_elements)
q90_pred = torch.tensor(ninth_elements)

y_test = torch.tensor(y_test)

dates_test = X[(X['Date_local'] >= begin_test) & (X['Date_local'] < end_test)]['Date_local']

print(len(dates_test))
print(len(y_test))
print(len(median_pred))

final = pd.DataFrame({
    'date':dates_test,
    'obs':y_test,
    'q10_pred':q10_pred,
    'q90_pred':q90_pred,
    'median_pred':median_pred})

final.to_csv(folder_path + '/sequence.csv',index=False)

#### END
