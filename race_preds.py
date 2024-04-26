import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

data_path = 'data/'

def time_str_to_sec(time_str):
    if isinstance(time_str, float):
        return time_str
    if '\\N' in str(time_str):
        return None
    if '.' not in str(time_str):
        return None
    parts = str(time_str).split(':')
    if len(parts) > 1:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

# on already cleaned data
race_df = pd.read_csv(data_path + 'races.csv')
results_df = pd.read_csv(data_path + 'results.csv')

race_subset = race_df[race_df['year'] >= 2001]
race_ids = race_subset['raceId']

unique_drivers = pd.DataFrame(columns=['driverId'])
for race_id in tqdm(race_ids):
    race_results = results_df[results_df['raceId'] == race_id]
    unique_drivers = pd.concat([unique_drivers, race_results[['driverId']]], ignore_index=True)

driver_df = pd.DataFrame({'driverId': unique_drivers['driverId'].unique()})
driver_df = driver_df.sort_values(by=['driverId']).reset_index(drop=True)
driver_df.to_csv(data_path + 'drivers_short.csv', index=False)

for year in range(2001, 2020):
    year_path = f'{data_path}/races/{year}'
    if not os.path.exists(year_path):
        os.makedirs(year_path)

    cur_year_races = os.listdir(f'{data_path}/races_v3/{year}/')
    for race_file in cur_year_races:
        race_data = pd.read_csv(f'{data_path}/races_v3/{year}/{race_file}')
        for i in range(20):
            for lap in range(len(race_data) - 1):
                if race_data[f'inPit{i + 1}'][lap + 1] == 1:
                    race_data[f'inPit{i + 1}'][lap] = 1
                    race_data[f'inPit{i + 1}'][lap + 1] = 0
            race_data.rename(columns={f'inPit{i + 1}': f'pitting{i + 1}'}, inplace=True)

        race_data.to_csv(f'{data_path}/races/{year}/{race_file}', index=False)

class RaceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(RaceModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def zero_states(self, batch_size):
        hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
        cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
        return hidden_state, cell_state

    def forward(self, input_seq, prev_states=None):
        lstm_out, new_states = self.lstm(input_seq, prev_states)
        output = self.fc(lstm_out)
        return output, new_states



class RaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, start_year):
        self.data_path = data_path
        self.year = start_year
        self.round = 1
        self._load_race_data()

    def _load_race_data(self):
        self.cur_year_files = os.listdir(f'{self.data_path}/{self.year}/')
        if self.round <= len(self.cur_year_files):
            self.cur_race = pd.read_csv(f'{self.data_path}/{self.year}/{self.cur_year_files[self.round - 1]}')
        else:
            self.cur_race = pd.read_csv(f'{self.data_path}/{self.year}/{self.cur_year_files[-1]}')

    def set_year(self, year):
        self.year = year
        self._load_race_data()

    def set_round(self, round_num):
        self.round = round_num
        self._load_race_data()

    def __len__(self):
        return len(self.cur_race) - 1

    def __getitem__(self, idx):
        cur_lap = torch.tensor(self.cur_race.iloc[idx].values[1:142], dtype=torch.float32)
        next_lap = torch.tensor(self.cur_race.iloc[idx + 1].values[1:142], dtype=torch.float32)
        cur_lap[cur_lap.isnan()] = 0
        next_lap[next_lap.isnan()] = 0
        return cur_lap, next_lap

def train_model(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train() 
    loss_history = [] 

    for epoch in range(num_epochs):
        epoch_loss = 0.0  

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  
            targets = targets.to(device)

            optimizer.zero_grad() 
            outputs, _ = model(inputs)  
            loss = criterion(outputs, targets)  

            loss.backward()  
            optimizer.step() 

            epoch_loss += loss.item()  

        avg_loss = epoch_loss / len(train_loader) 
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    return loss_history


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = RaceModel(141, 141, 2, 0.2)
model.to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

dataset = RaceDataset(data_path + 'races/', 2001)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
loss_history = train_model(model, train_loader, optimizer, criterion, device, 10)


plt.plot(np.arange(1, 10 + 1), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()
