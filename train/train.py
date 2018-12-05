from itertools import count

import torch
import torch.nn as nn
import torch.autograd
from torch.utils import data
import torch.nn.functional as F

class Dataset(data.Dataset):
    
    def __init__(self, csv_source):
        self.data = []
        self.pos = []
        with open(csv_source) as ds:
             for line in ds:
                dp = line.rstrip().split(',')
                self.data.append(torch.tensor([float(dp[0])]))
                self.pos.append(torch.tensor([float(dp[1])]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.pos[index]

class Network(nn.Module):
    
    def __init__(self, num_hidden, hidden_size):
        super(Network, self).__init__()
        self.num_hidden = num_hidden
        self.fc = []
        self.relu = []
        input_size = 1
        for fc_idx in range(num_hidden):
            self.fc.append(nn.Linear(input_size, hidden_size[fc_idx]))
            self.relu.append(nn.ReLU())
            input_size = hidden_size[fc_idx]
        self.last = nn.Linear(input_size, 1)

    def forward(self, x):
        out = x
        for fc_idx in range(self.num_hidden):
            out = self.fc[fc_idx](out)
            out = self.relu[fc_idx](out)
        out = self.last(out)
        return out

random_dataset = Dataset("random.csv")
dataset_generator = data.DataLoader(random_dataset, batch_size=1, shuffle=False)

device = torch.device('cpu')
model = Network(0, []).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.5) 

for epoch in range(100):
    print(epoch)
    for local_data, local_pos in dataset_generator:
        local_data, local_pos = local_data.to(device), local_pos.to(device)

        outputs = model(local_data)
        loss = criterion(outputs, local_pos)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("loss =", loss.item())
    with torch.set_grad_enabled(False):
        loss_tot = 0.0
        for local_data, local_pos in dataset_generator:
            local_data, local_pos = local_data.to(device), local_pos.to(device)
            outputs = model(local_data)
            loss = criterion(outputs, local_pos)
            loss_tot += loss.item()
        print(loss_tot)
            

