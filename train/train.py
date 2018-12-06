from itertools import count

import torch
import torch.nn as nn
import torch.utils.data as torch_data

def load_data(csv_source):
    data = []
    pos = []
    with open(csv_source) as ds:
        for line in ds:
            dp = line.rstrip().split(',')
            data.append(torch.tensor([float(dp[0])]))
            pos.append(torch.tensor([float(dp[1])]))
    return data, pos

class Dataset(torch_data.Dataset):
    def __init__(self, data_list, pos_list):
        self.data = data_list
        self.pos = pos_list

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

            
if __name__ == "__main__":
    data, pos = load_data("random.csv")
    ds = [[Dataset(data, pos)]]
    max_pos = 1000

    device = torch.device('cpu')
    models = []
    num_model = (1, 10)
    model_params = ((0, []), (0, []))
    num_stage = len(num_model)

    for stage_idx in range(num_stage):
        models.append([])
        if stage_idx != num_stage - 1:
            next_data = [[] for i in range(num_model[stage_idx+1])]
            next_pos = [[] for i in range(num_model[stage_idx+1])]

        for model_idx in range(num_model[stage_idx]):

            model = Network(model_params[stage_idx][0], model_params[stage_idx][1]).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
            data_gen = torch_data.DataLoader(ds[stage_idx][model_idx], batch_size=100, shuffle=False)

            for epoch in range(100):
                print(epoch)
                for local_data, local_pos in data_gen:
                    local_data, local_pos = local_data.to(device), local_pos.to(device)

                    outputs = model(local_data)
                    loss = criterion(outputs, local_pos)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                with torch.set_grad_enabled(False):
                    loss_tot = 0.0
                    for local_data, local_pos in data_gen:
                        local_data, local_pos = local_data.to(device), local_pos.to(device)
                        outputs = model(local_data)
                        loss = criterion(outputs, local_pos)
                        loss_tot += loss.item()
                    print(loss_tot)

            models[stage_idx].append(model)

            if stage_idx != num_stage - 1:
                data_gen = torch_data.DataLoader(ds[stage_idx][model_idx], batch_size=1, shuffle=False)
                for local_data, local_pos in data_gen:
                    local_data, local_pos = local_data.to(device), local_pos.to(device)
                    output = model(local_data)
                    model_sel = int(output.item() * num_model[stage_idx+1] / max_pos)
                    if model_sel >= num_model[stage_idx+1]:
                        model_sel = num_model[stage_idx+1] - 1
                    elif model_sel <= 0:
                        model_sel = 0
                    next_data[model_sel].append(local_data)
                    next_pos[model_sel].append(local_pos)

        if stage_idx != num_stage - 1:
            ds.append([])
            for next_model_idx in range(num_model[stage_idx+1]):
                ds[stage_idx+1].append(Dataset(next_data[next_model_idx], next_pos[next_model_idx]))
    
    test_ds = Dataset(data, pos)
    test_gen = torch_data.DataLoader(test_ds, batch_size=1, shuffle=False)
    err_calc = nn.MSELoss()
    err_tot = 0.0
    for local_data, local_pos in test_gen:
        local_data, local_pos = local_data.to(device), local_pos.to(device)
        model_sel = 0
        for stage_idx in range(num_stage):
            model = models[stage_idx][model_sel]
            output = model(local_data)
            if stage_idx != num_stage - 1:
                model_sel = int(output.item() * num_model[stage_idx+1] / max_pos)
                if model_sel >= num_model[stage_idx+1]:
                    model_sel = num_model[stage_idx+1] - 1
                elif model_sel <= 0:
                    model_sel = 0
            else:
                err_tot += err_calc(output, local_pos).item()
    print(err_tot / 100000)


            
            
