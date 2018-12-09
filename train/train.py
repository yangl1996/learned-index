#!/usr/local/bin/python3
from itertools import count

import torch
import torch.nn as nn
import torch.utils.data as torch_data

def load_data(csv_source):
    """
    load data in csv file
    """
    data = []
    pos = []
    with open(csv_source) as ds:
        for line in ds:
            dp = line.rstrip().split(',')
            # each data/index should be a 1x1 PyTorch tensor
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
        """
        num_hidden: number of hidden layers
        hidden_size: a list of the sizes (num of neurons) of the hidden layers
        """
        super(Network, self).__init__()
        self.num_hidden = num_hidden
        self.fc = []    # fc layers
        self.relu = []  # ReLU activations
        input_size = 1  # size of the previous layer (input of current layer)
        for fc_idx in range(num_hidden):
            # add fc layer and ReLU activation
            self.fc.append(nn.Linear(input_size, hidden_size[fc_idx]))
            self.relu.append(nn.ReLU())
            # input of next layer should be output of this layer
            input_size = hidden_size[fc_idx]
        # the layer layer should always have 1-dim output
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
    # ds is a 2-dim list, entry i, j is the dataset for stage i, model j
    # initially it only contains the whole set, to be used for stage 0
    ds = [[Dataset(data, pos)]]

    # max of index (position), used to determine next-stage model
    max_pos = len(data)

    torch.set_num_threads(8)
    device = torch.device('cpu')

    # model is a 2-dim list, entry i, j is the model for stage i, model j
    models = []

    # num_model is a tuple, entry i is the number of models for stage i
    num_model = (1, 10)

    # model_params is a tuple, entry i is the params of models in stage i
    # each entry specifies (num of hidden layers, size of each hidden layer)
    model_params = ((2, [4, 8]), (2, [4, 8]))

    # number of stages
    num_stage = len(num_model)

    for stage_idx in range(num_stage):
        models.append([])
        # if it's the last stage, we don't need to prepare datasets for the
        # next stage. or, we need to initialize datasets for the next stage
        if stage_idx != num_stage - 1:
            next_data = [[] for i in range(num_model[stage_idx+1])]
            next_pos = [[] for i in range(num_model[stage_idx+1])]

        for model_idx in range(num_model[stage_idx]):
            # initialize a model
            model = Network(model_params[stage_idx][0], model_params[stage_idx][1]).to(device)
            # use MSE loss for training
            criterion = nn.MSELoss()
            # use Adam algo for training
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
            # load dataset
            data_gen = torch_data.DataLoader(ds[stage_idx][model_idx], batch_size=64, shuffle=False)
            # we will stop training when loss stops decreasing
            last_loss = float('inf')

            print("Stage={}, Model={}, {} data points".format(stage_idx, model_idx, len(ds[stage_idx][model_idx])))
            for epoch in range(5000):
                print("Epoch", epoch)
                # train model
                for local_data, local_pos in data_gen:
                    local_data, local_pos = local_data.to(device), local_pos.to(device)
                    # feedforward
                    outputs = model(local_data)
                    # calc loss
                    loss = criterion(outputs, local_pos)
                    # back propagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # calculate and print loss for this model at this epoch
                with torch.set_grad_enabled(False):
                    loss_tot = 0.0
                    for local_data, local_pos in data_gen:
                        local_data, local_pos = local_data.to(device), local_pos.to(device)
                        outputs = model(local_data)
                        loss = criterion(outputs, local_pos)
                        loss_tot += loss.item()
                    print("Loss:", loss_tot / len(ds[stage_idx][model_idx]))
                    # if lost stops decreasing, just stop training
                    if (last_loss - loss_tot) / last_loss < 0.001:
                        break
                    else:
                        last_loss = loss_tot

            # append the model we just trained to model tree
            models[stage_idx].append(model)

            # prepare datasets for the next stage. only when we're not at the last stage
            if stage_idx != num_stage - 1:
                # load the datapoints in current set one by one. we need to assign each of them to a
                # model in the next stage
                data_gen = torch_data.DataLoader(ds[stage_idx][model_idx], batch_size=1, shuffle=False)
                for local_data, local_pos in data_gen:
                    local_data, local_pos = local_data.to(device), local_pos.to(device)
                    # calculate which model in the next stage to assign to
                    # model_idx = output * num_model_next_stage / max_position
                    output = model(local_data)
                    model_sel = int(output.item() * num_model[stage_idx+1] / max_pos)
                    if model_sel >= num_model[stage_idx+1]:
                        model_sel = num_model[stage_idx+1] - 1
                    elif model_sel <= 0:
                        model_sel = 0
                    # append this datapoint to corresponding dataset
                    next_data[model_sel].append(local_data)
                    next_pos[model_sel].append(local_pos)
        
        # create the Dataset objects for the next stage
        if stage_idx != num_stage - 1:
            ds.append([])
            for next_model_idx in range(num_model[stage_idx+1]):
                ds[stage_idx+1].append(Dataset(next_data[next_model_idx], next_pos[next_model_idx]))
    
    # testing. load datapoints one by one
    test_ds = Dataset(data, pos)
    test_gen = torch_data.DataLoader(test_ds, batch_size=1, shuffle=False)
    err_tot = 0
    for local_data, local_pos in test_gen:
        local_data, local_pos = local_data.to(device), local_pos.to(device)
        model_sel = 0
        for stage_idx in range(num_stage):
            model = models[stage_idx][model_sel]
            output = model(local_data)
            # if it's not the last stage, the output determines which model
            # in the next stage to use
            if stage_idx != num_stage - 1:
                model_sel = int(output.item() * num_model[stage_idx+1] / max_pos)
                if model_sel >= num_model[stage_idx+1]:
                    model_sel = num_model[stage_idx+1] - 1
                elif model_sel <= 0:
                    model_sel = 0
            # if it's the last layer, the output is the position (index)
            else:
                err_tot += abs(int(output.item()) - int(local_pos.item()))

    print("Final Loss:", float(err_tot) / len(test_ds))


