import json

with open("100000.json") as f:
    model = json.load(f)

"""
Format
num of stage: int

num of models in this stage: int

num of layers in this model: int

weight size (row): int
weight size (col): int
weight matrix (rows)
bias size (row): int
bias size (col): int
bias matrix (rows)
"""

parsed = {}
num_stage = len(model)
print num_stage

for stage_idx in range(num_stage):
    for stage in model:
        if stage['stage'] == stage_idx + 1:
            params = stage['parameters']
            num_model = len(params)
            print num_model
            for model_idx in range(num_model):
                md = params[str(model_idx)]
                num_layer = len(md['bias'])
                print num_layer
                for layer_idx in range(num_layer):
                    weight = md['weights'][layer_idx]
                    bias = md['bias'][layer_idx]
                    weight_rows = len(weight)
                    weight_cols = len(weight[0])
                    print weight_rows
                    print weight_cols
                    for i in range(weight_rows):
                        for j in range(weight_cols):
                            print weight[i][j]
                    if type(bias[0]) is float:
                        bias_rows = 1
                        bias_cols = len(bias)
                        print bias_rows
                        print bias_cols
                        for i in range(bias_cols):
                            print bias[i]
                    else:
                        bias_rows = len(bias)
                        bias_cols = len(bias[0])
                        print bias_rows
                        print bias_cols
                        for i in range(bias_rows):
                            for j in range(bias_cols):
                                print bias[i][j]

            
    
