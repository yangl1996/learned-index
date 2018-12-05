#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>

using Eigen::MatrixXd;

int main(int argc, char *argv[])
{
    int num_stage;      // num of stages
    int *num_model;     // num of models in each stage
    int **num_layer;    // num of NN layers in each layer
    int ***weight_rows;
    int ***weight_cols;
    int ***bias_rows;
    int ***bias_cols;
    double ***weights;
    double ***biases;
    
    std::ifstream modelfs(argv[1]);
    if(!modelfs) {
        std::cout << "Failed to open model file." << std::endl;
        return 1;
    }
    modelfs >> num_stage;
    
    num_model = new int[num_stage];
    num_layer = new int*[num_stage];
    weight_rows = new int**[num_stage];
    weight_cols = new int**[num_stage];
    bias_rows = new int**[num_stage];
    bias_cols = new int**[num_stage];
    weights = new double**[num_stage];
    biases = new double**[num_stage];
            
    for (int i = 0; i < num_stage; i++) {
        modelfs >> num_model[i];
        num_layer[i] = new int[num_model[i]];
        weight_rows[i] = new int*[num_model[i]];
        weight_cols[i] = new int*[num_model[i]];
        bias_rows[i] = new int*[num_model[i]];
        bias_cols[i] = new int*[num_model[i]];
        weights[i] = new double*[num_model[i]];
        biases[i] = new double*[num_model[i]];
        
        for (int j = 0; j < num_model[i]; j++) {
            modelfs >> num_layer[i][j];
            weight_rows[i][j] = new int[num_layer[i][j]];
            weight_cols[i][j] = new int[num_layer[i][j]];
            bias_rows[i][j] = new int[num_layer[i][j]];
            bias_cols[i][j] = new int[num_layer[i][j]];
            weights[i][j] = new double[num_layer[i][j]];
            biases[i][j] = new double[num_layer[i][j]];
            
            for (int k = 0; k < num_layer[i][j]; k++) {
                modelfs >> weight_rows[i][j][k];
                modelfs >> weight_cols[i][j][k];
                if (weight_rows[i][j][k] != 1 || weight_cols[i][j][k] != 1) {
                    std::cout << "Weights should be 1x1" << std::endl;
                    return 1;
                }
                for (int r = 0; r < weight_rows[i][j][k]; r++) {
                    for (int c = 0; c < weight_cols[i][j][k]; c++) {
                        double temp;
                        modelfs >> temp;
                        weights[i][j][k] = temp;
                    }
                }
                
                modelfs >> bias_rows[i][j][k];
                modelfs >> bias_cols[i][j][k];
                if (bias_rows[i][j][k] != 1 || bias_cols[i][j][k] != 1) {
                    std::cout << "Biases should be 1x1" << std::endl;
                    return 1;
                }
                for (int r = 0; r < bias_rows[i][j][k]; r++) {
                    for (int c = 0; c < bias_cols[i][j][k]; c++) {
                        double temp;
                        modelfs >> temp;
                        biases[i][j][k] = temp;
                    }
                }
            }
        }
    }
    
    modelfs.close();
    
    std::ifstream datafs(argv[2]);
    if(!datafs) {
        std::cout << "Failed to open data file." << std::endl;
        return 1;
    }

    int num_data;
    int *data;
    int *pos;

    datafs >> num_data;
    data = new int[num_data];
    pos = new int[num_data];

    for (int i = 0; i < num_data; i++) {
        datafs >> data[i];
        datafs >> pos[i];
    }

    datafs.close();
    
    std::cout << "Press enter to start" << std::endl;
    std::cin.get();
    
    double err = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();
    double res;
    for (int i = 0; i < num_data; i++) {
        int model_idx = 0;
        for (int stg = 0; stg < num_stage; stg++) {
            res = double(data[i]);
            for (int lay = 0; lay < num_layer[stg][model_idx]; lay++) {
                res = res * weights[stg][model_idx][lay] + biases[stg][model_idx][lay];
            }
            model_idx = int(res);
            if (model_idx >= num_model[stg+1]) {
                model_idx = num_model[stg+1] - 1;
            }
        }
        int final_res = int(res);
        err += abs(final_res - pos[i]);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / (double)(1000000 * num_data) << std::endl;
    
    std::cout << err / num_data << std::endl;
}
