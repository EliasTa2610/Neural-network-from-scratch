// mlp.cpp : This file contains the `main` function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <functional>
#include <limits>
#include <Eigen/Core>
#include <input.h>
#include <net.h>
#include <labels.h>

using std::string;

int main()
{  
    // Step 1: Load data
    constexpr unsigned num_files = 3;

    string data_path = "./data/iris_data_files/";
    string filenames[num_files] = { "iris_training.dat", 
                                    "iris_validation.dat", 
                                    "iris_test.dat" };
    
    ArrayX_RowMajor<float> train_data_labels, val_data_labels, test_data_labels;
    ArrayX_RowMajor<float>* data_objs[num_files] = { &train_data_labels, &val_data_labels,
                                                                &test_data_labels };
    for (int i = 0; i < num_files; i++) {
        string filename = data_path + filenames[i]; 
        std::ifstream file(filename);
        *data_objs[i] = Input::readData(file);
    }

    // Step 2: Prepare data
    auto toDataIndicesPairs = [](auto& data_labels) {
        auto inputs = data_labels(Eigen::all, Eigen::seq(0, 3));
        auto ones_labels = data_labels(Eigen::all, Eigen::lastN(3)).cast<bool>();
        return std::make_pair(inputs, ones_labels);
    };

	auto train_pair = toDataIndicesPairs(train_data_labels);
    auto val_pair = toDataIndicesPairs(val_data_labels);
    auto test_pair = toDataIndicesPairs(val_data_labels);

    auto train_inputs = train_pair.first;
    auto train_labels = train_pair.second;
    auto val_inputs = val_pair.first;
    auto val_labels = val_pair.second;
    auto test_inputs = test_pair.first;
    auto test_labels = test_pair.second;

    // Step 3: Build neural net
    auto hidden_layer = Neural::PlainLinearLayer<decltype(train_inputs.eval())>(4, 4, 1.0);
    auto output_layer = Neural::PlainLinearLayer<decltype(train_inputs.eval())>(4, 3, 1.0);

    auto nn = Neural::MultiClassNN(train_inputs.eval(), train_labels.eval(), output_layer);
    nn.pushLayer(hidden_layer);
   
    // Step 4: Train neural net
    float lr = 0.1;
    float decay_rate = 0.1;
    float decayed_lr = lr;
    float val_loss = std::numeric_limits<float>::max();
    float val_loss_new = std::numeric_limits<float>::max() - 1;
    int violations = 0;

    // Stopping condition for training loop is whether validation loss (categorical cross entropy) has
    // stopped decreasing 3 times.
    int i = 0;
    while (violations < 3) {
        i++;

        nn.train(decayed_lr);
       
        val_loss_new = nn.test(val_inputs, val_labels).first;
        violations += (int)(val_loss_new >= val_loss);
        val_loss = val_loss_new;
        
        decayed_lr = lr/(1.0 + (float)i * decay_rate);
   }

    // Step 5: Test the neural net
    auto test_misclas = nn.test(test_inputs, test_labels).second;
    std::cout << "Test misclass. loss: " << test_misclas << std::endl;

     


}
