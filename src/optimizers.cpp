#include "optimizers.h"
#include "neural_network.h"

void Gd::operator()(size_t iteration)
{
    for (size_t layer = 1; layer < net.getLayerCount(); ++layer)
        for (size_t next_neuron = 0; next_neuron < net.layers[layer].size; ++next_neuron)
            for (size_t neuron = 0; neuron < net.layers[layer].input_size; ++neuron)
                net.layers[layer].weights[next_neuron][neuron] += learning_rate * net.layers[layer].delta_weights[next_neuron][neuron];
    for (size_t layer = 1; layer < net.getLayerCount(); ++layer)
        for (size_t neuron = 0; neuron < net.layers[layer].size; ++neuron)
            net.layers[layer].biases[neuron] += learning_rate * net.layers[layer].delta_biases[neuron];
}

Sgd::Sgd(NeuralNetwork& net, double learning_rate, double momentum)
    : momentum(momentum), Optimizer(net, Type::SGD, learning_rate)
{
    weight_velocities.resize(net.getLayerCount() - 1);
    bias_velocities.resize(net.getLayerCount() - 1);
    for (size_t layer = 0; layer < weight_velocities.size(); ++layer)
    {
        weight_velocities[layer].resize(net.layers[layer + 1].size);
        bias_velocities[layer].resize(net.layers[layer + 1].size);
        for (size_t neuron = 0; neuron < weight_velocities[layer].size(); ++neuron)
            weight_velocities[layer][neuron].resize(net.layers[layer].size);
    }
}

void Sgd::operator()(size_t iteration)
{
    for (size_t layer = 1; layer < net.getLayerCount(); ++layer)
        for (size_t next_neuron = 0; next_neuron < net.layers[layer].size; ++next_neuron)
            for (size_t neuron = 0; neuron < net.layers[layer].input_size; ++neuron)
            {
                auto& weight_vel = weight_velocities[layer - 1][next_neuron][neuron];
                weight_vel = momentum * weight_vel + (1.0 - momentum) * net.layers[layer].delta_weights[next_neuron][neuron];
                net.layers[layer].weights[next_neuron][neuron] += learning_rate * weight_vel;
            }
    for (size_t layer = 1; layer < net.getLayerCount(); ++layer)
        for (size_t neuron = 0; neuron < net.layers[layer].size; ++neuron)
        {
            auto& bias_vel = bias_velocities[layer - 1][neuron];
            bias_vel = momentum * bias_vel + (1.0 - momentum) * net.layers[layer].delta_biases[neuron];
            net.layers[layer].biases[neuron] += learning_rate * bias_vel;
        }
}

void Sgd::reset()
{
    for (size_t layer = 0; layer < weight_velocities.size(); ++layer)
    {
        std::fill(bias_velocities[layer].begin(), bias_velocities[layer].end(), 0.0);
        for (size_t neuron = 0; neuron < weight_velocities[layer].size(); ++neuron)
            std::fill(weight_velocities[layer][neuron].begin(), weight_velocities[layer][neuron].end(), 0.0);
    }
}

Adam::Adam(NeuralNetwork& net, double learning_rate, double beta1, double beta2)
    : beta1(beta1), beta2(beta2), Optimizer(net, Type::ADAM, learning_rate)
{
    weight_velocities.resize(net.getLayerCount() - 1);
    square_weight_velocities.resize(weight_velocities.size());
    bias_velocities.resize(net.getLayerCount() - 1);
    square_bias_velocities.resize(bias_velocities.size());
    for (size_t layer = 0; layer < weight_velocities.size(); ++layer)
    {
        weight_velocities[layer].resize(net.layers[layer + 1].size);
        square_weight_velocities[layer].resize(weight_velocities[layer].size());
        bias_velocities[layer].resize(net.layers[layer + 1].size);
        square_bias_velocities[layer].resize(bias_velocities[layer].size());
        for (size_t neuron = 0; neuron < weight_velocities[layer].size(); ++neuron)
        {
            weight_velocities[layer][neuron].resize(net.layers[layer].size);
            square_weight_velocities[layer][neuron].resize(weight_velocities[layer][neuron].size());
        }
    }
}

void Adam::reset()
{
    for (size_t layer = 0; layer < weight_velocities.size(); ++layer)
    {
        std::fill(bias_velocities[layer].begin(), bias_velocities[layer].end(), 0.0);
        std::fill(square_bias_velocities[layer].begin(), square_bias_velocities[layer].end(), 0.0);
        for (size_t neuron = 0; neuron < weight_velocities[layer].size(); ++neuron)
        {
            std::fill(weight_velocities[layer][neuron].begin(), weight_velocities[layer][neuron].end(), 0.0);
            std::fill(square_weight_velocities[layer][neuron].begin(), square_weight_velocities[layer][neuron].end(), 0.0);
        }
    }
}

void Adam::operator()(size_t iteration)
{
    double epsilon = 1e-7;

    double bi1 = 1.0 - pow(beta1, iteration);
    double bi2 = 1.0 - pow(beta2, iteration);

    for (size_t layer = 1; layer < net.getLayerCount(); ++layer)
        for (size_t next_neuron = 0; next_neuron < net.layers[layer].size; ++next_neuron)
            for (size_t neuron = 0; neuron < net.layers[layer].input_size; ++neuron)
            {
                auto& vel = weight_velocities[layer - 1][next_neuron][neuron];
                auto& sq_vel = square_weight_velocities[layer - 1][next_neuron][neuron];
                vel = beta1 * vel + (1.0 - beta1) * net.layers[layer].delta_weights[next_neuron][neuron];
                sq_vel = beta2 * sq_vel + (1.0 - beta2) * net.layers[layer].delta_weights[next_neuron][neuron] * net.layers[layer].delta_weights[next_neuron][neuron];
                net.layers[layer].weights[next_neuron][neuron] += learning_rate * (vel / bi1) / (sqrt(sq_vel / bi2) + epsilon);
            }
    for (size_t layer = 1; layer < net.getLayerCount(); ++layer)
        for (size_t neuron = 0; neuron < net.layers[layer].size; ++neuron)
        {
            auto& vel = bias_velocities[layer - 1][neuron];
            auto& sq_vel = square_bias_velocities[layer - 1][neuron];
            vel = beta1 * vel + (1.0 - beta1) * net.layers[layer].delta_biases[neuron];
            sq_vel = beta2 * sq_vel + (1.0 - beta2) * net.layers[layer].delta_biases[neuron] * net.layers[layer].delta_biases[neuron];
            net.layers[layer].biases[neuron] += learning_rate * (vel / bi1) / (sqrt(sq_vel / bi2) + epsilon);
        }
}