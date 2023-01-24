#include "neural_network.h"
#include "random.h"

void NeuralNetwork::forward(const std::vector<double> &inputs)
{
    layers.front().activated_neurons = inputs;

    for (size_t layer = 1; layer < layers.size(); ++layer)
        layers[layer].forward();
}
void NeuralNetwork::backpropagate(const std::vector<double> &targets, size_t iteration)
{
    calculateGradient(targets);
    optimize(iteration);
}

void NeuralNetwork::calculateGradient(const std::vector<double> &targets)
{
    for (size_t neuron = 0; neuron < layers.back().size; ++neuron)
        layers.back().neuron_errors[neuron] = (targets[neuron] - layers.back().activated_neurons[neuron]) * layers.back().activation->derivative(layers.back().neurons[neuron]);

    for (size_t layer = layers.size() - 1; layer >= 1; --layer)
        layers[layer].calculateGradients(targets);
}

void NeuralNetwork::optimize(size_t iteration)
{
    (*optimizer)(iteration);

    for (size_t layer = 1; layer < layers.size(); ++layer)
        for (size_t neuron = 0; neuron < layers[layer].size; ++neuron)
        {
            layers[layer].delta_biases[neuron] = 0.0;
            for (size_t prev_neuron = 0; prev_neuron < layers[layer].input_size; ++prev_neuron)
                layers[layer].delta_weights[neuron][prev_neuron] = 0.0;
        }
}

void NeuralNetwork::train(const std::vector<double> &inputs, const std::vector<double> &targets, size_t iteration)
{
    forward(inputs);
    backpropagate(targets, iteration);
}
void NeuralNetwork::train(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targets, size_t epochs)
{
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            forward(inputs[i]);
            calculateGradient(targets[i]);
        }
        optimize(epoch + 1);
    }
}

double NeuralNetwork::test(const std::vector<double> &inputs, const std::vector<double> &targets)
{
    forward(inputs);
    double cost = 0.0;
    for (size_t i = 0; i < getOutputCount(); ++i)
        cost += (targets[i] - getOutput()[i]) * (targets[i] - getOutput()[i]);
    cost /= getOutputCount();
    return cost;
}
double NeuralNetwork::test(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets)
{
    double cost = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i)
        cost += test(inputs[i], targets[i]);
    cost /= inputs.size();
    return cost;
}

void NeuralNetwork::initWeights()
{
    for (size_t layer = 1; layer < layers.size(); ++layer)
        for (size_t neuron = 0; neuron < layers[layer].size; ++neuron)
            for (size_t prev_neuron = 0; prev_neuron < layers[layer].input_size; ++prev_neuron)
                layers[layer].weights[neuron][prev_neuron] = (Random::Float() * 2.0 - 1.0) * 0.1;
}

static std::ostream &operator<<(std::ostream& os, const NeuralNetwork& net)
{
    uint32_t layer_count = net.getLayerCount();
    os.write((const char *)&layer_count, sizeof(layer_count));

    Optimizer::Type optimizer_type = net.optimizer->getType();
    os.write((const char *)&optimizer_type, sizeof(optimizer_type));
    os << *net.optimizer;

    for (const auto& layer : net.layers)
        os << layer;

    return os;
}
static std::istream &operator>>(std::istream& is, NeuralNetwork& net)
{
    uint32_t layer_count = net.getLayerCount();
    is.read((char *)&layer_count, sizeof(layer_count));
    net.layers.resize(layer_count, Layer(net));

    Optimizer::Type optimizer_type;
    is.read((char *)&optimizer_type, sizeof(optimizer_type));
    net.optimizer = OptimizerFactory::build(optimizer_type, net);
    is >> *net.optimizer;

    for (auto& layer : net.layers)
    {
        layer.net = &net;
        is >> layer;
    }

    return is;
}