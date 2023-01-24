#include "layer.h"
#include "neural_network.h"

Layer::Layer(NeuralNetwork& neural_network, size_t index, size_t input_size, size_t size, const std::shared_ptr<Activation>& activation)
    : net(&neural_network), index(index), input_size(input_size), size(size), activation(activation)
{
    build();
}

void Layer::build()
{
    activated_neurons.resize(size);
    if (index)
    {
        neurons.resize(size);
        neuron_errors.resize(size);
        biases.resize(size);
        delta_biases.resize(size);
        weights.resize(size);
        delta_weights.resize(size);
        for (size_t neuron = 0; neuron < size; ++neuron)
        {
            weights[neuron].resize(input_size);
            delta_weights[neuron].resize(input_size);
        }
    }
}

void Layer::forward()
{
    const auto& prev_layer = net->layers[index - 1];
    for (size_t neuron = 0; neuron < size; ++neuron)
    {
        neurons[neuron] = biases[neuron];
        for (size_t prev_neuron = 0; prev_neuron < input_size; ++prev_neuron)
            neurons[neuron] += prev_layer.activated_neurons[prev_neuron] * weights[neuron][prev_neuron];
        activated_neurons[neuron] = (*activation)(neurons[neuron]);
    }
}

void Layer::calculateGradients(const std::vector<double>& targets)
{
    auto& prev_layer = net->layers[index - 1];
    if (index > 1)
    {
        for (size_t prev_neuron = 0; prev_neuron < input_size; ++prev_neuron)
        {
            prev_layer.neuron_errors[prev_neuron] = 0.0;
            for (size_t neuron = 0; neuron < size; ++neuron)
                prev_layer.neuron_errors[prev_neuron] += neuron_errors[neuron] * weights[neuron][prev_neuron];
            prev_layer.neuron_errors[prev_neuron] *= prev_layer.activation->derivative(prev_layer.neurons[prev_neuron]);
        }
    }
    for (size_t prev_neuron = 0; prev_neuron < input_size; ++prev_neuron)
        for (size_t neuron = 0; neuron < size; ++neuron)
            delta_weights[neuron][prev_neuron] += neuron_errors[neuron] * prev_layer.activated_neurons[prev_neuron];

    for (size_t neuron = 0; neuron < size; ++neuron)
        delta_biases[neuron] += neuron_errors[neuron];
}


void Layer::save(std::ostream& os) const
{
    os.write((const char*)&size, sizeof(size));
    os.write((const char*)&input_size, sizeof(input_size));
    os.write((const char*)&index, sizeof(index));
    os.write((const char*)&activation->type, sizeof(activation->type));
    os << *activation;
    for (size_t neuron = 0; neuron < size; ++neuron)
        for (size_t prev_neuron = 0; prev_neuron < weights.size(); ++prev_neuron)
            os.write((const char*)&weights[neuron][prev_neuron], sizeof(weights[neuron][prev_neuron]));
    for (size_t neuron = 0; neuron < size; ++neuron)
        os.write((const char*)&biases[neuron], sizeof(biases[neuron]));
}

void Layer::load(std::istream& is)
{
    is.read((char*)&size, sizeof(size));
    is.read((char*)&input_size, sizeof(input_size));
    is.read((char*)&index, sizeof(index));
    Activation::Type activation_type;
    is.read((char*)&activation_type, sizeof(activation_type));
    activation = ActivationFactory::build(activation_type);
    is >> *activation;
    build();
    for (size_t neuron = 0; neuron < size; ++neuron)
        for (size_t prev_neuron = 0; prev_neuron < weights.size(); ++prev_neuron)
            is.read((char*)&weights[neuron][prev_neuron], sizeof(weights[neuron][prev_neuron]));
    for (size_t neuron = 0; neuron < size; ++neuron)
        is.read((char*)&biases[neuron], sizeof(biases[neuron]));

}