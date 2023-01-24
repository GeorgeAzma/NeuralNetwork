#pragma once

#include <vector>
#include <memory>
#include "activations.h"

class Layer
{
    friend class NeuralNetwork;
public:
    Layer(NeuralNetwork& neural_network): net(&neural_network) {}
    Layer(NeuralNetwork& neural_network, size_t index, size_t input_size, size_t size, const std::shared_ptr<Activation>& activation);

    void build();

    void forward();
    void calculateGradients(const std::vector<double>& targets);

    void save(std::ostream& os) const;
    void load(std::istream& is);

    static friend std::ostream &operator<<(std::ostream &os, const Layer &layer)
    {
        layer.save(os);
        return os;
    }
    static friend std::istream &operator>>(std::istream &is, Layer &layer)
    {
        layer.load(is);
        return is;
    }

    NeuralNetwork* net;
    size_t size;
    size_t input_size;
    size_t index;
    std::vector<double> neurons;
    std::vector<double> activated_neurons;
    std::vector<double> neuron_errors;
    std::vector<double> biases;
    std::vector<double> delta_biases;
    std::vector<std::vector<double>> weights; // [Layer][Neuron][Weight coming from previous neuron layer neurons to this neuron]
    std::vector<std::vector<double>> delta_weights;
    std::shared_ptr<Activation> activation;
};