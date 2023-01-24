#pragma once

#include <iostream>
#include <vector>
#include <concepts>
#include "optimizers.h"
#include "layer.h"

class NeuralNetwork
{
    friend class Layer;
public:
    NeuralNetwork() = default;

    template <typename T = Relu, typename... Args>
    void add(uint32_t size, Args &&...args)
    {
        layers.push_back(Layer(*this, layers.size(), layers.size() ? layers.back().size : 0, size, std::make_shared<T>(std::forward<Args>(args)...)));
    }

    void forward(const std::vector<double> &inputs);
    void backpropagate(const std::vector<double> &targets, size_t iteration = 1);

    void calculateGradient(const std::vector<double> &targets);

    void optimize(size_t iteration = 1);

    void train(const std::vector<double> &inputs, const std::vector<double> &targets, size_t iteration = 1);
    void train(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &targets, size_t epochs = 1);

    double test(const std::vector<double> &inputs, const std::vector<double> &targets);
    double test(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets);

    void initWeights();

    size_t getInputCount() const
    {
        return layers.front().size;
    }
    size_t getOutputCount() const
    {
        return layers.back().size;
    }
    size_t getLayerCount() const
    {
        return layers.size();
    }
    const std::vector<double> &getOutput() const
    {
        return layers.back().activated_neurons;
    }

    template <std::derived_from<Optimizer> T, typename... Args>
    void setOptimizer(Args&&... args)
    {
        optimizer = std::make_shared<T>(*this, std::forward<Args>(args)...);
    }

    void operator()(const std::vector<double> &input)
    {
        forward(input);
    }

    static friend std::ostream &operator<<(std::ostream & os, const NeuralNetwork & net);
    static friend std::istream &operator>>(std::istream & is, NeuralNetwork & net);

public:
    std::vector<Layer> layers;

protected:
    std::shared_ptr<Optimizer> optimizer = nullptr;
};
