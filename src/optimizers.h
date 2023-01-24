#pragma once

#include <vector>
#include <iostream>

class NeuralNetwork;
class Layer;

struct Optimizer
{
    enum class Type: uint8_t
    {
        GD,
        SGD,
        ADAM
    };

public:
    Optimizer(NeuralNetwork& net, Type type = Type::GD, double learning_rate = 0.001)
        : net(net), type(type), learning_rate(learning_rate)
    {
    }

    virtual void operator()(size_t iteration = 0) = 0;

    virtual void reset() {}

    Type getType() const
    {
        return type;
    }

    void save(std::ostream &os) const
    {
        os.write((const char *)&learning_rate, sizeof(learning_rate));
        saveData(os);
    }
    void load(std::istream &is)
    {
        is.read((char *)&learning_rate, sizeof(learning_rate));
        loadData(is);
    }

    static friend std::ostream &operator<<(std::ostream &os, const Optimizer &optimizer)
    {
        optimizer.save(os);
        return os;
    }
    static friend std::istream &operator>>(std::istream &is, Optimizer &optimizer)
    {
        optimizer.load(is);
        return is;
    }

protected:
    virtual void saveData(std::ostream &os) const {}
    virtual void loadData(std::istream &is) {}

protected:
    NeuralNetwork& net;
    const Type type;
    double learning_rate;
};

struct Gd: public Optimizer
{
    Gd(NeuralNetwork& net, double learning_rate = 0.001)
        : Optimizer(net, Type::GD, learning_rate)
    {}

    void operator()(size_t iteration = 0) override;
};

struct Sgd: public Optimizer
{
    Sgd(NeuralNetwork& net, double learning_rate = 0.001, double momentum = 0.9);

    void operator()(size_t iteration = 0) override;

    void reset() override;

    void saveData(std::ostream &os) const override
    {
        os.write((const char *)&momentum, sizeof(momentum));
    }
    void loadData(std::istream &is) override
    {
        is.read((char *)&momentum, sizeof(momentum));
    }

    std::vector<std::vector<std::vector<double>>> weight_velocities;
    std::vector<std::vector<double>> bias_velocities;
    double momentum;
};

struct Adam: public Optimizer
{
    Adam(NeuralNetwork& net, double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999);

    void operator()(size_t iteration = 0) override;

    void reset() override;

    void saveData(std::ostream &os) const override
    {
        os.write((const char *)&learning_rate, sizeof(learning_rate));
        os.write((const char *)&beta1, sizeof(beta1));
        os.write((const char *)&beta2, sizeof(beta2));
    }
    void loadData(std::istream &is) override
    {
        is.read((char *)&learning_rate, sizeof(learning_rate));
        is.read((char *)&beta1, sizeof(beta1));
        is.read((char *)&beta2, sizeof(beta2));
    }

    std::vector<std::vector<std::vector<double>>> weight_velocities;
    std::vector<std::vector<double>> bias_velocities;
    std::vector<std::vector<std::vector<double>>> square_weight_velocities;
    std::vector<std::vector<double>> square_bias_velocities;
    double beta1;
    double beta2;
};

class OptimizerFactory
{
public:
    static std::shared_ptr<Optimizer> build(Optimizer::Type optimizer_type, NeuralNetwork& net)
    {
        switch (optimizer_type)
        {
        case Optimizer::Type::GD:
            return std::make_shared<Gd>(net);
        case Optimizer::Type::SGD:
            return std::make_shared<Sgd>(net);
        case Optimizer::Type::ADAM:
            return std::make_shared<Adam>(net);
        }
        return nullptr;
    }
};