#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

struct Activation
{
    enum class Type: uint8_t
    {
        RELU,
        LRELU,
        SIGMOID,
        ARCTAN,
        TANH,
        STEP,
        LINEAR,
        SOFTMAX,
        ELU,
        SWISH,
        SOFTPLUS
    };

public:
    Activation(Type type)
        : type(type)
    {}

    virtual double operator()(double x) const { return x; }
    virtual double derivative(double x) const { return 1.0; }
    virtual std::vector<double> operator()(const std::vector<double>& x) const { return {}; }
    virtual std::vector<double> derivative(const std::vector<double>& x) const { return {}; }

    virtual void save(std::ostream &os) const
    {
    }
    virtual void load(std::istream &is)
    {
    }

    static friend std::ostream &operator<<(std::ostream &os, const Activation &activation)
    {
        activation.save(os);
        return os;
    }
    static friend std::istream &operator>>(std::istream &is, Activation &activation)
    {
        activation.load(is);
        return is;
    }

    const Type type;
};

struct Relu: public Activation
{
    Relu()
        : Activation(Type::RELU)
    {}

    double operator()(double x) const override
    {
        return x * (x >= 0);
    }
    double derivative(double x) const override
    {
        return x >= 0;
    }
};

struct Lrelu: public Activation
{
    Lrelu(double scale = 0.01)
        : scale(scale), Activation(Type::LRELU)
    {}

    double operator()(double x) const override
    {
        return (x < 0) ? (-scale * x) : x;
    }
    double derivative(double x) const override
    {
        return x >= 0 ? 1 : scale;
    }

    void save(std::ostream &os) const
    {
        os.write((const char *)&scale, sizeof(scale));
    }
    virtual void load(std::istream &is)
    {
        is.read((char *)&scale, sizeof(scale));
    }

public:
    double scale;
};

struct Sigmoid: public Activation
{
    Sigmoid()
        : Activation(Type::SIGMOID)
    {}

    double operator()(double x) const override
    {
        return 1.0 / (1.0 + exp(-x));
    }
    double derivative(double x) const override
    {
        double s = 1.0 / (1.0 + exp(-x));
        return s * (1.0 - s);
    }
};

struct Arctan: public Activation
{
    Arctan()
        : Activation(Type::ARCTAN)
    {}

    double operator()(double x) const override
    {
        return atan(x);
    }
    double derivative(double x) const override
    {
        return 1.0 / (x * x + 1.0);
    }
};

struct Tanh: public Activation
{
    Tanh()
        : Activation(Type::TANH)
    {}

    double operator()(double x) const override
    {
        return tanh(x);
    }
    double derivative(double x) const override
    {
        double t = tanh(x);
        return 1.0 - t * t;
    }
};

struct Step: public Activation
{
    Step()
        : Activation(Type::STEP)
    {}

    double operator()(double x) const override
    {
        return x < 0.0 ? -1.0 : 1.0;
    }
    double derivative(double x) const override
    {
        return std::signbit(x);
    }
};

struct Linear: public Activation
{
    Linear()
        : Activation(Type::LINEAR)
    {}

    double operator()(double x) const override
    {
        return x;
    }
    double derivative(double x) const override
    {
        return 1.0;
    }
};

struct Softmax: public Activation
{
    Softmax()
        : Activation(Type::SOFTMAX)
    {}

    std::vector<double> operator()(const std::vector<double>& x) const override
    {
        std::vector<double> activations = x;
        // For avoiding overflow
        double max = *std::max_element(activations.begin(), activations.end());
        double sum = 0.0;
        for (auto &x : activations)
        {
            x = exp(x - max);
            sum += x;
        }

        sum = 1.0 / sum;
        for (auto &x : activations)
            x *= sum;
        return activations;
    }
    std::vector<double> derivative(const std::vector<double>& x) const override
    {
        std::vector<double> y = operator()(x);
        std::vector<double> derivative(x.size());
        for (int i = 0; i < x.size(); i++)
            for (int j = 0; j < x.size(); j++)
                derivative[i] += y[j] * (i == j ? (1 - y[j]) : -y[i]);

        return derivative;
    }
};

struct Elu: public Activation
{
    Elu(double scale = 0.1f)
        : scale(scale), Activation(Type::ELU)
    {}

    double operator()(double x) const override
    {
        return x >= 0.0 ? x : scale * (exp(x) - 1.0);
    }
    double derivative(double x) const override
    {
        return x >= 0.0 ? 1.0 : scale * exp(x);
    }

    void save(std::ostream &os) const
    {
        os.write((const char *)&scale, sizeof(scale));
    }
    virtual void load(std::istream &is)
    {
        is.read((char *)&scale, sizeof(scale));
    }

public:
    double scale;
};

struct Swish: public Activation
{
    Swish()
        : Activation(Type::SWISH)
    {}

    double operator()(double x) const override
    {
        return x / (1.0 + exp(-x));
    }
    double derivative(double x) const override
    {
        double e = exp(x) + 1.0;
        return (e - 1.0) * (e + x) / (e * e);
    }
};

struct Softplus: public Activation
{
    Softplus()
        : Activation(Type::SOFTPLUS)
    {}

    double operator()(double x) const override
    {
        return log(1.0 + exp(x));
    }
    double derivative(double x) const override
    {
        return 1.0 / (1.0 + exp(-x));
    }
};

class ActivationFactory
{
public:
    static std::shared_ptr<Activation> build(Activation::Type activation_type)
    {
        switch (activation_type)
        {
        case Activation::Type::RELU:
            return std::make_shared<Relu>();
        case Activation::Type::LRELU:
            return std::make_shared<Lrelu>();
        case Activation::Type::SIGMOID:
            return std::make_shared<Sigmoid>();
        case Activation::Type::ARCTAN:
            return std::make_shared<Arctan>();
        case Activation::Type::TANH:
            return std::make_shared<Tanh>();
        case Activation::Type::STEP:
            return std::make_shared<Step>();
        case Activation::Type::LINEAR:
            return std::make_shared<Linear>();
        case Activation::Type::SOFTMAX:
            return std::make_shared<Softmax>();
        case Activation::Type::ELU:
            return std::make_shared<Elu>();
        case Activation::Type::SWISH:
            return std::make_shared<Swish>();
        case Activation::Type::SOFTPLUS:
            return std::make_shared<Softplus>();
        }
        return nullptr;
    }
};