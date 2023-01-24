#include <iostream>
#include <fstream>
#include <chrono>
#include "neural_network.h"
#include "random.h"

class DebugTimer
{
public:
    DebugTimer(const char *name = "Timer", bool showMilliseconds = true)
        : m_name(name), stopped(false), show_millis(showMilliseconds), start_point(std::chrono::high_resolution_clock::now())
    {
    }

    ~DebugTimer()
    {
        if (!stopped)
            stop();
    }

    void stop()
    {
        auto end_point = std::chrono::high_resolution_clock::now();
        long long start = std::chrono::time_point_cast<std::chrono::microseconds>(start_point).time_since_epoch().count();
        long long end = std::chrono::time_point_cast<std::chrono::microseconds>(end_point).time_since_epoch().count();
        long long duration = end - start;
        std::cout << m_name << ": " << duration << "us";
        if (show_millis)
        {
            const float ms = (float)duration * 0.001f;
            std::cout << " (" << ms << "ms)";
        }
        std::cout << '\n';
        stopped = true;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
    const char *m_name;
    bool stopped;
    bool show_millis;
};

static void makeLittleEndian(uint32_t& num)
{
    num = ((num >> 24) & 0xff) |
        ((num << 8) & 0xff0000) |
        ((num >> 8) & 0xff00) |
        ((num << 24) & 0xff000000);
}

static std::vector<double> classToVector(size_t Class, size_t max_size)
{
    //TODO: Assert that Class < max_size;
    std::vector<double> vec(max_size, 0.0f);
    vec[Class] = 1.0f;
    return vec;
}

static size_t vectorToClass(const std::vector<float>& vec)
{
    return size_t(std::max_element(vec.begin(), vec.end()) - vec.begin());
}

template<typename T>
static std::vector<std::vector<T>> splitVector(const std::vector<T>& vec, size_t n)
{
    std::vector<std::vector<T>> out_vec;
    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;
    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < n; ++i)
    {
        end += length + (i < remain);
        out_vec.emplace_back(vec.begin() + begin, vec.begin() + end);
        begin = end;
    }

    return out_vec;
}

static std::vector<std::vector<double>> loadImages(const char* path)
{
    std::ifstream is(path, std::ios::binary);
    uint32_t magic_number = 0;
    is.read((char*)&magic_number, 4); //TODO: Assert that magic number is correct
    makeLittleEndian(magic_number);
    uint32_t images = 0;
    is.read((char*)&images, 4);
    makeLittleEndian(images);
    uint32_t width = 0;
    is.read((char*)&width, 4);
    makeLittleEndian(width);
    uint32_t height = 0;
    is.read((char*)&height, 4);
    makeLittleEndian(height);
    std::vector<std::vector<uint8_t>> data(images);
    for (auto& image : data)
    {
        image.resize(width * height);
        is.read((char*)image.data(), width * height);
    }
    std::vector<std::vector<double>> normalized_data(data.size());
    for (size_t i = 0; i < normalized_data.size(); ++i)
    {
        normalized_data[i].resize(data[i].size());
        for (size_t j = 0; j < width * height; ++j)
            normalized_data[i][j] = (data[i][j] >= 128 ? 1.0 : 0.0);
    }
    return normalized_data;
}

static std::vector<std::vector<double>> loadLabels(const char* path)
{
    std::ifstream is(path, std::ios::binary);
    uint32_t magic_number = 0;
    is.read((char*)&magic_number, 4); //TODO: Assert that magic number is correct
    makeLittleEndian(magic_number);
    uint32_t labels = 0;
    is.read((char*)&labels, 4);
    makeLittleEndian(labels);
    std::vector<uint8_t> data(labels);
    is.read((char*)data.data(), data.size());
    std::vector<std::vector<double>> normalized_data(data.size());
    for (size_t i = 0; i < normalized_data.size(); ++i)
        normalized_data[i] = classToVector(data[i], 10);
    return normalized_data;
}

int main()
{
    NeuralNetwork net;

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> labels;

    std::vector<std::vector<double>> input_tests;
    std::vector<std::vector<double>> label_tests;

    size_t epochs = 1;

#if 0
    net.add(DenseLayer(784));
    net.add(DenseLayer<Relu>(16));
    net.add(DenseLayer<Sigmoid>(10));

    inputs = loadImages("data/mnist.input");
    labels = loadLabels("data/mnist.label");

    input_tests = loadImages("data/mnist-test.input");
    label_tests = loadLabels("data/mnist-test.label");
#else
    epochs = 5000;

    net.add(2);
    net.add<Relu>(2);
    net.add<Linear>(1);

    inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    labels = { { 0 }, { 1 }, { 1 }, { 0 } };

    for (size_t i = 0; i < 1000; ++i)
    {
        bool x = Random::Bool();
        bool y = Random::Bool();
        input_tests.push_back({ (double)x, (double)y });
        label_tests.push_back({ double(x ^ y) });
    }
#endif
        //-0.68, 0.68     -0.84,  0.84
        //-0.1, -0.02      0.84, -0.84

        // 0.97, 0.08      1.18,  1.18

    net.initWeights();

    net.setOptimizer<Gd>(0.101);

    DebugTimer t;
    net.train(inputs, labels, epochs);
    t.stop();

    double result = net.test(input_tests, label_tests);
    std::cout << "Loss: " << result << '\n';



    return 0;
}