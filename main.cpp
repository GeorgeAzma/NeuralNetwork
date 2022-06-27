#include "genetic_algorithm.h"
#include "neat.h"
#include "genome.h"

int main()
{
    // NeuralNetwork net;
    // net.setOptimizer<SGD>(0.0005f, 0.5f, 0.0f, false); //0.001f, 0.9f, 0.0001f
    // net.setCostFunction<CCE>();
    // net.add(DenseLayer<>(784));
    // net.add(DenseLayer<RELU>(32));
    // net.add(DenseLayer<RELU>(32));
    // net.add(DenseLayer<Softmax>(10));
    // net.build();

    // std::vector<std::vector<float>> inputs = loadImages("data/mnist.input");
    // std::vector<std::vector<float>> labels = loadLabels("data/mnist.label");

    // std::vector<std::vector<float>> input_tests = loadImages("data/mnist-test.input");
    // std::vector<std::vector<float>> label_tests = loadLabels("data/mnist-test.label");

    // DebugTimer t;
    // net.train(inputs, labels, 1, 32);
    // t.stop();

    // TestResult result = net.test(input_tests, label_tests);
    // std::cout << "Loss: " << result.loss << " | Accuracy: " << result.accuracy << std::endl;
    //
    // std::ofstream save("save.net", std::ios::binary);
    // save << net;

    // GeneticAlgorithm<Dot> algo;
    // algo.setMutationRate(0.05f);

    // std::string target = "George";
    // std::vector<Dot> dots(100, Dot(target));
    // DebugTimer t;
    // for (size_t i = 0; i < 64; ++i)
    // {
    //     for (auto &dot : dots)
    //         dot.update();

    //     algo.update(dots);
    // }
    // t.stop();
    // for (auto &d : dots)
    //     std::cout << d.getValue() << std::endl;
    // std::cout << "Gen: " << algo.getGeneration() << " | Gen Fitness: " << algo.getFitnessSum() << std::endl;
    // std::cout << "Fittest: " << dots.front().getValue() << std::endl;

	Neat neat(3, 3, 100);
	
    Genome g = neat.emptyGenome();
    std::cout << g.getNodes().size() << std::endl;

    return 0;
}