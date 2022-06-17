enum class ActivationFunctions : size_t
{
	RELU = 0,
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

struct ActivationFunction
{
	ActivationFunctions type;
	std::vector<float>(*activate)(const std::vector<float>&) = nullptr;
	std::vector<float>(*derivate)(const std::vector<float>&) = nullptr;
};

enum class CostFunctions
{
	MSE, //Mean squared error
	CCE, //Categorical cross entropy, Use this with softmax activation function for outputs
};

struct CostFunction
{
	CostFunctions type;

	CostFunction& operator=(float(*func)(const std::vector<float>&, const std::vector<float>&))
	{
		this->func = func;
		return *this;
	}
	float operator()(const std::vector<float>& target, const std::vector<float>& output) const 
	{ 
		return func(target, output); 
	}
	operator bool() const { return func; }

private:
	float(*func)(const std::vector<float>&, const std::vector<float>&) = nullptr;
};

struct DenseLayer
{
	DenseLayer(size_t size, ActivationFunctions activation = ActivationFunctions::RELU) 
	: size(size), activation(activation) {}

	ActivationFunctions activation;
	size_t size;
};

class NeuralNetwork
{
public:
	NeuralNetwork() = default;

	void setCostFunction(CostFunctions cost_function)
	{
		this->cost_function = toCostFunction(cost_function);
	}
	void add(const DenseLayer& layer)
	{
		topology.emplace_back(layer.size);
		activation_functions.emplace_back(toActivationFunction(layer.activation));
	}
	void build()
	{
		if(!cost_function) 
			cost_function = toCostFunction(CostFunctions::MSE);

		//TODO: Assert that topology.size() >= 2;
		neurons.resize(topology.size());
		activated_neurons.resize(neurons.size());
		neuron_errors.resize(neurons.size());
		biases.resize(topology.size() - 1);
		delta_biases.resize(biases.size());
		weights.resize(topology.size() - 1);
		delta_weights.resize(weights.size());

		for(size_t layer = 0; layer < neurons.size(); ++layer)
		{
			neurons[layer].resize(topology[layer]);
			activated_neurons[layer].resize(neurons[layer].size());
			neuron_errors[layer].resize(neurons[layer].size());
		}

		for(size_t layer = 0; layer < weights.size(); ++layer)
		{
			weights[layer].resize(topology[layer + 1]);
			delta_weights[layer].resize(weights[layer].size());
		}

		for(size_t layer = 0; layer < biases.size(); ++layer)
		{
			biases[layer].resize(neurons[layer + 1].size());
			delta_biases[layer].resize(biases[layer].size());
		}

		for(size_t layer = 0; layer < weights.size(); ++layer)
		{
			size_t next_layer = layer + 1;
			for(size_t next_neuron = 0; next_neuron < topology[next_layer]; ++next_neuron)
			{
				weights[layer][next_neuron].resize(neurons[layer].size());
				delta_weights[layer][next_neuron].resize(weights[layer][next_neuron].size());
			}
		}

		//Random weights calculation
		for(auto& lw : weights)
			for(auto& nw : lw)
				for(auto& w : nw)
					w = (RNG::Float() * 2.0f - 1.0f) * 0.1f;
	}

	void forward(const std::vector<float>& inputs)
	{
		//TODO: Assert that input.size() == inputCount();
		activated_neurons.front() = inputs;

		for(size_t layer = 0; layer < weights.size(); ++layer)
		{
			size_t next_layer = layer + 1;
			for(size_t next_neuron = 0; next_neuron < topology[next_layer]; ++next_neuron)
				neurons[next_layer][next_neuron] = std::inner_product(activated_neurons[layer].begin(), activated_neurons[layer].end(), weights[layer][next_neuron].begin(), biases[layer][next_neuron]);
			activated_neurons[next_layer] = activation_functions[next_layer].activate(neurons[next_layer]);
		}
	}
	void backpropagate(const std::vector<float>& labels)
	{
		calculateGradient(labels);
		update();
	}

	void calculateGradient(const std::vector<float>& labels)
	{
		//Calculate output errors
		for (size_t neuron = 0; neuron < getOutputCount(); ++neuron)
			neuron_errors.back()[neuron] = labels[neuron] - getOutput()[neuron];
	
		//Calculate all layer errors based on output errors
		for (size_t ri = 0; ri < topology.size() - 2; ++ri) //-1 because we already calculated output layer and another - 1 because input layer does not have errors
		{
			size_t layer = topology.size() - 2 - ri;
			size_t next_layer = layer + 1;
			std::vector<float> derivative = activation_functions[layer].derivate(neurons[layer]);
			for (size_t neuron = 0; neuron < topology[layer]; ++neuron)
			{
				neuron_errors[layer][neuron] = 0.0f;
				for (size_t next_neuron = 0; next_neuron < topology[next_layer]; ++next_neuron)
					neuron_errors[layer][neuron] += neuron_errors[next_layer][next_neuron] * derivative[neuron] * weights[layer][next_neuron][neuron];
			}
		}

		//Calculate gradient
		for(size_t layer = 0; layer < weights.size(); ++layer)
		{
			//Calculate delta weights
			size_t next_layer = layer + 1;
			for(size_t next_neuron = 0; next_neuron < topology[next_layer]; ++next_neuron)
				for(size_t neuron = 0; neuron < activated_neurons[layer].size(); ++neuron)
					delta_weights[layer][next_neuron][neuron] += neuron_errors[next_layer][next_neuron] * activated_neurons[layer][neuron];
			
			//Calculate delta biases
			for(size_t next_neuron = 0; next_neuron < topology[next_layer]; ++next_neuron)
				delta_biases[layer][next_neuron] += neuron_errors[next_layer][next_neuron];
		}
	}
	void update()
	{
		//Update weights
		for(size_t layer = 0; layer < delta_weights.size(); ++layer)
		{
			for(size_t next_neuron = 0; next_neuron < delta_weights[layer].size(); ++next_neuron)
			{
				for(size_t neuron = 0; neuron < delta_weights[layer][next_neuron].size(); ++neuron)
				{
					weights[layer][next_neuron][neuron] += learning_rate * delta_weights[layer][next_neuron][neuron];
					delta_weights[layer][next_neuron][neuron] = 0.0f;
				}
			}
		}

		//Update biases
		for(size_t layer = 0; layer < delta_biases.size(); ++layer)
		{
			for(size_t neuron = 0; neuron < delta_biases[layer].size(); ++neuron)
			{
				biases[layer][neuron] += learning_rate * delta_biases[layer][neuron];
				delta_biases[layer][neuron] = 0.0f;
			}
		}
	}

	void train(const std::vector<float>& input, const std::vector<float>& target)
	{
		forward(input);
		backpropagate(target);
	}	
	void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, size_t epochs = 1)
	{
		std::vector<size_t> indices(inputs.size());
		std::iota(indices.begin(), indices.end(), (size_t)0);
		for(size_t epoch = 0; epoch < epochs; ++epoch)
		{  
			RNG rd;
			std::shuffle(indices.begin(), indices.end(), rd);
			for(size_t i = 0; i < inputs.size(); ++i)
				train(inputs[indices[i]], targets[indices[i]]);
		}
	}
	
	size_t getInputCount() const { return topology.front(); }
	size_t getOutputCount() const { return topology.back(); }
	float calculateLoss(const std::vector<float>& target) { return cost_function(target, getOutput()); }
	const std::vector<float>& getOutput() const { return activated_neurons.back(); }
	
	void setLearningRate(float learning_rate) { this->learning_rate = learning_rate; }

	void operator()(const std::vector<float>& input) { forward(input); }	
	static friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net)
	{
		size_t topology_size = net.topology.size();
		os.write((const char*)&topology_size, sizeof(topology_size));
		for(size_t layer = 0; layer < net.topology.size(); ++layer)
			os.write((const char*)&net.topology[layer], sizeof(net.topology[layer]));

		for(size_t i = 0; i < net.topology.size(); ++i)
			os.write((const char*)&net.activation_functions[i].type, sizeof(net.activation_functions[i].type));
		os.write((const char*)&net.cost_function.type, sizeof(net.cost_function.type));
		os.write((const char*)&net.learning_rate, sizeof(net.learning_rate));

		for(size_t layer = 0; layer < net.weights.size(); ++layer)
			for(size_t neuron = 0; neuron < net.weights[layer].size(); ++neuron)
				for(size_t weight = 0; weight < net.weights[layer][neuron].size(); ++weight)
					os.write((const char*)&net.weights[layer][neuron][weight], sizeof(net.weights[layer][neuron][weight]));

		for(size_t layer = 0; layer < net.biases.size(); ++layer)
			for(size_t neuron = 0; neuron < net.biases[layer].size(); ++neuron)
				os.write((const char*)&net.biases[layer][neuron], sizeof(net.biases[layer][neuron]));

		return os;
	}
	static friend std::istream& operator>>(std::istream& is, NeuralNetwork& net)
	{
		size_t topology_size = 0;
		is.read((char*)&topology_size, sizeof(topology_size));
		net.topology.resize(topology_size);

		size_t i = 0;
		for(size_t i = 0; i < net.topology.size(); ++i)
			is.read((char*)&net.topology[i], sizeof(net.topology[i]));

		net.activation_functions.resize(net.topology.size());
		for(size_t i = 0; i < net.topology.size(); ++i)
		{
			ActivationFunctions activation;
			is.read((char*)&activation, sizeof(activation));
			net.activation_functions[i] = NeuralNetwork::toActivationFunction(activation);
		}
		
		CostFunctions cost;
		is.read((char*)&cost, sizeof(cost));
		net.cost_function = NeuralNetwork::toCostFunction(cost);
		is.read((char*)&net.learning_rate, sizeof(net.learning_rate));
		net.build();

		for(size_t layer = 0; layer < net.weights.size(); ++layer)
			for(size_t neuron = 0; neuron < net.weights[layer].size(); ++neuron)
				for(size_t weight = 0; weight < net.weights[layer][neuron].size(); ++weight)
					is.read((char*)&net.weights[layer][neuron][weight], sizeof(net.weights[layer][neuron][weight]));

		for(size_t layer = 0; layer < net.biases.size(); ++layer)
			for(size_t neuron = 0; neuron < net.biases[layer].size(); ++neuron)
				is.read((char*)&net.biases[layer][neuron], sizeof(net.biases[layer][neuron]));

		return is;
	}

private:
	static ActivationFunction toActivationFunction(ActivationFunctions activation)
	{
		ActivationFunction activation_function{};
		activation_function.type = activation;

		switch (activation)
		{
		case ActivationFunctions::RELU:
			activation_function.activate = [](const std::vector<float>& input) 
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = a * (a >= 0);
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = a >= 0;
				return activation; 
			};
			break;
		case ActivationFunctions::LRELU:
			activation_function.activate = [](const std::vector<float>& input)
			{
				constexpr float k = -0.01f;
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = (a < 0) ? (k * a) : a;
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				constexpr float k = 0.01f;
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = a >= 0 ? 1 : k;
				return activation; 
			};
			break;
		case ActivationFunctions::SIGMOID:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = 1.0f / (1.0f + std::exp(-a));
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
				{
					float s = 1.0f / (1.0f + std::exp(-a));
					a = s * (1.0f - s);
				}
				return activation; 
			};
			break;
		case ActivationFunctions::ARCTAN:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = std::atan(a);
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = 1.0f / (a * a + 1.0f);
				return activation; 
			};
			break;
		case ActivationFunctions::TANH:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = std::tanh(a);
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
				{
					float t = std::tanh(a);
					a = 1.0f - t * t;
				}
				return activation; 
			};
			break;
		case ActivationFunctions::STEP:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = a <= 0 ? -1 : 1;
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				return std::vector<float>(input.size(), 0.0f); 
			};
			break;
		case ActivationFunctions::LINEAR:
			activation_function.activate = [](const std::vector<float>& input)
			{
				return input;
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				return std::vector<float>(input.size(), 1.0f); 
			};
			break;
		case ActivationFunctions::SOFTMAX:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input;
				//For avoiding overflow
				float max = *std::max_element(activation.begin(), activation.end());
				for(auto& a : activation)
				{
					a -= max;
					a = std::exp(a);
				}
				float isum = 1.0f / std::accumulate(activation.begin(), activation.end(), 0.0f);
				for(auto& a : activation)
					a *= isum;
				return activation;
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				return input;  //TODO:
			};
			break;
		case ActivationFunctions::ELU:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = a >= 0 ? a : 0.1f * (std::exp(a) - 1.0f);
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = a >= 0 ? 1.0f : 0.1f * std::exp(a);
				return activation; 
			};
			break;
		case ActivationFunctions::SWISH:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = a / (1.0f + std::exp(-a));
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
				{
					float e = std::exp(a) + 1;
					a = (e - 1) * (e + a) / (e * e);
				}
				return activation; 
			};
			break;
		case ActivationFunctions::SOFTPLUS:
			activation_function.activate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = std::log(1.0f + std::exp(a));
				return activation; 
			};
			activation_function.derivate = [](const std::vector<float>& input)
			{
				std::vector<float> activation = input; 
				for(auto& a : activation) 
					a = 1.0f / (1.0f + std::exp(-a));
				return activation; 
			};
			break;
		}
		return activation_function;
	}
	static CostFunction toCostFunction(CostFunctions cost)
	{
		CostFunction cost_function{};
		cost_function.type = cost;
		
		switch (cost)
		{
		case CostFunctions::MSE:
			cost_function = [](const std::vector<float>& target, const std::vector<float>& output) 
			{
				float loss = 0.0f;
				for(size_t i = 0; i < output.size(); ++i)
					loss += (target[i] - output[i]) * (target[i] - output[i]);
				loss /= output.size();
				return loss;
			};
			break;
		case CostFunctions::CCE:
			cost_function = [](const std::vector<float>& target, const std::vector<float>& output) 
			{
				float loss = 0.0f;
				for(size_t i = 0; i < target.size(); ++i)
					loss += std::log(std::clamp(output[i], 1e-7f, 1.0f - 1e-7f)) * target[i];
				loss *= -1.0f;
				return loss;
			};
			break;
		}

		return cost_function;
	}

private:
	std::vector<size_t> topology;
	std::vector<std::vector<float>> neurons;
	std::vector<std::vector<float>> activated_neurons;
	std::vector<std::vector<float>> neuron_errors;
	std::vector<std::vector<float>> biases;
	std::vector<std::vector<float>> delta_biases;
	std::vector<std::vector<std::vector<float>>> weights; // [Layer][Neuron][Weight coming from previous neuron layer neurons to this neuron]
	std::vector<std::vector<std::vector<float>>> delta_weights;
	std::vector<ActivationFunction> activation_functions;
	CostFunction cost_function;
	float learning_rate = 0.01f;
};

int main()
{
	NeuralNetwork net;
	net.setLearningRate(0.01f);
	net.setCostFunction(CostFunctions::CCE);
	net.add(DenseLayer(784));
	net.add(DenseLayer(16, ActivationFunctions::RELU));
	net.add(DenseLayer(10, ActivationFunctions::SOFTMAX));
	net.build();

	std::vector<std::vector<float>> inputs = loadImages("mnist.input");
	std::vector<std::vector<float>> labels = loadLabels("mnist.label");

	DebugTimer t;
	net.train(inputs, labels);
	t.stop();
	
	for(size_t i = 0; i < 8; ++i)
	{
		{
			std::ofstream save("save.net", std::ios::binary);
			save << net;
		}
		{
			std::ifstream save("save.net", std::ios::binary);
			save >> net;
		}
	}
	
	float accuracy = 0.0f;
	float loss = 0.0f;
	size_t tests = 10000;
	for(size_t i = 0; i < tests; ++i)
	{
		size_t rand = RNG::Uint() % inputs.size();
		net.forward(inputs[rand]);
		accuracy += vectorToClass(net.getOutput()) == vectorToClass(labels[rand]);
		loss += net.calculateLoss(labels[rand]);
	}
	accuracy /= tests;
	loss /= tests;
	std::cout << "Accuracy: " << (accuracy * 100) << '%' << std::endl;
	std::cout << "Loss: " << loss << std::endl;
	return 0;
}