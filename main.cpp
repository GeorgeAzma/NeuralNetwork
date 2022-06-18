

struct ActivationFunction
{
	enum class Type : uint8_t
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
	ActivationFunction(Type type) 
	: type(type) {}

	virtual std::vector<float> operator()(const std::vector<float>& inputs) const = 0;
	virtual std::vector<float> derivative(const std::vector<float>& inputs) const = 0;
	
	Type getType() const { return type; }

	virtual void save(std::ostream& os) const {}
	virtual void load(std::istream& is) {}

	static friend std::ostream& operator<<(std::ostream& os, const ActivationFunction& activation_function)
	{
		activation_function.save(os);
		return os;
	}
	static friend std::istream& operator>>(std::istream& is, ActivationFunction& activation_function)
	{
		activation_function.load(is);
		return is;
	}

private:
	Type type;
};
		
struct RELU : public ActivationFunction
{
	RELU() 
	: ActivationFunction(Type::RELU) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = a * (a >= 0);
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = a >= 0;
		return activations; 
	}
};

struct LRELU : public ActivationFunction
{
	LRELU(float scale = 0.01f) 
	: scale(scale), ActivationFunction(Type::LRELU) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = (a < 0) ? (-scale * a) : a;
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = a >= 0.0f ? 1.0f : scale;
		return activations; 
	}
	
	void save(std::ostream& os) const 
	{
		os.write((const char*)&scale, sizeof(scale));
	}
	virtual void load(std::istream& is) 
	{
		is.read((char*)&scale, sizeof(scale));
	}

public:
	float scale;
};

struct Sigmoid : public ActivationFunction
{
	Sigmoid() 
	: ActivationFunction(Type::SIGMOID) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = 1.0f / (1.0f + std::exp(-a));
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
		{
			float s = 1.0f / (1.0f + std::exp(-a));
			a = s * (1.0f - s);
		}
		return activations; 
	}
};

struct Arctan : public ActivationFunction
{
	Arctan() 
	: ActivationFunction(Type::ARCTAN) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = std::atan(a);
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = 1.0f / (a * a + 1.0f);
		return activations; 
	}
};

struct Tanh : public ActivationFunction
{
	Tanh() 
	: ActivationFunction(Type::TANH) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = std::tanh(a);
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
		{
			float t = std::tanh(a);
			a = 1.0f - t * t;
		}
		return activations; 
	}
};

struct Step : public ActivationFunction
{
	Step() 
	: ActivationFunction(Type::STEP) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = a <= 0.0f ? -1.0f : 1.0f;
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		return std::vector<float>(inputs.size(), 0.0f); 
	}
};

struct Linear : public ActivationFunction
{
	Linear() 
	: ActivationFunction(Type::LINEAR) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		return inputs;
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		return std::vector<float>(inputs.size(), 1.0f); 
	}
};

struct Softmax : public ActivationFunction
{
	Softmax() 
	: ActivationFunction(Type::SOFTMAX) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs;
		//For avoiding overflow
		float max = *std::max_element(activations.begin(), activations.end());
		for(auto& a : activations)
		{
			a -= max;
			a = std::exp(a);
		}
		float isum = 1.0f / std::accumulate(activations.begin(), activations.end(), 0.0f);
		for(auto& a : activations)
			a *= isum;
		return activations;
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		return std::vector<float>(inputs.size(), 1.0f);
	}
};

struct ELU : public ActivationFunction
{
	ELU(float scale = 0.1f) 
	: scale(scale), ActivationFunction(Type::ELU) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = a >= 0.0f ? a : scale * (std::exp(a) - 1.0f);
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = a >= 0.0f ? 1.0f : scale * std::exp(a);
		return activations; 
	}
	
	void save(std::ostream& os) const 
	{
		os.write((const char*)&scale, sizeof(scale));
	}
	virtual void load(std::istream& is) 
	{
		is.read((char*)&scale, sizeof(scale));
	}

public:
	float scale;
};

struct Swish : public ActivationFunction
{
	Swish() 
	: ActivationFunction(Type::SWISH) {}

	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = a / (1.0f + std::exp(-a));
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
		{
			float e = std::exp(a) + 1.0f;
			a = (e - 1.0f) * (e + a) / (e * e);
		}
		return activations; 
	}
};

struct Softplus : public ActivationFunction
{
	Softplus() 
	: ActivationFunction(Type::SOFTPLUS) {}
	
	std::vector<float> operator()(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = std::log(1.0f + std::exp(a));
		return activations; 
	}
	std::vector<float> derivative(const std::vector<float>& inputs) const override
	{
		std::vector<float> activations = inputs; 
		for(auto& a : activations) 
			a = 1.0f / (1.0f + std::exp(-a));
		return activations; 
	}
};

struct CostFunction
{
	enum class Type : uint8_t
	{
		MSE,
		CCE,
	};

public:
	CostFunction(Type type) 
	: type(type) {}
	
	virtual float operator()(const std::vector<float>& targets, const std::vector<float>& outputs) const = 0;
	
	Type getType() const { return type; }

private:
	Type type;
};

struct MSE : public CostFunction //Mean squared error
{
	MSE() 
	: CostFunction(Type::MSE) {}

	float operator()(const std::vector<float>& targets, const std::vector<float>& outputs) const
	{
		float loss = 0.0f;
		for(size_t i = 0; i < outputs.size(); ++i)
			loss += (targets[i] - outputs[i]) * (targets[i] - outputs[i]);
		loss /= outputs.size();
		return loss;
	}
};

struct CCE : public CostFunction //Categorical cross entropy
{
	CCE() 
	: CostFunction(Type::CCE) {}

	float operator()(const std::vector<float>& targets, const std::vector<float>& outputs) const
	{
		float loss = 0.0f;
		for(size_t i = 0; i < targets.size(); ++i)
			loss += std::log(std::clamp(outputs[i], 1e-7f, 1.0f - 1e-7f)) * targets[i];
		loss *= -1.0f;
		return loss;
	}
};

struct Optimizer
{
	enum class Type : uint8_t
	{
		SGD
	};

public:
	Optimizer(Type type = Type::SGD) 
	: type(type) {}

	virtual void optimize(
		std::vector<std::vector<std::vector<float>>>& weights, 
		const std::vector<std::vector<std::vector<float>>>& delta_weights,
		std::vector<std::vector<std::vector<float>>>& weight_velocities,
		std::vector<std::vector<float>>& biases,
		const std::vector<std::vector<float>>& delta_biases,
		std::vector<std::vector<float>>& bias_velocities, size_t iteration = 0) const = 0;
	
	Type getType() const { return type; }
	
	virtual void save(std::ostream& os) const {}
	virtual void load(std::istream& is) {}

	static friend std::ostream& operator<<(std::ostream& os, const Optimizer& optimizer)
	{
		optimizer.save(os);
		return os;
	}
	static friend std::istream& operator>>(std::istream& is, Optimizer& optimizer)
	{
		optimizer.load(is);
		return is;
	}

private:
	Type type;
};

struct SGD : public Optimizer
{
	SGD(float learning_rate = 0.001f, float momentum = 0.0f, float decay = 0.0f, bool nesterov = false) 
	: learning_rate(learning_rate), momentum(momentum), decay(decay), nesterov(nesterov), Optimizer(Type::SGD)
	{}

	void optimize(
		std::vector<std::vector<std::vector<float>>>& weights, 
		const std::vector<std::vector<std::vector<float>>>& delta_weights,
		std::vector<std::vector<std::vector<float>>>& weight_velocities,
		std::vector<std::vector<float>>& biases,
		const std::vector<std::vector<float>>& delta_biases,
		std::vector<std::vector<float>>& bias_velocities, size_t iteration = 0) const override
	{
		float lr = learning_rate * (1.0f / (1.0f + decay * iteration));

		//Update weights
		for(size_t layer = 0; layer < delta_weights.size(); ++layer)
			for(size_t next_neuron = 0; next_neuron < delta_weights[layer].size(); ++next_neuron)
				for(size_t neuron = 0; neuron < delta_weights[layer][next_neuron].size(); ++neuron)
				{
					weight_velocities[layer][next_neuron][neuron] = momentum * weight_velocities[layer][next_neuron][neuron] + lr * delta_weights[layer][next_neuron][neuron];
					if(nesterov)
						weights[layer][next_neuron][neuron] += momentum * weight_velocities[layer][next_neuron][neuron] + lr * delta_weights[layer][next_neuron][neuron];
					else
						weights[layer][next_neuron][neuron] += weight_velocities[layer][next_neuron][neuron];
				}
		//Update biases
		for(size_t layer = 0; layer < delta_biases.size(); ++layer)
			for(size_t neuron = 0; neuron < delta_biases[layer].size(); ++neuron)
			{
				bias_velocities[layer][neuron] = momentum * bias_velocities[layer][neuron] + lr * delta_biases[layer][neuron];
				if(nesterov)
					biases[layer][neuron] += bias_velocities[layer][neuron];
				else
					biases[layer][neuron] += momentum * bias_velocities[layer][neuron] + lr * delta_biases[layer][neuron];
			}
	}

	void save(std::ostream& os) const override
	{
		os.write((const char*)&learning_rate, sizeof(learning_rate));
		os.write((const char*)&momentum, sizeof(momentum));
		os.write((const char*)&decay, sizeof(decay));
		os.write((const char*)&nesterov, sizeof(nesterov));
	}
	void load(std::istream& is) override
	{
		is.read((char*)&learning_rate, sizeof(learning_rate));
		is.read((char*)&momentum, sizeof(momentum));
		is.read((char*)&decay, sizeof(decay));
		is.read((char*)&nesterov, sizeof(nesterov));
	}

	float learning_rate;
	float momentum;
	float decay;
	bool nesterov;
};

template<typename T = RELU, typename... Args>
struct DenseLayer
{
	DenseLayer(size_t size, Args&&... args) 
	: size(size), activation_function(std::make_shared<T>(std::forward<Args>(args)...)) {}

	std::shared_ptr<ActivationFunction> activation_function = nullptr;
	size_t size;
};

struct TestResult
{
	float accuracy = 0.0f;
	float loss = 0.0f;
};

class NeuralNetwork
{
public:
	NeuralNetwork() = default;

	template<typename T = RELU, typename... Args>
	void add(const DenseLayer<T, Args...>& layer)
	{
		topology.emplace_back(layer.size);
		activation_functions.emplace_back(layer.activation_function);
	}
	void build()
	{
		if(!cost_function.get()) 
			cost_function = std::make_unique<MSE>();

		if(!optimizer.get()) 
			optimizer = std::make_unique<SGD>();

		//TODO: Assert that topology.size() >= 2;
		neurons.resize(topology.size());
		activated_neurons.resize(neurons.size());
		neuron_errors.resize(neurons.size());
		biases.resize(topology.size() - 1);
		delta_biases.resize(biases.size());
		bias_velocities.resize(biases.size());
		weights.resize(topology.size() - 1);
		delta_weights.resize(weights.size());
		weight_velocities.resize(weights.size());

		for(size_t layer = 0; layer < neurons.size(); ++layer)
		{
			neurons[layer].resize(topology[layer]);
			activated_neurons[layer].resize(neurons[layer].size());
			neuron_errors[layer].resize(neurons[layer].size());
		}

		for(size_t layer = 0; layer < weights.size(); ++layer)
		{
			size_t next_layer = layer + 1;
			weights[layer].resize(topology[next_layer]);
			delta_weights[layer].resize(weights[layer].size());
			weight_velocities[layer].resize(weights[layer].size());

			for(size_t next_neuron = 0; next_neuron < topology[next_layer]; ++next_neuron)
			{
				weights[layer][next_neuron].resize(neurons[layer].size());
				delta_weights[layer][next_neuron].resize(weights[layer][next_neuron].size());
				weight_velocities[layer][next_neuron].resize(weights[layer][next_neuron].size());
				
				//Random weights calculation	
				for(auto& w : weights[layer][next_neuron])
					w = (RNG::Float() * 2.0f - 1.0f) * 0.1f;
			}
			
			biases[layer].resize(neurons[next_layer].size());
			delta_biases[layer].resize(biases[layer].size());
			bias_velocities[layer].resize(biases[layer].size());
		}

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
			activated_neurons[next_layer] = (*activation_functions[next_layer])(neurons[next_layer]);
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
			neuron_errors.back()[neuron] = labels[neuron] - activated_neurons.back()[neuron];
	
		//Calculate all layer errors based on output errors
		for (size_t ri = 0; ri < topology.size() - 2; ++ri) //-1 because we already calculated output layer and another - 1 because input layer does not have errors
		{
			size_t layer = topology.size() - 2 - ri;
			size_t next_layer = layer + 1;
			std::vector<float> derivative = activation_functions[layer]->derivative(neurons[layer]);
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
	void update(size_t iteration = 0)
	{
		optimizer->optimize(weights, delta_weights, weight_velocities, biases, delta_biases, bias_velocities, iteration);

		for(auto& delta_weight : delta_weights)
			for(auto& dw : delta_weight)
				for(auto& w : dw)
					w = 0.0f;

		for(auto& delta_bias : delta_biases)
			for(auto& db : delta_bias)
				db = 0.0f;
	}

	void train(const std::vector<float>& inputs, const std::vector<float>& labels)
	{
		forward(inputs);
		backpropagate(labels);
	}	
	void train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels, size_t epochs = 1, size_t minibatches = 1)
	{
		RNG r;
		r.seed = time(NULL);
		size_t iteration = 0;
		for(size_t epoch = 0; epoch < epochs; ++epoch)
		{
			auto last_seed = r.seed;
			std::shuffle(inputs.begin(), inputs.end(), r);  
			r.seed = last_seed;
			std::shuffle(labels.begin(), labels.end(), r); 

			auto input_batches = splitVector(inputs, inputs.size() / minibatches);
			auto label_batches = splitVector(labels, labels.size() / minibatches);

			for(size_t i = 0; i < input_batches.size(); ++i)
			{
				for(size_t j = 0; j < input_batches[i].size(); ++j)
				{	
					forward(input_batches[i][j]);
					calculateGradient(label_batches[i][j]);
				}
				update(iteration++);
			}
		}
	}
	std::vector<TestResult> train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& labels, size_t epochs, size_t minibatches, size_t batches)
	{
		std::vector<TestResult> results(epochs * batches);
		auto input_batches = splitVector(inputs, batches);
		auto label_batches = splitVector(labels, batches);
		
		for(size_t epoch = 0; epoch < epochs; ++epoch)
		{
			for(size_t batch = 0; batch < input_batches.size(); ++batch)
			{
				results[epoch * batches + batch] = test(input_batches[batch], label_batches[batch]);
				train(input_batches[batch], label_batches[batch], 1, minibatches);
			}
		}

		return results;
	}

	TestResult test(const std::vector<float>& inputs, const std::vector<float>& labels)
	{
		forward(inputs);
		TestResult result{};
		result.accuracy = isCorrect(labels);
		result.loss = calculateLoss(labels);
		return result;
	}
	TestResult test(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& labels)
	{
		TestResult result{};
		for(size_t i = 0; i < inputs.size(); ++i)
		{
			TestResult test_result = test(inputs[i], labels[i]);
			result.accuracy += test_result.accuracy;
			result.loss += test_result.loss;
		}
		result.accuracy /= inputs.size();
		result.loss /= inputs.size();
		return result;
	}

	void reset()
	{
		for(size_t l = 0; l < neurons.size(); ++l)
		{
			for(size_t n = 0; n < neurons[l].size(); ++n)
			{
				neurons[l][n] = 0.0f;
				activated_neurons[l][n] = 0.0f;
				neuron_errors[l][n] = 0.0f;
			}
		}
		for(size_t l = 0; l < topology.size() - 1; ++l)
		{
			for(size_t nn = 0; nn < neurons[l + 1].size(); ++nn)
			{
				biases[l][nn] = 0.0f;
				delta_biases[l][nn] = 0.0f;
				bias_velocities[l][nn] = 0.0f;
				for(size_t n = 0; n < neurons[l].size(); ++n)
				{
					weights[l][nn][n] = (RNG::Float() * 2.0f - 1.0f) * 0.1f;
					delta_weights[l][nn][n] = 0.0f;
					weight_velocities[l][nn][n] = 0.0f;
				}
			}
		}
	}
	
	bool isCorrect(const std::vector<float>& labels) const { return vectorToClass(getOutput()) == vectorToClass(labels); }
	size_t getInputCount() const { return topology.front(); }
	size_t getOutputCount() const { return topology.back(); }
	float calculateLoss(const std::vector<float>& labels) { return (*cost_function)(labels, getOutput()); }
	const std::vector<float>& getOutput() const { return activated_neurons.back(); }

	template<typename T, typename... Args> 
	void setCostFunction(Args... args) { cost_function = std::make_unique<T>(std::forward<Args>(args)...); }
	template<typename T, typename... Args>
	void setOptimizer(Args... args) { optimizer = std::make_unique<T>(std::forward<Args>(args)...); }

	void operator()(const std::vector<float>& input) { forward(input); }	
	static friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net)
	{
		size_t topology_size = net.topology.size();
		os.write((const char*)&topology_size, sizeof(topology_size));
		for(const auto& t : net.topology)
			os.write((const char*)&t, sizeof(t));

		for(const auto& a : net.activation_functions)
		{
			ActivationFunction::Type activation_function_type = a->getType();
			os.write((const char*)&activation_function_type, sizeof(activation_function_type));
			os << *a;
		}
		CostFunction::Type cost_function_type = net.cost_function->getType();
		os.write((const char*)&cost_function_type, sizeof(cost_function_type));
		Optimizer::Type optimizer_type = net.optimizer->getType();
		os.write((const char*)&optimizer_type, sizeof(optimizer_type));
		os << *net.optimizer;

		for(const auto& weights : net.weights)
			for(const auto& weight : weights)
				for(const auto& w : weight)
					os.write((const char*)&w, sizeof(w));

		for(const auto& biases : net.biases)
			for(const auto& b : biases)
				os.write((const char*)&b, sizeof(b));

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
			ActivationFunction::Type activation_function_type;
			is.read((char*)&activation_function_type, sizeof(activation_function_type));
			switch(activation_function_type)
			{
			case ActivationFunction::Type::RELU:
				net.activation_functions[i] = std::make_shared<RELU>();
				break;
			case ActivationFunction::Type::LRELU:
				net.activation_functions[i] = std::make_shared<LRELU>();
				break;
			case ActivationFunction::Type::SIGMOID:
				net.activation_functions[i] = std::make_shared<Sigmoid>();
				break;
			case ActivationFunction::Type::ARCTAN:
				net.activation_functions[i] = std::make_shared<Arctan>();
				break;
			case ActivationFunction::Type::TANH:
				net.activation_functions[i] = std::make_shared<Tanh>();
				break;
			case ActivationFunction::Type::STEP:
				net.activation_functions[i] = std::make_shared<Step>();
				break;
			case ActivationFunction::Type::LINEAR:
				net.activation_functions[i] = std::make_shared<Linear>();
				break;
			case ActivationFunction::Type::SOFTMAX:
				net.activation_functions[i] = std::make_shared<Softmax>();
				break;
			case ActivationFunction::Type::ELU:
				net.activation_functions[i] = std::make_shared<ELU>();
				break;
			case ActivationFunction::Type::SWISH:
				net.activation_functions[i] = std::make_shared<Swish>();
				break;
			case ActivationFunction::Type::SOFTPLUS:
				net.activation_functions[i] = std::make_shared<Softplus>();
				break;
			}
			is >> *net.activation_functions[i];
		}
		
		CostFunction::Type cost_function_type;
		is.read((char*)&cost_function_type, sizeof(cost_function_type));
		switch(cost_function_type)
		{
		case CostFunction::Type::MSE:
			net.cost_function = std::make_unique<MSE>();
			break;
		case CostFunction::Type::CCE:
			net.cost_function = std::make_unique<CCE>();
			break;
		}
		//is >> *net.cost_function;
		Optimizer::Type optimizer_type;
		is.read((char*)&optimizer_type, sizeof(optimizer_type));
		switch(optimizer_type)
		{
		case Optimizer::Type::SGD:
			net.optimizer = std::make_unique<SGD>();
			break;
		}
		is >> *net.optimizer;
		net.build();

		for(const auto& weights : net.weights)
			for(const auto& weight : weights)
				for(const auto& w : weight)
					is.read((char*)&w, sizeof(w));

		for(const auto& biases : net.biases)
			for(const auto& b : biases)
				is.read((char*)&b, sizeof(b));

		return is;
	}

private:
	std::vector<size_t> topology;
	std::vector<std::vector<float>> neurons;
	std::vector<std::vector<float>> activated_neurons;
	std::vector<std::vector<float>> neuron_errors;
	std::vector<std::vector<float>> biases;
	std::vector<std::vector<float>> delta_biases;
	std::vector<std::vector<float>> bias_velocities;
	std::vector<std::vector<std::vector<float>>> weights; // [Layer][Neuron][Weight coming from previous neuron layer neurons to this neuron]
	std::vector<std::vector<std::vector<float>>> delta_weights;
	std::vector<std::vector<std::vector<float>>> weight_velocities;
	std::vector<std::shared_ptr<ActivationFunction>> activation_functions;
	std::unique_ptr<CostFunction> cost_function = nullptr;
	std::unique_ptr<Optimizer> optimizer = nullptr;
};

int main()
{
	NeuralNetwork net;
	net.setOptimizer<SGD>(0.001f, 0.9f, 0.000002f, false); //0.001f, 0.9f, 0.0001f
	net.setCostFunction<CCE>();
	net.add(DenseLayer<>(784));
	net.add(DenseLayer<RELU>(30));
	net.add(DenseLayer<Softmax>(10));
	net.build();

	std::vector<std::vector<float>> inputs = loadImages("data/mnist.input");
	std::vector<std::vector<float>> labels = loadLabels("data/mnist.label");

	std::ofstream plot("plot.csv");
	plot << "Index,Data,Type\n";

	DebugTimer t;
	auto results = net.train(inputs, labels, 15, 16, 64);
	t.stop();
	
	size_t index = 0;
	for(const auto& result : results)
	{
		plot << index << ',' << result.accuracy << ",A\n";
		plot << index << ',' << result.loss << ",L\n";
		++index;
	}

	TestResult result = net.test(inputs, labels);
	std::cout << "Loss: " << result.loss << " | Accuracy: " << result.accuracy << std::endl;

	std::ofstream save("saveGood.net", std::ios::binary);
	save << net;


	return 0;
}