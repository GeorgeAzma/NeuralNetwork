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
		std::vector<std::vector<float>>& bias_velocities, uint32_t iteration = 0) const = 0;
	
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
		std::vector<std::vector<float>>& bias_velocities, uint32_t iteration = 0) const override
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
	DenseLayer(uint32_t size, Args&&... args) 
	: size(size), activation_function(std::make_shared<T>(std::forward<Args>(args)...)) {}

	std::shared_ptr<ActivationFunction> activation_function = nullptr;
	uint32_t size;
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
			cost_function = std::make_shared<MSE>();

		if(!optimizer.get()) 
			optimizer = std::make_shared<SGD>();

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
			}
			
			biases[layer].resize(neurons[next_layer].size());
			delta_biases[layer].resize(biases[layer].size());
			bias_velocities[layer].resize(biases[layer].size());
		}

		reset();
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
	void update(uint32_t iteration = 0)
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
		uint32_t iteration = 0;
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
	
	bool isCorrect(const std::vector<float>& labels) const 
	{ 
		return (size_t(std::max_element(getOutput().begin(), getOutput().end()) - getOutput().begin()))
		== size_t(std::max_element(labels.begin(), labels.end()) - labels.begin()); 
	}
	uint32_t getInputCount() const { return topology.front(); }
	uint32_t getOutputCount() const { return topology.back(); }
	uint32_t getLayerCount() const { return topology.size(); }
	uint32_t getLayerSize(size_t layer) const { return topology[layer]; }
	float getWeight(size_t layer, size_t neuron, size_t weight) const { return weights[layer][neuron][weight]; }
	const std::vector<std::vector<std::vector<float>>>& getWeights() const { return weights; }
	float getBias(size_t layer, size_t neuron) const { return biases[layer][neuron]; }
	const std::vector<std::vector<float>>& getBiases() const { return biases; }
	float calculateLoss(const std::vector<float>& labels) { return (*cost_function)(labels, getOutput()); }
	const std::vector<float>& getOutput() const { return activated_neurons.back(); }

	template<typename T, typename... Args> 
	void setCostFunction(Args... args) { cost_function = std::make_shared<T>(std::forward<Args>(args)...); }
	template<typename T, typename... Args>
	void setOptimizer(Args... args) { optimizer = std::make_shared<T>(std::forward<Args>(args)...); }
	void setWeight(float value, uint32_t layer, uint32_t neuron, uint32_t weight) { weights[layer][neuron][weight] = value; }
	void setBias(float value, uint32_t layer, uint32_t neuron) { biases[layer][neuron] = value; }
	
	void operator()(const std::vector<float>& input) { forward(input); }	
	static friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& net)
	{
		uint32_t topology_size = net.topology.size();
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
		uint32_t topology_size = 0;
		is.read((char*)&topology_size, sizeof(topology_size));
		net.topology.resize(topology_size);

		for(size_t i = 0; i < net.topology.size(); ++i)
			is.read((char*)&net.topology[i], sizeof(net.topology[i]));

		net.activation_functions.resize(net.topology.size());
		for(size_t i = 0; i < net.activation_functions.size(); ++i)
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
			net.cost_function = std::make_shared<MSE>();
			break;
		case CostFunction::Type::CCE:
			net.cost_function = std::make_shared<CCE>();
			break;
		}
		//is >> *net.cost_function;
		Optimizer::Type optimizer_type;
		is.read((char*)&optimizer_type, sizeof(optimizer_type));
		switch(optimizer_type)
		{
		case Optimizer::Type::SGD:
			net.optimizer = std::make_shared<SGD>();
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
	std::vector<uint32_t> topology;
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
	std::shared_ptr<CostFunction> cost_function = nullptr;
	std::shared_ptr<Optimizer> optimizer = nullptr;
};

class Agent
{
public:
	virtual void calculateFitness() = 0;
	virtual std::unique_ptr<Agent> createBaby(const Agent& other) const = 0;
	virtual void mutate(float mutation_rate) = 0;
	float getFitness() const { return fitness; }
	bool operator>=(const Agent& other) const { return fitness >= other.fitness; }
	bool operator<=(const Agent& other) const { return fitness <= other.fitness; }
	bool operator>(const Agent& other) const { return fitness > other.fitness; }
	bool operator<(const Agent& other) const { return fitness < other.fitness; }

protected:
	float fitness = 0.0f;
};

class Dot : public Agent
{
public:
	Dot() {}
	Dot(const std::pair<float, float>& target) 
	: target(target)
	{
		net.setOptimizer<SGD>();
		net.add(DenseLayer<>(4));
		net.add(DenseLayer<Tanh>(4));
		net.add(DenseLayer<Linear>(2));
		net.build();
	}

	float distance2()
	{
		return (target.first - position.first) * (target.first - position.first)
			 + (target.second - position.second) * (target.second - position.second);
	}

	void update() 
	{ 
		if(finished) return;
		net.forward({ position.first, position.second, target.first, target.second });
		acceleration.first = net.getOutput()[0];
		acceleration.second = net.getOutput()[1];
		velocity.first += acceleration.first;
		velocity.second += acceleration.second;
		velocity.first = std::min(velocity.first, 5.0f);
		velocity.second = std::min(velocity.second, 5.0f);
		position.first += velocity.first;
		position.second += velocity.second;
		if(position.first > 100.0f || position.first < -100.0f
		|| position.second > 100.0f || position.second < -100.0f || distance2() < 0.2f)
			finished = true;
	}

	std::pair<float, float> getValue() const { return position; }
	void calculateFitness() override 
	{ 
		fitness = 1.0f / (0.001f + distance2());
	}
	std::unique_ptr<Agent> createBaby(const Agent& other) const override
	{
		const Dot& other_agent = (const Dot&)other;
		std::unique_ptr<Dot> new_agent = std::make_unique<Dot>(target);
		new_agent->net = net;
		#if 1
		for(size_t i = 0; i < net.getWeights().size(); ++i)
			for(size_t j = 0; j < net.getWeights()[i].size(); ++j)
				for(size_t k = 0; k < net.getWeights()[i][j].size(); ++k)
					new_agent->net.setWeight(RNG::Bool() ? net.getWeight(i, j, k) : other_agent.net.getWeight(i, j, k), i, j, k);

			for(size_t i = 0; i < net.getBiases().size(); ++i)
				for(size_t j = 0; j < net.getBiases()[i].size(); ++j)
					new_agent->net.setBias(RNG::Bool() ? net.getBias(i, j) : other_agent.net.getBias(i, j), i, j);
		#else
			// std::vector<float> flat;
			// for(size_t i = 0; i < net.getWeights().size(); ++i)
			// 	for(size_t j = 0; j < net.getWeights()[i].size(); ++j)
			// 		flat.insert(flat.end(), net.getWeights()[i][j].begin(), net.getWeights()[i][j].end());
	
			
			// std::vector<float> other_flat;
			// for(size_t i = 0; i < net.getWeights().size(); ++i)
			// 	for(size_t j = 0; j < net.getWeights()[i].size(); ++j)
			// 		other_flat.insert(other_flat.end(), other_agent.net.getWeights()[i][j].begin(), other_agent.net.getWeights()[i][j].end());

			// size_t split = RNG::Uint() % flat.size();

			// std::vector<float> joined(flat.size());
			// std::copy(flat.begin(), flat.begin() + split, joined.begin());
			// std::copy(other_flat.begin() + split, other_flat.end(), joined.begin() + split);
			// for(size_t i = 0; i < net.getWeights().size(); ++i)
			// 	for(size_t j = 0; j < net.getWeights()[i].size(); ++j)
			// 		for(size_t k = 0; k < net.getWeights()[i][j].size(); ++k)
			// 			new_agent->net.setWeight();
		#endif
		return new_agent;
	}
	void mutate(float mutation_rate) override
	{
		for(size_t i = 0; i < net.getWeights().size(); ++i)
			for(size_t j = 0; j < net.getWeights()[i].size(); ++j)
				for(size_t k = 0; k < net.getWeights()[i][j].size(); ++k)
					if(RNG::Float() <= mutation_rate)
						net.setWeight(net.getWeight(i, j, k) + (RNG::Float() * 2.0f - 1.0f), i, j, k);
	}

private:
	NeuralNetwork net;
	std::pair<float, float> position = { 0.0f, 0.0f };
	std::pair<float, float> acceleration = { 0.0f, 0.0f };
	std::pair<float, float> velocity = { 0.0f, 0.0f };
	std::pair<float, float> target = { 0.0f, 0.0f };
	bool finished = false;
};

template<typename T>
class GeneticAlgorithm
{
public:
	GeneticAlgorithm() = default;

	void setMutationRate(float mutation_rate = 0.05f)
	{
		this->mutation_rate = mutation_rate;
	}

	void update(std::vector<T>& agents)
	{
		calculateFitness(agents);
		calculateFitnessSum(agents);
		doNaturalSelection(agents);
		mutateAgents(agents);
		++generation;
	}

	uint32_t getGeneration() const { return generation; }
	float getFitnessSum() const { return fitness_sum; }

private:

	void calculateFitness(std::vector<T>& agents)
	{
		for(auto& agent : agents)
			agent.calculateFitness();
	}
	
	void calculateFitnessSum(const std::vector<T>& agents)
	{
		fitness_sum = 0.0f;
		for(const auto& agent : agents)
			fitness_sum += agent.getFitness();
	}

	void doNaturalSelection(std::vector<T>& agents)
	{
		std::vector<T> new_agents = agents;

		for(size_t i = 0; i < agents.size(); ++i)
		{
			const T& mother = selectParent(agents);
			const T* father = &selectParent(agents); 
			for(uint32_t i = 0; (&mother == father) && (i < 8); ++i)
				father = &selectParent(agents);
			const T& new_agent1 = ((const T&)*mother.createBaby(*father));
			const T& new_agent2 = ((const T&)*father->createBaby(mother));
			new_agents.emplace_back(new_agent1);
			new_agents.back().calculateFitness();
			new_agents.emplace_back(new_agent2);
			new_agents.back().calculateFitness();
		}

		std::sort(new_agents.begin(), new_agents.end(), std::greater<>());
		agents = std::vector<T>(new_agents.begin(), new_agents.begin() + agents.size());
	}

	void mutateAgents(std::vector<T>& agents)
	{
		for(auto& agent : agents)
			agent.mutate(mutation_rate);
	}

	const T& selectParent(const std::vector<T>& agents) const
	{
		float rand = RNG::Float() * fitness_sum;
		float running_sum = 0.0f;
		for(const auto& agent : agents)
		{
			running_sum += agent.getFitness();
			if(running_sum >= rand)
				return agent;
		}
		return agents.back();
	}

private:
	std::vector<T> agents = {};
	float mutation_rate = 0.05f;
	float fitness_sum = 0.0f;
	uint32_t generation = 0;
};

int main()
{
	//NeuralNetwork net;
	//net.setOptimizer<SGD>(0.0005f, 0.5f, 0.0f, false); //0.001f, 0.9f, 0.0001f
	//net.setCostFunction<CCE>();
	//net.add(DenseLayer<>(784));
	//net.add(DenseLayer<RELU>(32));
	//net.add(DenseLayer<RELU>(32));
	//net.add(DenseLayer<Softmax>(10));
	//net.build();

	//std::vector<std::vector<float>> inputs = loadImages("data/mnist.input");
	//std::vector<std::vector<float>> labels = loadLabels("data/mnist.label");

	//std::vector<std::vector<float>> input_tests = loadImages("data/mnist-test.input");
	//std::vector<std::vector<float>> label_tests = loadLabels("data/mnist-test.label");

	//DebugTimer t;
	//net.train(inputs, labels, 1, 32);
	//t.stop();

	//TestResult result = net.test(input_tests, label_tests);
	//std::cout << "Loss: " << result.loss << " | Accuracy: " << result.accuracy << std::endl;
	//
	//std::ofstream save("save.net", std::ios::binary);
	//save << net;

	GeneticAlgorithm<Dot> algo;
	algo.setMutationRate(0.15f);

	std::pair<float, float> target = { 5.0f, -6.0f };
	std::vector<Dot> dots(100, Dot(target));

	for(size_t i = 0; i < 40; ++i)
	{
		for(size_t j = 0; j < 20; ++j)
			for(auto& dot : dots)
				dot.update();

		algo.update(dots);
	}

	for(auto& dot : dots)
		std::cout << dot.getValue().first << ", " << dot.getValue().second << std::endl;
	std::cout << "Gen: " << algo.getGeneration() << " | Gen Fitness: " << algo.getFitnessSum() << std::endl;

	return 0;
}