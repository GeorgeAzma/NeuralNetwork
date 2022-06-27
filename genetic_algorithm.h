#pragma once

template <typename T>
class Agent
{
  public:
    virtual void calculateFitness() = 0;
    virtual T createBaby(const T &other) const = 0;
    virtual void mutate(float mutation_rate) = 0;
    float getFitness() const
    {
        return fitness;
    }
    bool operator>=(const Agent &other) const
    {
        return fitness >= other.fitness;
    }
    bool operator<=(const Agent &other) const
    {
        return fitness <= other.fitness;
    }
    bool operator>(const Agent &other) const
    {
        return fitness > other.fitness;
    }
    bool operator<(const Agent &other) const
    {
        return fitness < other.fitness;
    }

  protected:
    float fitness = 0.0f;
};

class Dot : public Agent<Dot>
{
  public:
    Dot()
    {
    }
    Dot(const std::string &target)
        : target(target)
    {
        value.resize(target.size());
        for (auto &c : value)
            c = rand();
    }

    char rand()
    {
        return RNG::Int() % 256 - 128;
        char r = RNG::Uint() % 55;
        if (r < 26)
            return 'a' + r;
        if (r < 52)
            return 'A' + (r - 26);
        if (r < 53)
            return ' ';
        if (r < 54)
            return '.';
        if (r < 55)
            return ',';
        return r;
    }

    void update()
    {
    }

    const std::string &getValue() const
    {
        return value;
    }

    void calculateFitness() override
    {
        fitness = 0.0f;
        for (size_t i = 0; i < target.size(); ++i)
            fitness += (target[i] == value[i]);
        fitness /= target.size();
        fitness *= fitness;
    }

    Dot createBaby(const Dot &other) const override
    {
        Dot new_agent = Dot(target);

        for (size_t i = 0; i < value.size(); ++i)
            new_agent.value[i] = RNG::Bool() ? value[i] : other.value[i];
        new_agent.fitness = (getFitness() + other.getFitness()) * 0.5f;
        return new_agent;
    }

    void mutate(float mutation_rate) override
    {
        for (auto &c : value)
            if (RNG::Float() <= mutation_rate)
                c = rand();
    }

  private:
    std::string target;
    std::string value;
};

template <typename T>
class GeneticAlgorithm
{
  public:
    GeneticAlgorithm() = default;

    void setMutationRate(float mutation_rate = 0.05f)
    {
        this->mutation_rate = mutation_rate;
    }

    void update(std::vector<T> &agents)
    {
        calculateFitness(agents);
        calculateFitnessSum(agents);
        doNaturalSelection(agents);
        mutateAgents(agents);
        ++generation;
    }

    uint32_t getGeneration() const
    {
        return generation;
    }
    float getFitnessSum() const
    {
        return fitness_sum;
    }

  private:
    void calculateFitness(std::vector<T> &agents)
    {
        for (auto &agent : agents)
            agent.calculateFitness();
    }

    void calculateFitnessSum(const std::vector<T> &agents)
    {
        fitness_sum = 0.0f;
        for (const auto &agent : agents)
            fitness_sum += agent.getFitness();
    }

    void doNaturalSelection(std::vector<T> &agents)
    {
        std::vector<T> new_agents = agents;

        for (size_t i = 0; i < agents.size(); ++i)
        {
            const T &mother = selectParent(agents);
            const T &father = selectParent(agents, &mother);
            new_agents.emplace_back(mother.createBaby(father));
            new_agents.back().calculateFitness();
            new_agents.emplace_back(father.createBaby(mother));
            new_agents.back().calculateFitness();
        }

        std::sort(new_agents.begin(), new_agents.end(), std::greater<>());
        agents = std::vector<T>(new_agents.begin(), new_agents.begin() + agents.size());
    }

    void mutateAgents(std::vector<T> &agents)
    {
        for (auto &agent : agents)
            agent.mutate(mutation_rate);
    }

    const T &selectParent(const std::vector<T> &agents, const T *exclude = nullptr) const
    {
        if (RNG::Bool()) // Tournament selection
        {
            for (unsigned int i = 0; i < 16; ++i)
            {
                const T &a1 = agents[RNG::Uint() % agents.size()];
                const T &a2 = agents[RNG::Uint() % agents.size()];
                if (a1 > a2)
                {
                    if (&a1 != exclude)
                        return a1;
                }
                else if (&a2 != exclude)
                    return a2;
            }
        }
        else // Fitness biased selection
        {
            float rand = RNG::Float() * fitness_sum;
            float running_sum = 0.0f;
            for (const auto &agent : agents)
            {
                running_sum += agent.getFitness();
                if (running_sum >= rand)
                {
                    if (&agent == exclude)
                        break;
                    return agent;
                }
            }
        }

        return agents[RNG::Uint() % agents.size()];
    }

  private:
    std::vector<T> agents = {};
    float mutation_rate = 0.05f;
    float fitness_sum = 0.0f;
    uint32_t generation = 0;
};