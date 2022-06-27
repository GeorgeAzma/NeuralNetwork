#include "genome.h"
#include "neat.h"

float Genome::distance(const Genome &other) const
{
    const Genome* g1 = this;
    const Genome* g2 = &other;

    uint32_t highest_innovation1 = g1->connections.back().innovation;
    uint32_t highest_innovation2 = g2->connections.back().innovation;

    if(highest_innovation1 < highest_innovation2)
        std::swap(g1, g2);

    uint32_t index1 = 0;
    uint32_t index2 = 0;

    uint32_t disjoint = 0;
    uint32_t excess = 0;
    float weight_diff = 0.0f;
    uint32_t similar = 0;

    while (index1 < g1->connections.size() && index2 < g2->connections.size())
    {
        Connection c1 = g1->connections[index1];
        Connection c2 = g1->connections[index2];

        uint32_t &in1 = c1.innovation;
        uint32_t &in2 = c2.innovation;

        if (in1 == in2) // Similar gene
        {
            ++similar;
            weight_diff += abs(c1.weight - c2.weight);
            ++index1;
            ++index2;
        }
        else if (in1 > in2) // Disjoint gene of g2
        {
            ++disjoint;
            ++index2;
        }
        else // Disjoint gene of g1
        {
            ++disjoint;
            ++index1;
        }
    }

    weight_diff /= similar;
    excess = g1->connections.size() - index1;
    
    uint32_t N = std::max(g1->connections.size(), g2->connections.size());
    if(N < 20)
        N = 1;

    return (neat.c1 * float(excess) / N) + (neat.c2 * float(disjoint) / N) + (neat.c3 * weight_diff);
}

Genome Genome::crossover(const Genome &g1, const Genome &g2)
{
    Neat& neat = g1.neat;
    Genome child = neat.emptyGenome();

    uint32_t index1 = 0;
    uint32_t index2 = 0;

    while (index1 < g1.connections.size() && index2 < g2.connections.size())
    {
        Connection c1 = g1.connections[index1];
        Connection c2 = g2.connections[index2];

        const uint32_t &in1 = c1.innovation;
        const uint32_t &in2 = c2.innovation;

        if (in1 == in2) // Similar gene
        {
            if(RNG::Bool())
                child.connections.add(c1);
            else
                child.connections.add(c2);

            ++index1;
            ++index2;
        }
        else if (in1 > in2) // Disjoint gene of g2
        {
            //TODO: Add c2 connection to child only if g2 is more fit
            ++index2;
        }
        else // Disjoint gene of g1
        {
            //TODO: Add c1 connection to child only if g1 is more fit
            ++index1;
        }
    }

    while(index1 < g1.connections.size())
    {
        Connection c1 = g1.connections[index1];
        child.connections.add(c1); //TODO: Add c1 connection to child only if g1 is more fit
        ++index1;
    }

    for(Connection& c : child.connections)
    {
        child.nodes.add(c.from);
        child.nodes.add(c.to);
    }

    return child;
}

void Genome::mutate()
{
    if(neat.mutate_link_chance >= RNG::Float())
        mutateLink();
    if(neat.mutate_node_chance >= RNG::Float())
        mutateNode();
    if(neat.mutate_weight_shift_chance >= RNG::Float())
        mutateWeightShift();
    if(neat.mutate_weight_random_chance >= RNG::Float())
        mutateWeightRandom();
    if(neat.mutate_link_toggle_chance >= RNG::Float())
        mutateLinkToggle();
}

void Genome::mutateLink()
{
    for(uint32_t i = 0; i < 32; ++i)
    {
        const Node* a = &nodes[RNG::Uint() % nodes.size()];
        const Node* b = &nodes[RNG::Uint() % nodes.size()];

        if(a->x == b->x)
            continue;
        
        if(a->x > b->x)
            std::swap(a, b);

        Connection connection(*a, *b);

        if(connections.contains(connection))
            continue;

        connection = neat.addConnection(connection.from, connection.to);
        connection.weight = neat.weight_random_strength * (RNG::Float() * 2.0f - 1.0f);
        connections.addSorted(connection);
        return;
    }
}

void Genome::mutateNode()
{
    Connection& connection = connections[RNG::Uint() % connections.size()];
    
    Node& from = connection.from;
    Node& to = connection.to;
    Node middle = neat.addNode();
    middle.x = (from.x + to.x) * 0.5f;
    middle.y = (from.y + to.y) * 0.5f;

    Connection connection1 = neat.addConnection(from, middle);
    Connection connection2 = neat.addConnection(middle, to);
    connection1.weight = 1.0f;
    connection2.weight = connection.weight;
    connection2.enabled = connection.enabled;
    connections.remove(connection);
    connections.add(connection1);
    connections.add(connection2);
    nodes.add(middle);
}

void Genome::mutateWeightShift()
{
    Connection& connection = connections[RNG::Uint() % connections.size()];
    connection.weight += neat.weight_shift_strength * (RNG::Float() * 2.0f - 1.0f);
}

void Genome::mutateWeightRandom()
{
    Connection& connection = connections[RNG::Uint() % connections.size()];
    connection.weight = neat.weight_random_strength * (RNG::Float() * 2.0f - 1.0f);
}

void Genome::mutateLinkToggle()
{
    Connection& connection = connections[RNG::Uint() % connections.size()];
    connection.enabled = !connection.enabled;
}