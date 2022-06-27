#include "neat.h"
#include "genome.h"

Neat::Neat(uint32_t input_size, uint32_t output_size, uint32_t individuals)
{
    reset(input_size, output_size, individuals);
}

void Neat::reset(uint32_t input_size, uint32_t output_size, uint32_t individuals)
{
    input_nodes.clear();
    hidden_nodes.clear();
    output_nodes.clear();
    all_connections.clear();
    all_nodes.clear();

    input_nodes.resize(input_size);
    output_nodes.resize(output_size);
    this->individuals = individuals;
    
    for(uint32_t i = 0; i < input_size; ++i)
    {
        Node n = addNode();
        n.x = 0.0f;
        n.y = float(i + 1) / float(input_size + 1);
    }
    for(uint32_t i = 0; i < input_size; ++i)
    {
        Node n = addNode();
        n.x = 1.0f;
        n.y = float(i + 1) / float(input_size + 1);
    }
}

void Neat::calculate(const Genome& g)
{
    hidden_nodes.clear();

    std::map<uint32_t, Node> node_map;

    size_t input_index = 0;
    size_t output_index = 0;
    for(const Node& n : g.nodes)
    {
        node_map.emplace(n.innovation, n);
        if(n.x <= 0.0f)
            input_nodes[input_index++] = n;
        else if(n.x >= 1.0f)
            output_nodes[output_index++] = n;
        else
            hidden_nodes.emplace_back(n);
    }
}

Genome Neat::emptyGenome()
{
    Genome g(*this);
    for(uint32_t i = 0; i < input_nodes.size() + output_nodes.size(); ++i)
        g.nodes.add(getNode(i + 1));
    return g;
}

Connection Neat::addConnection(const Node& from, const Node& to)
{
    Connection connection(from, to);
    auto c = all_connections.find(connection);
    if(c != all_connections.end())
        connection.innovation = c->second.innovation;
    else
    {
        connection.innovation = all_connections.size() + 1;
        all_connections.emplace(connection, connection);
    }

    return connection;
}

Node Neat::addNode()
{
    Node node(all_nodes.size() + 1);
    all_nodes.emplace_back(node);
    return node;
}

Node Neat::getNode(uint32_t id)
{
    if(id < all_nodes.size()) 
        return all_nodes[id];
    return addNode();
}