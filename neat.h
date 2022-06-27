#pragma once

#include "genome.h"
struct Node;
struct Connection;

class Neat
{
public:
    Neat(uint32_t input_size, uint32_t output_size, uint32_t individuals);

    void reset(uint32_t input_size, uint32_t output_size, uint32_t individuals);
    
    void calculate(const Genome& g);

    Genome emptyGenome();
    Connection addConnection(const Node& from, const Node& to);
    Node addNode();
    Node getNode(uint32_t id);

public:
    float c1 = 1.0f;
    float c2 = 1.0f;
    float c3 = 0.4f;

    float weight_shift_strength = 0.2f;
    float weight_random_strength = 1.0f;

    float mutate_link_chance = 0.2f;
    float mutate_node_chance = 0.1f;
    float mutate_weight_shift_chance = 0.6f;
    float mutate_weight_random_chance = 0.3f;
    float mutate_link_toggle_chance = 0.3f;

private: 
    std::vector<Node> input_nodes;
    std::vector<Node> hidden_nodes;
    std::vector<Node> output_nodes;
    uint32_t individuals = 0;
    std::unordered_map<Connection, Connection> all_connections;
    std::vector<Node> all_nodes;
};