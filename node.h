#pragma once

#include "gene.h"

struct Connection;

struct Node : public Gene
{
    Node() = default;
    Node(uint32_t innovation) 
    : Gene(innovation) {}
    
    void calculate();

    bool operator==(const Node& other) const 
    {
        return innovation == other.innovation;
    }

    float x = 0.0f;
    float y = 0.0f;
    float output = 0.0f;
    std::vector<Connection> connections;
};

namespace std
{
    template<>
    struct hash<Node>
    {
        size_t operator()(const Node& node) const
        {
            return node.innovation;
        }
    };
}