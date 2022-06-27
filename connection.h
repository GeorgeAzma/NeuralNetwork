#pragma once

#include "node.h"

struct Connection : public Gene
{
    Connection() = default;
    Connection(const Node& from, const Node& to);

    bool operator==(const Connection& other) const;

    Node from;
    Node to;
    float weight;
    bool enabled = true;
};

namespace std
{
    template<>
    struct hash<Connection>
    {
        size_t operator()(const Connection& connection) const
        {
            return connection.from.innovation * 1000000 + connection.to.innovation;
        }
    };
}