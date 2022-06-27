#include "node.h"
#include "connection.h"

void Node::calculate()
{
    float sum = 0.0f;
    for(const Connection& c : connections)
        if(c.enabled)
            sum += c.from.output * c.weight;
    output = std::tanh(sum);
}