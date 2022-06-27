#pragma once

#include "connection.h"
#include "random_unordered_set.h"


class Genome
{
    friend class Neat;
public:
    // We need neat class because it is keeping track of innovations
    // so if this genome decides to mutate new connection
    // it asks "neat" class if it innovated or not
    // if it did it's innovation number increases
    Genome(Neat& neat)
        : neat(neat) {}

    float distance(const Genome& other) const;

    static Genome crossover(const Genome& g1, const Genome& g2);

    void mutate();
    void mutateLink();
    void mutateNode();
    void mutateWeightShift();
    void mutateWeightRandom();
    void mutateLinkToggle();

    const RandomUnorderedSet<Node>& getNodes() const { return nodes; }

private: 

    RandomUnorderedSet<Connection> connections;
    RandomUnorderedSet<Node> nodes;
    Neat& neat;
};