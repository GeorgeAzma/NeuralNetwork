#include "connection.h"

Connection::Connection(const Node& from, const Node& to)
    : from(from), to(to), weight(0.0f), Gene(-1)
{
}

bool Connection::operator==(const Connection & other) const
{
    return from == other.from && to == other.to;
}