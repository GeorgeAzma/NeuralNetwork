#pragma once

struct Gene
{
    Gene() = default;
    Gene(uint32_t innovation) 
    : innovation(innovation) {}

    bool operator==(const Gene& other) const { return innovation == other.innovation; }
    bool operator!=(const Gene& other) const { return innovation != other.innovation; }
    bool operator<(const Gene& other) const { return innovation < other.innovation; }
    bool operator>(const Gene& other) const { return innovation > other.innovation; }
    bool operator<=(const Gene& other) const { return innovation <= other.innovation; }
    bool operator>=(const Gene& other) const { return innovation >= other.innovation; }
    operator uint32_t() const { return innovation; }

    uint32_t innovation;
};