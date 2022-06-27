#pragma once

template <typename T>
class RandomUnorderedSet
{
public:
    std::unordered_set<T>::iterator find(const T& element)
    {
        return set.find(element);
    }

    std::unordered_set<T>::const_iterator find(const T& element) const
    {
        return set.find(element);
    }

    bool contains(const T& element)
    {
        return find(element) != set.end();
    }

    void add(const T& element)
    {
        if(set.insert(element).second)
            vec.emplace_back(element);
    }

    void addSorted(const T& element)
    {
        for(size_t i = 0; i < size(); ++i)
        {
            if(element < vec[i])
            {
                vec.emplace(vec.begin() + i, element);
                set.insert(element);
                return;
            }
        }
        vec.emplace_back(element);
        set.insert(element);
    }

    void remove(size_t index)
    {
        set.erase(vec[index]);
        vec.erase(vec.begin() + index);
    }

    void clear()
    {
        set.clear();
        vec.clear();
    }

    size_t size() const { return vec.size(); }

    const T& operator[](size_t index) const { return vec[index]; }
    T& operator[](size_t index) { return vec[index]; }
    const T& back() const { return vec.back(); }
    T& back() { return vec.back(); }
    const T& front() const { return vec.front(); }
    T& front() { return vec.front(); }
    std::vector<T>::const_iterator begin() const { return vec.begin(); }
    std::vector<T>::iterator begin() { return vec.begin(); }
    std::vector<T>::const_iterator end() const { return vec.end(); }
    std::vector<T>::iterator end() { return vec.end(); }

private:
    std::unordered_set<T> set;
    std::vector<T> vec;
};