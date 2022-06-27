#pragma once

template<typename T, typename ScoreType = float>
class RandomSelector
{
public:
    void add(const T& element, ScoreType score)
    {
        objects.emplace_back(element);
        scores.emplace_back(score);
        total_score += score;
    }

    const T& random() const
    {
        ScoreType rand = RNG::Float() * total_score;
        ScoreType running_sum = ScoreType(0);
        for (size_t i = 0; i < scores.size(); ++i)
        {
            running_sum += scores[i];
            if (running_sum >= rand)
                return objects[i];
        }
        return objects.back();
    }

    void clear()
    {
        objects.clear();
        scores.clear();
        total_score = ScoreType(0);
    }

private:
    std::vector<T> objects;
    std::vector<ScoreType> scores;
    ScoreType total_score = ScoreType(0);
};