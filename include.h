#pragma once 
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <concepts>
#include <thread>
#include <mutex>
#include <future>
#include <queue>

class DebugTimer
{
public:
	DebugTimer(const char *name = "Timer", bool showMilliseconds = true)
		: m_name(name), stopped(false), show_millis(showMilliseconds), start_point(std::chrono::high_resolution_clock::now())
	{
	}

	~DebugTimer()
	{
		if (!stopped)
			stop();
	}

	void stop()
	{
		auto end_point = std::chrono::high_resolution_clock::now();
		long long start = std::chrono::time_point_cast<std::chrono::microseconds>(start_point).time_since_epoch().count();
		long long end = std::chrono::time_point_cast<std::chrono::microseconds>(end_point).time_since_epoch().count();
		long long duration = end - start;
		std::cout << m_name << ": " << duration << "us";
		if (show_millis)
		{
			const float ms = (float)duration * 0.001f;
			std::cout << " (" << ms << "ms)";
		}
		std::cout << '\n';
		stopped = true;
	}

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_point;
	const char *m_name;
	bool stopped;
	bool show_millis;
};

static void makeLittleEndian(uint32_t& num)
{
	num = ((num >> 24) & 0xff)      |
          ((num << 8)  & 0xff0000)  |
          ((num >> 8)  & 0xff00)    |
          ((num << 24) & 0xff000000);
}

static std::vector<float> classToVector(size_t Class, size_t max_size)
{
	//TODO: Assert that Class < max_size;
	std::vector<float> vec(max_size, 0.0f);
	vec[Class] = 1.0f;
	return vec;
}

static size_t vectorToClass(const std::vector<float>& vec)
{
	return size_t(std::max_element(vec.begin(), vec.end()) - vec.begin());
}

template<typename T>
static std::vector<std::vector<T>> splitVector(const std::vector<T>& vec, size_t n)
{
    std::vector<std::vector<T>> out_vec;
    size_t length = vec.size() / n;
    size_t remain = vec.size() % n;
    size_t begin = 0;
    size_t end = 0;

    for (size_t i = 0; i < n; ++i)
    {
        end += length + (i < remain);
        out_vec.emplace_back(vec.begin() + begin, vec.begin() + end);
        begin = end;
    }

    return out_vec;
}

static std::vector<std::vector<float>> loadImages(const char* path)
{
	std::ifstream is(path, std::ios::binary);
	uint32_t magic_number = 0;
	is.read((char*)&magic_number, 4); //TODO: Assert that magic number is correct
	makeLittleEndian(magic_number);
	uint32_t images = 0;
	is.read((char*)&images, 4);
	makeLittleEndian(images);
	uint32_t width = 0;
	is.read((char*)&width, 4);
	makeLittleEndian(width);
	uint32_t height = 0;
	is.read((char*)&height, 4);
	makeLittleEndian(height);
	std::vector<std::vector<uint8_t>> data(images);
	for(auto& image : data)
	{
		image.resize(width * height);
		is.read((char*)image.data(), width * height);
	}
	std::vector<std::vector<float>> normalized_data(data.size());
	for(size_t i = 0; i < normalized_data.size(); ++i)
	{
		normalized_data[i].resize(data[i].size());
		for(size_t j = 0; j < width * height; ++j)
			normalized_data[i][j] = float(data[i][j]) / 255.0f;
	}
	return normalized_data;
}

static std::vector<std::vector<float>> loadLabels(const char* path)
{
	std::ifstream is(path, std::ios::binary);
	uint32_t magic_number = 0;
	is.read((char*)&magic_number, 4); //TODO: Assert that magic number is correct
	makeLittleEndian(magic_number);
	uint32_t labels = 0;
	is.read((char*)&labels, 4);
	makeLittleEndian(labels);
	std::vector<uint8_t> data(labels);
	is.read((char*)data.data(), data.size());
	std::vector<std::vector<float>> normalized_data(data.size());
	for(size_t i = 0; i < normalized_data.size(); ++i)
		normalized_data[i] = classToVector(data[i], 10);
	return normalized_data;
}

class RNG 
{
public:
	using result_type = uint64_t;
	static inline result_type seed = 3773452183ULL;

public:
	static double Float()
	{
		return (double)next() / max();
	}
	static uint64_t Uint()
	{
		return next();
	}
	static int64_t Int()
	{
		return next();
	}
	static bool Bool()
	{
		return next() >> 63ULL;
	}
	static constexpr result_type max()
	{
		return std::numeric_limits<result_type>::max();
	}
	static constexpr result_type min()
	{
		return std::numeric_limits<result_type>::min();
	}

	result_type operator()()
	{
		return next();
	}

private:
	static result_type next()
	{
		result_type z = (seed += result_type(0x9E3779B97F4A7C15));
		z = (z ^ (z >> 30)) *    result_type(0xBF58476D1CE4E5B9);
		z = (z ^ (z >> 27)) *    result_type(0x94D049BB133111EB);
		return z ^ (z >> 31);
	}
};

class ThreadPool
{
public:
    ThreadPool(unsigned int thread_count = std::thread::hardware_concurrency())
    : threads(thread_count)
	{
	    for (auto& t : threads)
	    {
	        t = std::thread(&ThreadPool::work, this);
	    }
	}
    ~ThreadPool()
	{
	    running = false;
	    condition.notify_all();
	    wait();
	    for (auto& t : threads)
	    {
	        if (t.joinable())
	            t.join();
	    }
	}

    template<typename T, typename... Args>
    void submit(T&& function, Args&&... args)
    {
        {
            std::scoped_lock lock(queue_mutex);
            tasks.emplace([function, args...]
                {
                    function(args...);
                });
        }
        ++running_tasks;
        condition.notify_one();
    }

    template<typename Fn, typename... Args, typename R = std::invoke_result_t<std::decay_t<Fn>, std::decay_t<Args>...>, typename = std::enable_if_t<!std::is_void_v<R>>>
    std::future<R> submitFuture(Fn&& function, Args&&... args)
    {
        std::shared_ptr<std::promise<R>> task_promise(new std::promise<R>);
        std::future<R> future = task_promise->get_future();
        {
             std::scoped_lock lock(queue_mutex);
             tasks.emplace([function, args..., task_promise]
                 {
                     task_promise->set_value(function(args...));
                 });
        }
        ++running_tasks;
        condition.notify_one();

        return future;
    }

    template<typename Fn>
    void forEach(size_t count, Fn&& func)
    {
        size_t length = count / threads.size();
        size_t remain = count % threads.size();
        size_t index = 0u;
        for (size_t i = 0u; i < threads.size(); ++i)
        {
            const size_t invocations = length + (i < remain);
            submit([invocations, index, func]{
                for(size_t i = 0u; i < invocations; ++i)
                    func(index + i);
                });
            index += invocations;
        }
        wait();
    }

    void wait()	
	{
	    while (running_tasks)
	        std::this_thread::yield();
	}
    size_t runningTasks() const { return running_tasks; }
    size_t size() const { return threads.size(); }

private:
    void work()
	{
	    while (running)
	    {
	        std::function<void()> task = nullptr;
	        {
	            std::unique_lock lock(queue_mutex);
	            condition.wait(lock, [this]()->bool { return tasks.size() || !running; });
	            if (tasks.empty())
	                break;
	            task = std::move(tasks.front());
	            tasks.pop();
	        }
	        if (task)
	        {
	            task();
	            --running_tasks;
	        }
	    }
	}

private:
    std::mutex queue_mutex;
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> threads;
    std::condition_variable_any condition;
    std::atomic_bool running = true;
    std::atomic<size_t> running_tasks = 0;
};