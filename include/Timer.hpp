
#pragma once
#include <chrono>
#include <iostream>
#include <string>

class Timer {
public:
    Timer(const std::string& name)
        : name(name), start(std::chrono::high_resolution_clock::now()), stopped(false) {}

    ~Timer() {
        if (!stopped) s();  // auto-stop if not done manually
    }

    void s() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << name << " took " << ms << " ms\n";
        stopped = true;
    }

private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    bool stopped;
};



#include <deque>

class FrameTimeAverager {
public:
    FrameTimeAverager(size_t maxSize = 30) 
        : maxSize(maxSize), sum(0.0) {}

    void addFrameTime(double frametime) {
        if (times.size() == maxSize) {
            // remove oldest
            sum -= times.front();
            times.pop_front();
        }
        // add newest
        times.push_back(frametime);
        sum += frametime;
    }

    double getAverage() const {
        if (times.empty()) return 0.0;
        return sum / times.size();
    }

private:
    size_t maxSize;
    std::deque<double> times;
    double sum;
};

