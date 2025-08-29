
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
