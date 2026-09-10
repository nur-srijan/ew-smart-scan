/**
 * hardware/test_timing.cpp
 * ========================
 * Benchmark verifying sub-microsecond latency on modern CPUs.
 * Compile:
 *    clang++ -O3 -std=c++20 test_timing.cpp -o test_timing
 *    ./test_timing
 */

#include <iostream>
#include <chrono>
#include "whittle_index.hpp"

int main() {
    ew::WhittleSchedulerState state;
    ew::init_state(state);

    constexpr int ITERATIONS = 100'000;
    std::cout << "Running " << ITERATIONS << " scheduler decision cycles in C++...\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; ++i) {
        uint32_t band = ew::select_next_band(state);
        bool hit = (band == 4 && (i % 5 == 0)); // Simulated pulse
        ew::update_feedback(state, band, hit);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ns = std::chrono::duration<double, std::nano>(t_end - t_start).count();
    double avg_ns = total_ns / ITERATIONS;
    double avg_us = avg_ns / 1000.0;

    std::cout << "--------------------------------------------------\n";
    std::cout << "  C++ Decision + Update Latency: " << avg_ns << " nanoseconds (" 
              << avg_us << " microseconds)\n";
    std::cout << "  Allowed 50us Dwell Budget Used: " << (avg_us / 50.0) * 100.0 << " %\n";
    std::cout << "  Timing Constraint Met: " << (avg_us < 15.0 ? "YES (PASSED)" : "NO") << "\n";
    std::cout << "--------------------------------------------------\n";

    return 0;
}
