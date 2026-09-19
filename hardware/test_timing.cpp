/**
 * hardware/test_timing.cpp
 * ========================
 * High-resolution C++20 timing benchmark and 50µs hardware dwell timing harness.
 * 
 * Compile:
 *    clang++ -O3 -std=c++20 hardware/test_timing.cpp -Ihardware -o hardware/test_timing
 *    ./hardware/test_timing
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include "whittle_engine.hpp"
#include "dwell_timer.hpp"

int main() {
    std::cout << "==============================================================\n";
    std::cout << "  SIH 2026 EW SmartScan: C++20 RMAB Engine & Dwell Timing Loop\n";
    std::cout << "==============================================================\n\n";

    constexpr int TOTAL_CYCLES = 100'000;
    constexpr int BATCH_SIZE = 10;
    constexpr int NUM_BATCHES = TOTAL_CYCLES / BATCH_SIZE;
    bool all_passed = true;

    // ── 1. 100,000-Cycle Decision Latency Benchmark (Single Tuner) ─────────
    {
        rmab::WhittleEngine<35> engine;
        engine.reset(0); // Deterministic mode

        std::cout << "[1/3] Benchmarking Single-Tuner Decision Latency (100,000 cycles)...\n";

        std::vector<double> latencies;
        latencies.reserve(NUM_BATCHES);

        for (int b = 0; b < NUM_BATCHES; ++b) {
            auto t0 = std::chrono::steady_clock::now();
            int band = 0;
            for (int i = 0; i < BATCH_SIZE; ++i) {
                band = engine.select_action();
            }
            auto t1 = std::chrono::steady_clock::now();

            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / BATCH_SIZE;
            latencies.push_back(ns);

            // Periodic feedback update
            if ((b & 7) == 0) {
                engine.update_feedback(band, (b % 5 == 0));
            }
        }

        std::sort(latencies.begin(), latencies.end());
        double median_ns = latencies[NUM_BATCHES / 2];
        double p99_ns = latencies[static_cast<size_t>(NUM_BATCHES * 0.99)];

        std::cout << "      Median Decision Latency: " << std::fixed << std::setprecision(2)
                  << median_ns << " ns\n";
        std::cout << "      99th Percentile Latency: " << p99_ns << " ns\n";
        bool p1 = (median_ns < 100.0);
        std::cout << "      Target (<100ns): " << (p1 ? "PASSED" : "FAILED") << "\n\n";
        all_passed &= p1;
    }

    // ── 2. 100,000-Cycle Decision Latency Benchmark (Multi-Tuner M=4) ──────
    {
        rmab::WhittleEngine<35> engine;
        engine.reset(0);

        std::cout << "[2/3] Benchmarking Multi-Tuner Top-4 Decision Latency (100,000 cycles)...\n";

        std::vector<double> latencies;
        latencies.reserve(NUM_BATCHES);

        for (int b = 0; b < NUM_BATCHES; ++b) {
            auto t0 = std::chrono::steady_clock::now();
            std::array<int, 4> bands{};
            for (int i = 0; i < BATCH_SIZE; ++i) {
                bands = engine.select_actions<4>();
            }
            auto t1 = std::chrono::steady_clock::now();

            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / BATCH_SIZE;
            latencies.push_back(ns);

            if ((b & 7) == 0) {
                engine.update_feedback(bands[0], (b % 3 == 0));
            }
        }

        std::sort(latencies.begin(), latencies.end());
        double median_ns = latencies[NUM_BATCHES / 2];
        double p99_ns = latencies[static_cast<size_t>(NUM_BATCHES * 0.99)];

        std::cout << "      Median Top-4 Decision Latency: " << std::fixed << std::setprecision(2)
                  << median_ns << " ns\n";
        std::cout << "      99th Percentile Latency      : " << p99_ns << " ns\n";
        bool p2 = (median_ns < 100.0);
        std::cout << "      Target (<100ns): " << (p2 ? "PASSED" : "FAILED") << "\n\n";
        all_passed &= p2;
    }

    // ── 3. 50µs Hardware Dwell Timing Loop Simulation (2,000 slots) ───────
    {
        std::cout << "[3/3] Simulating 50us Hardware Dwell Timing Loop (2,000 slots)...\n";
        rmab::DwellTimerSimulation sim(50.0);
        sim.run_dwell_loop(2000, 50.0);
        auto stats = sim.get_statistics();

        std::cout << "      Total Slots Evaluated : " << stats.total_slots << "\n";
        std::cout << "      Deadline Misses (>50us): " << stats.deadline_misses << "\n";
        std::cout << "      Median Decision Time  : " << std::fixed << std::setprecision(2)
                  << stats.median_compute_ns << " ns\n";
        std::cout << "      99th Percentile Compute: " << stats.p99_compute_ns << " ns\n";
        std::cout << "      50us Dwell Budget Used: " << (stats.median_compute_ns / 50000.0) * 100.0 << " %\n";
        std::cout << "      Median Dwell Jitter   : " << std::fixed << std::setprecision(3)
                  << stats.median_jitter_us << " us\n";
        std::cout << "      99th Percentile Jitter : " << stats.p99_jitter_us << " us\n";
        std::cout << "      Max Observed Jitter    : " << stats.max_jitter_us << " us\n";

        bool p3_misses = (stats.deadline_misses == 0);
        bool p3_time = (stats.median_compute_ns < 100.0);
        bool p3_jitter = (stats.p99_jitter_us < 5.0);

        std::cout << "      Jitter Target (<5us)  : " << (p3_jitter ? "PASSED" : "FAILED") << "\n";
        std::cout << "      Zero Deadline Misses  : " << (p3_misses ? "PASSED" : "FAILED") << "\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << "  Overall Result: " << (all_passed && p3_misses && p3_time && p3_jitter ? "ALL CONSTRAINTS PASSED" : "FAILED") << "\n";
        std::cout << "==============================================================\n";

        if (!all_passed || !p3_misses || !p3_time || !p3_jitter) return 1;
    }

    return 0;
}
