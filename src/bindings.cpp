/**
 * src/bindings.cpp
 * ================
 * pybind11 Python 3.12 bindings for rmab_cpp C++20 RMAB Whittle Index Engine
 * and 50µs Hardware Dwell Timing Loop Simulation Harness.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "whittle_engine.hpp"
#include "dwell_timer.hpp"

namespace py = pybind11;

using rmab::WhittleEngine;
using rmab::MultiWhittleEngine;
using rmab::TimingStats;
using rmab::DwellTimerSimulation;

PYBIND11_MODULE(rmab_cpp, m) {
    m.doc() = "rmab_cpp: Production Zero-Allocation C++20 RMAB Whittle Index Engine & Dwell Timing Harness";

    // ── 1. Single-Tuner Engine (WhittleEngine<35> / RMABSchedulerCore) ──
    py::class_<WhittleEngine<35>>(m, "WhittleEngine", "Zero-allocation C++20 Whittle Index Scheduler Core")
        .def(py::init<double, double, double, double, double, double, uint64_t>(),
             py::arg("pd") = 0.95,
             py::arg("pfa") = 1e-4,
             py::arg("aoi_weight") = 0.60,
             py::arg("aoi_max") = 50.0,
             py::arg("lr_transition") = 0.05,
             py::arg("camping_penalty_weight") = 0.40,
             py::arg("seed") = 0)
        .def("reset", &WhittleEngine<35>::reset, py::arg("seed") = 0)
        .def("set_deterministic", &WhittleEngine<35>::set_deterministic, py::arg("deterministic"))
        .def("compute_whittle_index", &WhittleEngine<35>::compute_whittle_index, py::arg("k"))
        .def("select_action", &WhittleEngine<35>::select_action,
             py::arg("camping_penalty_weight") = -1.0,
             py::arg("aoi_weight") = -1.0,
             py::arg("aoi_cap") = -1.0)
        .def("select_band", &WhittleEngine<35>::select_band)
        .def("select_actions", [](WhittleEngine<35>& self, int m, double camping_wt, double aoi_wt, double aoi_cap) {
            if (m == 4) {
                auto arr = self.select_actions<4>(camping_wt, aoi_wt, aoi_cap);
                return std::vector<int>(arr.begin(), arr.end());
            } else if (m == 1) {
                int a = self.select_action(camping_wt, aoi_wt, aoi_cap);
                return std::vector<int>{a};
            } else {
                auto arr = self.select_actions<4>(camping_wt, aoi_wt, aoi_cap);
                std::vector<int> res(arr.begin(), arr.begin() + std::min(static_cast<size_t>(m), arr.size()));
                return res;
            }
        }, py::arg("m") = 4,
           py::arg("camping_penalty_weight") = -1.0,
           py::arg("aoi_weight") = -1.0,
           py::arg("aoi_cap") = -1.0)
        .def("update_feedback", &WhittleEngine<35>::update_feedback, py::arg("action"), py::arg("hit"))
        .def("update_feedback_multi", &WhittleEngine<35>::update_feedback_multi, py::arg("actions"), py::arg("hits"))
        .def_property_readonly("beliefs", &WhittleEngine<35>::get_beliefs)
        .def_property_readonly("aoi", &WhittleEngine<35>::get_aoi)
        .def_property_readonly("p01", &WhittleEngine<35>::get_p01)
        .def_property_readonly("p11", &WhittleEngine<35>::get_p11)
        .def_property_readonly("scores", &WhittleEngine<35>::get_scores)
        .def_property_readonly("consecutive_dwells", [](const WhittleEngine<35>& self) {
            return self.consecutive_dwells[static_cast<size_t>(self.last_action)];
        })
        .def_property_readonly("consecutive_dwells_array", &WhittleEngine<35>::get_consecutive_dwells)
        .def_property_readonly("last_action", [](const WhittleEngine<35>& self) { return self.last_action; })
        .def_property_readonly("last_actions", &WhittleEngine<35>::get_last_actions)
        .def_property_readonly("t", [](const WhittleEngine<35>& self) { return self.t; })
        .def_readwrite("Pd", &WhittleEngine<35>::Pd)
        .def_readwrite("Pfa", &WhittleEngine<35>::Pfa)
        .def_readwrite("aoi_weight", &WhittleEngine<35>::aoi_weight)
        .def_readwrite("aoi_max", &WhittleEngine<35>::aoi_max)
        .def_readwrite("lr_transition", &WhittleEngine<35>::lr_transition)
        .def_readwrite("camping_penalty_weight", &WhittleEngine<35>::camping_penalty_weight)
        .def("set_belief", &WhittleEngine<35>::set_belief, py::arg("k"), py::arg("val"))
        .def("set_aoi", &WhittleEngine<35>::set_aoi, py::arg("k"), py::arg("val"))
        .def("set_p01", &WhittleEngine<35>::set_p01, py::arg("k"), py::arg("val"))
        .def("set_p11", &WhittleEngine<35>::set_p11, py::arg("k"), py::arg("val"));

    // Expose alias RMABSchedulerCore
    m.attr("RMABSchedulerCore") = m.attr("WhittleEngine");

    // ── 2. Multi-Tuner Engine (MultiWhittleEngine<35, 4> / MultiRMABSchedulerCore) ──
    py::class_<MultiWhittleEngine<35, 4>>(m, "MultiWhittleEngine", "Zero-allocation C++20 Multi-Tuner RMAB Core")
        .def(py::init<double, double, double, double, double, double, uint64_t>(),
             py::arg("pd") = 0.95,
             py::arg("pfa") = 1e-4,
             py::arg("aoi_weight") = 0.60,
             py::arg("aoi_max") = 50.0,
             py::arg("lr_transition") = 0.05,
             py::arg("camping_penalty_weight") = 0.40,
             py::arg("seed") = 0)
        .def("reset", &MultiWhittleEngine<35, 4>::reset, py::arg("seed") = 0)
        .def("set_deterministic", &MultiWhittleEngine<35, 4>::set_deterministic, py::arg("deterministic"))
        .def("compute_whittle_index", &MultiWhittleEngine<35, 4>::compute_whittle_index, py::arg("k"))
        .def("select_bands", &MultiWhittleEngine<35, 4>::select_bands_vec)
        .def("update_feedback", &MultiWhittleEngine<35, 4>::update_feedback, py::arg("actions"), py::arg("hits"))
        .def_property_readonly("beliefs", &MultiWhittleEngine<35, 4>::get_beliefs)
        .def_property_readonly("aoi", &MultiWhittleEngine<35, 4>::get_aoi)
        .def_property_readonly("p01", &MultiWhittleEngine<35, 4>::get_p01)
        .def_property_readonly("p11", &MultiWhittleEngine<35, 4>::get_p11)
        .def_property_readonly("scores", &MultiWhittleEngine<35, 4>::get_scores)
        .def_property_readonly("consecutive_dwells", &MultiWhittleEngine<35, 4>::get_consecutive_dwells)
        .def_property_readonly("last_actions", &MultiWhittleEngine<35, 4>::get_last_actions)
        .def_property_readonly("t", &MultiWhittleEngine<35, 4>::get_t)
        .def_property("Pd", [](MultiWhittleEngine<35, 4>& self) { return self.core.Pd; },
                             [](MultiWhittleEngine<35, 4>& self, double v) { self.core.Pd = static_cast<float>(v); })
        .def_property("Pfa", [](MultiWhittleEngine<35, 4>& self) { return self.core.Pfa; },
                              [](MultiWhittleEngine<35, 4>& self, double v) { self.core.Pfa = static_cast<float>(v); })
        .def_property("aoi_weight", [](MultiWhittleEngine<35, 4>& self) { return self.core.aoi_weight; },
                                     [](MultiWhittleEngine<35, 4>& self, double v) { self.core.aoi_weight = static_cast<float>(v); })
        .def_property("aoi_max", [](MultiWhittleEngine<35, 4>& self) { return self.core.aoi_max; },
                                  [](MultiWhittleEngine<35, 4>& self, double v) { self.core.aoi_max = static_cast<float>(v); })
        .def_property("lr_transition", [](MultiWhittleEngine<35, 4>& self) { return self.core.lr_transition; },
                                        [](MultiWhittleEngine<35, 4>& self, double v) { self.core.lr_transition = static_cast<float>(v); })
        .def_property("camping_penalty_weight", [](MultiWhittleEngine<35, 4>& self) { return self.core.camping_penalty_weight; },
                                                 [](MultiWhittleEngine<35, 4>& self, double v) { self.core.camping_penalty_weight = static_cast<float>(v); })
        .def("set_belief", &MultiWhittleEngine<35, 4>::set_belief, py::arg("k"), py::arg("val"))
        .def("set_aoi", &MultiWhittleEngine<35, 4>::set_aoi, py::arg("k"), py::arg("val"))
        .def("set_p01", &MultiWhittleEngine<35, 4>::set_p01, py::arg("k"), py::arg("val"))
        .def("set_p11", &MultiWhittleEngine<35, 4>::set_p11, py::arg("k"), py::arg("val"));

    // Expose alias MultiRMABSchedulerCore
    m.attr("MultiRMABSchedulerCore") = m.attr("MultiWhittleEngine");

    // ── 3. Timing Statistics & Hardware Dwell Simulation ──
    py::class_<TimingStats>(m, "TimingStats", "Hardware dwell timing benchmark statistics")
        .def_readonly("total_slots", &TimingStats::total_slots)
        .def_readonly("deadline_misses", &TimingStats::deadline_misses)
        .def_readonly("median_compute_ns", &TimingStats::median_compute_ns)
        .def_readonly("p99_compute_ns", &TimingStats::p99_compute_ns)
        .def_readonly("max_compute_ns", &TimingStats::max_compute_ns)
        .def_readonly("median_jitter_us", &TimingStats::median_jitter_us)
        .def_readonly("p99_jitter_us", &TimingStats::p99_jitter_us)
        .def_readonly("max_jitter_us", &TimingStats::max_jitter_us)
        .def_readonly("total_jitter_us", &TimingStats::total_jitter_us)
        .def("__repr__", [](const TimingStats& s) {
            return "<TimingStats slots=" + std::to_string(s.total_slots) +
                   " misses=" + std::to_string(s.deadline_misses) +
                   " median_compute=" + std::to_string(s.median_compute_ns) + "ns" +
                   " p99_compute=" + std::to_string(s.p99_compute_ns) + "ns" +
                   " median_jitter=" + std::to_string(s.median_jitter_us) + "us" +
                   " p99_jitter=" + std::to_string(s.p99_jitter_us) + "us>";
        });

    py::class_<DwellTimerSimulation>(m, "DwellTimerSimulation", "50µs Hardware Dwell Timing Loop Simulation Harness")
        .def(py::init<double>(), py::arg("budget_us") = 50.0)
        .def("run_dwell_loop", &DwellTimerSimulation::run_dwell_loop,
             py::arg("num_slots") = 2000,
             py::arg("budget_us") = 50.0,
             py::arg("multi_tuner") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("get_statistics", &DwellTimerSimulation::get_statistics)
        .def_readonly("stats", &DwellTimerSimulation::stats);

    m.def("run_dwell_simulation", &rmab::run_dwell_simulation,
          py::arg("num_slots") = 2000,
          py::arg("budget_us") = 50.0,
          py::arg("multi_tuner") = false,
          py::call_guard<py::gil_scoped_release>(),
          "Run 50µs hardware dwell timing loop simulation and return statistics");

    // Microsecond / nanosecond latency benchmark helper
    m.def("benchmark_decision_latency", [](size_t iterations, bool multi_tuner) {
        WhittleEngine<35> engine;
        engine.reset(0); // Deterministic mode

        constexpr size_t BATCH_SIZE = 10;
        const size_t num_batches = (iterations < BATCH_SIZE) ? 1 : (iterations / BATCH_SIZE);
        std::vector<double> latencies_ns;
        latencies_ns.reserve(num_batches);

        for (size_t b = 0; b < num_batches; ++b) {
            auto t0 = std::chrono::steady_clock::now();
            int b0 = 0;
            if (multi_tuner) {
                for (size_t i = 0; i < BATCH_SIZE; ++i) {
                    auto actions = engine.select_actions<4>();
                    b0 = actions[0];
                }
            } else {
                for (size_t i = 0; i < BATCH_SIZE; ++i) {
                    b0 = engine.select_action();
                }
            }
            auto t1 = std::chrono::steady_clock::now();

            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / BATCH_SIZE;
            latencies_ns.push_back(ns);

            if ((b & 7) == 0) {
                engine.update_feedback(b0, (b % 5 == 0));
            }
        }

        std::sort(latencies_ns.begin(), latencies_ns.end());
        double sum = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0);

        py::gil_scoped_acquire acquire;
        py::dict res;
        res["iterations"] = iterations;
        res["median_ns"] = latencies_ns[num_batches / 2];
        res["mean_ns"] = sum / num_batches;
        res["min_ns"] = latencies_ns.front();
        res["max_ns"] = latencies_ns.back();
        res["p90_ns"] = latencies_ns[static_cast<size_t>(num_batches * 0.90)];
        res["p99_ns"] = latencies_ns[static_cast<size_t>(num_batches * 0.99)];
        return res;
    }, py::arg("iterations") = 100000, py::arg("multi_tuner") = true,
       py::call_guard<py::gil_scoped_release>(),
       "Benchmark C++ decision latency over N iterations on CPU");
}
