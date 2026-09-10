#pragma once
/**
 * hardware/whittle_index.hpp
 * ==========================
 * Sub-Microsecond Closed-Form Whittle Index Scheduler for Electronic Warfare.
 * Designed for zero dynamic memory allocation on ARM Cortex / x86-64.
 *
 * Formula:
 *   delta = P11 - P01
 *   W(p)  = [p * delta + P01] / [1 - delta + p * delta]
 *
 * Author: Mukesh / SIH 2026 EW Team
 */

#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace ew {

constexpr size_t NUM_BANDS = 35;

struct WhittleSchedulerState {
    std::array<float, NUM_BANDS> belief;     // b[k] in [0.001, 0.999]
    std::array<float, NUM_BANDS> aoi;        // Age-of-Information (slots)
    std::array<float, NUM_BANDS> P01;        // Transition P(0 -> 1)
    std::array<float, NUM_BANDS> P11;        // Transition P(1 -> 1)
    uint32_t last_action;
    uint32_t consecutive_dwells;
};

inline void init_state(WhittleSchedulerState& state) noexcept {
    state.belief.fill(0.15f);
    state.aoi.fill(0.0f);
    state.P01.fill(0.03f);
    state.P11.fill(0.92f);
    state.last_action = 0;
    state.consecutive_dwells = 0;
}

// Compute closed-form Whittle index for channel k
inline float compute_whittle_index(float p, float p01, float p11) noexcept {
    const float delta = p11 - p01;
    const float num   = p * delta + p01;
    const float denom = (1.0f - delta) + p * delta;
    if (denom <= 1e-7f) return p;
    return num / denom;
}

// Select best sub-band in < 100 nanoseconds
inline uint32_t select_next_band(WhittleSchedulerState& state) noexcept {
    float best_score = -1e9f;
    uint32_t best_band = 0;

    for (size_t k = 0; k < NUM_BANDS; ++k) {
        float w_idx = compute_whittle_index(state.belief[k], state.P01[k], state.P11[k]);
        
        // Exploration subsidy proportional to Age-of-Information (AoI)
        float aoi_bonus = 0.60f * std::min(1.0f, state.aoi[k] / 50.0f);

        // Anti-camping penalty
        float camping_penalty = (k == state.last_action) ? (0.40f * static_cast<float>(state.consecutive_dwells)) : 0.0f;

        float score = w_idx + aoi_bonus - camping_penalty;

        if (score > best_score) {
            best_score = score;
            best_band = static_cast<uint32_t>(k);
        }
    }

    if (best_band == state.last_action) {
        state.consecutive_dwells++;
    } else {
        state.consecutive_dwells = 0;
    }
    state.last_action = best_band;

    return best_band;
}

// Fast Bayesian belief update upon observing Hit/Miss byte from FPGA
inline void update_feedback(WhittleSchedulerState& state, uint32_t action, bool hit) noexcept {
    const float duty = 0.20f;
    const float Pd   = 0.95f;
    const float Pfa  = 1e-4f;

    float p = state.belief[action];

    if (hit) {
        const float p_hit_present = duty * Pd + (1.0f - duty) * Pfa;
        const float p_hit_absent  = Pfa;
        p = (p * p_hit_present) / std::max(1e-9f, p * p_hit_present + (1.0f - p) * p_hit_absent);
    } else {
        const float p_miss_present = duty * (1.0f - Pd) + (1.0f - duty) * (1.0f - Pfa);
        const float p_miss_absent  = 1.0f - Pfa;
        p = (p * p_miss_present) / std::max(1e-9f, p * p_miss_present + (1.0f - p) * p_miss_absent);
    }

    state.belief[action] = std::clamp(p, 0.001f, 0.999f);

    // Diffuse unsensed bands toward prior & update AoI
    constexpr float alpha = 0.02f;
    constexpr float prior = 0.15f;
    for (size_t k = 0; k < NUM_BANDS; ++k) {
        if (k != action) {
            state.belief[k] = (1.0f - alpha) * state.belief[k] + alpha * prior;
            state.aoi[k] += 1.0f;
        } else {
            state.aoi[k] = 0.0f;
        }
    }
}

} // namespace ew
