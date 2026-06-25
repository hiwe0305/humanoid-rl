#pragma once
/**
 * rollout_buffer.h
 * Lock-free ring buffer for storing environment transitions.
 * Used by multiple collector threads writing in parallel, one PPO trainer reading.
 *
 * Layout per slot:
 *   [obs (obs_dim)] [action (act_dim)] [reward (1)] [done (1)] [next_obs (obs_dim)]
 *
 * Thread safety:
 *   - Multiple producer threads (env workers) via atomic tail
 *   - Single consumer thread (PPO trainer) via head pointer
 */

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace humanoid_rl {

struct Transition {
    std::vector<float> obs;
    std::vector<float> action;
    float reward = 0.0f;
    bool done = false;
    std::vector<float> next_obs;
    float value = 0.0f;   // V(s) from critic
    float log_prob = 0.0f; // log π(a|s) from actor
};

class RolloutBuffer {
public:
    /**
     * @param capacity  Max number of transitions (power of 2 for fast modulo)
     * @param obs_dim   Observation space dimension
     * @param act_dim   Action space dimension
     */
    RolloutBuffer(std::size_t capacity, int obs_dim, int act_dim);
    ~RolloutBuffer() = default;

    // Non-copyable, movable
    RolloutBuffer(const RolloutBuffer&) = delete;
    RolloutBuffer& operator=(const RolloutBuffer&) = delete;
    RolloutBuffer(RolloutBuffer&&) = default;

    /**
     * Push a transition from any thread (producers).
     * Returns true on success, false if buffer is full.
     */
    bool push(const Transition& t);

    /**
     * Pop a batch of transitions (consumer only — not thread-safe across multiple consumers).
     * Returns number of transitions actually popped.
     */
    std::size_t pop_batch(std::size_t max_count,
                          float* obs_out,   // [max_count, obs_dim]
                          float* act_out,   // [max_count, act_dim]
                          float* rew_out,   // [max_count]
                          bool*  done_out,  // [max_count]
                          float* nobs_out,  // [max_count, obs_dim]
                          float* val_out,   // [max_count]
                          float* lp_out);   // [max_count]

    /** Drain all data into flat arrays. Returns number of transitions. */
    std::size_t drain(float* obs_out, float* act_out, float* rew_out,
                      bool* done_out, float* nobs_out,
                      float* val_out, float* lp_out);

    std::size_t size() const;
    std::size_t capacity() const { return capacity_; }
    bool empty() const { return size() == 0; }
    void clear();

    int obs_dim() const { return obs_dim_; }
    int act_dim() const { return act_dim_; }

private:
    const std::size_t capacity_;
    const std::size_t mask_;   // capacity_ - 1, for fast modulo
    const int obs_dim_;
    const int act_dim_;
    const int slot_floats_;    // floats per slot (excluding bool done)

    // Packed storage: obs | action | reward | value | log_prob | next_obs
    std::unique_ptr<float[]> data_;
    std::unique_ptr<bool[]>  done_;

    alignas(64) std::atomic<std::size_t> head_{0}; // consumer reads here
    alignas(64) std::atomic<std::size_t> tail_{0}; // producers write here

    void write_slot(std::size_t idx, const Transition& t);
    void read_slot(std::size_t idx,
                   float* obs, float* act, float* rew,
                   bool* done, float* nobs, float* val, float* lp) const;
};

} // namespace humanoid_rl
