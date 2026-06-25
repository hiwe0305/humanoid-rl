/**
 * rollout_buffer.cpp
 * Lock-free SPMC (single-producer-multi-consumer style, but here MPSC)
 * ring buffer implementation.
 */

#include "rollout_buffer.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace humanoid_rl {

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::size_t next_pow2(std::size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16; n |= n >> 32;
    return n + 1;
}

// Slot layout (all float32):
//   [obs_dim] [act_dim] [reward=1] [value=1] [log_prob=1] [next_obs=obs_dim]
static int compute_slot_floats(int obs_dim, int act_dim) {
    return obs_dim + act_dim + 1 + 1 + 1 + obs_dim;
}

// ─── Constructor ─────────────────────────────────────────────────────────────

RolloutBuffer::RolloutBuffer(std::size_t capacity, int obs_dim, int act_dim)
    : capacity_(next_pow2(capacity))
    , mask_(capacity_ - 1)
    , obs_dim_(obs_dim)
    , act_dim_(act_dim)
    , slot_floats_(compute_slot_floats(obs_dim, act_dim))
    , data_(std::make_unique<float[]>(capacity_ * slot_floats_))
    , done_(std::make_unique<bool[]>(capacity_))
    , head_(0)
    , tail_(0)
{
    if (obs_dim <= 0 || act_dim <= 0)
        throw std::invalid_argument("obs_dim and act_dim must be positive");
}

// ─── Private slot I/O ────────────────────────────────────────────────────────

void RolloutBuffer::write_slot(std::size_t idx, const Transition& t) {
    float* slot = data_.get() + (idx & mask_) * slot_floats_;
    std::size_t offset = 0;

    std::memcpy(slot + offset, t.obs.data(),     obs_dim_ * sizeof(float));
    offset += obs_dim_;
    std::memcpy(slot + offset, t.action.data(),  act_dim_ * sizeof(float));
    offset += act_dim_;
    slot[offset++] = t.reward;
    slot[offset++] = t.value;
    slot[offset++] = t.log_prob;
    std::memcpy(slot + offset, t.next_obs.data(), obs_dim_ * sizeof(float));

    done_[idx & mask_] = t.done;
}

void RolloutBuffer::read_slot(std::size_t idx,
                              float* obs, float* act, float* rew,
                              bool* done, float* nobs,
                              float* val, float* lp) const {
    const float* slot = data_.get() + (idx & mask_) * slot_floats_;
    std::size_t offset = 0;

    std::memcpy(obs,  slot + offset, obs_dim_ * sizeof(float)); offset += obs_dim_;
    std::memcpy(act,  slot + offset, act_dim_ * sizeof(float)); offset += act_dim_;
    *rew  = slot[offset++];
    *val  = slot[offset++];
    *lp   = slot[offset++];
    std::memcpy(nobs, slot + offset, obs_dim_ * sizeof(float));

    *done = done_[idx & mask_];
}

// ─── Public API ──────────────────────────────────────────────────────────────

bool RolloutBuffer::push(const Transition& t) {
    if (static_cast<int>(t.obs.size()) != obs_dim_ ||
        static_cast<int>(t.action.size()) != act_dim_ ||
        static_cast<int>(t.next_obs.size()) != obs_dim_)
        throw std::invalid_argument("Transition dimension mismatch");

    std::size_t tail = tail_.load(std::memory_order_relaxed);
    std::size_t head = head_.load(std::memory_order_acquire);

    // Buffer full?
    if (tail - head >= capacity_) return false;

    write_slot(tail, t);

    // Publish
    tail_.store(tail + 1, std::memory_order_release);
    return true;
}

std::size_t RolloutBuffer::pop_batch(std::size_t max_count,
                                     float* obs_out, float* act_out,
                                     float* rew_out, bool*  done_out,
                                     float* nobs_out, float* val_out,
                                     float* lp_out) {
    std::size_t head = head_.load(std::memory_order_relaxed);
    std::size_t tail = tail_.load(std::memory_order_acquire);
    std::size_t available = tail - head;
    std::size_t count = std::min(available, max_count);

    for (std::size_t i = 0; i < count; ++i) {
        read_slot(head + i,
                  obs_out  + i * obs_dim_,
                  act_out  + i * act_dim_,
                  rew_out  + i,
                  done_out + i,
                  nobs_out + i * obs_dim_,
                  val_out  + i,
                  lp_out   + i);
    }

    head_.store(head + count, std::memory_order_release);
    return count;
}

std::size_t RolloutBuffer::drain(float* obs_out, float* act_out,
                                  float* rew_out, bool* done_out,
                                  float* nobs_out, float* val_out,
                                  float* lp_out) {
    std::size_t n = size();
    if (n == 0) return 0;
    return pop_batch(n, obs_out, act_out, rew_out, done_out, nobs_out, val_out, lp_out);
}

std::size_t RolloutBuffer::size() const {
    std::size_t tail = tail_.load(std::memory_order_acquire);
    std::size_t head = head_.load(std::memory_order_acquire);
    return tail - head;
}

void RolloutBuffer::clear() {
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
}

} // namespace humanoid_rl
