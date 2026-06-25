#pragma once
/**
 * hdf5_writer.h
 * High-performance HDF5 dataset writer for RL trajectories.
 *
 * Supports:
 *  - Chunked storage with GZIP compression
 *  - Episode-based hierarchical layout
 *  - Metadata attributes (robot, task, timestamp)
 *  - Thread-safe episode writing (one writer thread)
 *
 * File layout:
 *   /metadata/robot, task, total_episodes, obs_dim, act_dim
 *   /episodes/ep_XXXXXX/observations  [T, obs_dim]
 *   /episodes/ep_XXXXXX/actions       [T, act_dim]
 *   /episodes/ep_XXXXXX/rewards       [T]
 *   /episodes/ep_XXXXXX/dones         [T]
 *   /episodes/ep_XXXXXX/next_obs      [T, obs_dim]
 *   /episodes/ep_XXXXXX/values        [T]  (critic values, for offline RL)
 *   /episodes/ep_XXXXXX/log_probs     [T]
 */

#include <H5Cpp.h>
#include <memory>
#include <string>
#include <vector>

namespace humanoid_rl {
namespace io {

struct EpisodeData {
    std::vector<std::vector<float>> observations;  // [T][obs_dim]
    std::vector<std::vector<float>> actions;        // [T][act_dim]
    std::vector<float>              rewards;         // [T]
    std::vector<bool>               dones;           // [T]
    std::vector<std::vector<float>> next_obs;        // [T][obs_dim]
    std::vector<float>              values;          // [T]
    std::vector<float>              log_probs;       // [T]

    int obs_dim() const { return observations.empty() ? 0 : (int)observations[0].size(); }
    int act_dim() const { return actions.empty()      ? 0 : (int)actions[0].size(); }
    int length()  const { return (int)rewards.size(); }
};

class HDF5Writer {
public:
    /**
     * @param filepath  Output .hdf5 file path
     * @param robot     Robot name (e.g. "unitree_h1")
     * @param task      Task name (e.g. "walk")
     * @param compress  Compression level 0-9 (0 = off, 4 = good balance)
     */
    HDF5Writer(const std::string& filepath,
               const std::string& robot,
               const std::string& task,
               int compress = 4);

    ~HDF5Writer();

    // Non-copyable
    HDF5Writer(const HDF5Writer&) = delete;
    HDF5Writer& operator=(const HDF5Writer&) = delete;

    /**
     * Write one complete episode.
     * Returns the episode index assigned.
     */
    int write_episode(const EpisodeData& episode);

    /**
     * Flush and update metadata (total_episodes, etc.)
     */
    void flush();

    /**
     * Close the file explicitly (also called in destructor).
     */
    void close();

    int total_episodes() const { return episode_count_; }

private:
    H5::H5File  file_;
    H5::Group   episodes_group_;
    std::string robot_;
    std::string task_;
    int         compress_;
    int         episode_count_ = 0;
    bool        closed_ = false;

    void write_metadata();
    void update_episode_count();

    template<typename T>
    void write_dataset(H5::Group& grp,
                       const std::string& name,
                       const std::vector<std::vector<T>>& data,
                       int rows, int cols);

    void write_dataset_1d(H5::Group& grp,
                          const std::string& name,
                          const std::vector<float>& data);

    void write_dataset_bool(H5::Group& grp,
                            const std::string& name,
                            const std::vector<bool>& data);
};

} // namespace io
} // namespace humanoid_rl
