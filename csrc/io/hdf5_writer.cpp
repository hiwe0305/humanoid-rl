/**
 * hdf5_writer.cpp
 * HDF5 dataset writer implementation using C++ API.
 */

#include "hdf5_writer.h"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace humanoid_rl {
namespace io {

// ─── Helpers ─────────────────────────────────────────────────────────────────

static std::string episode_name(int idx) {
    std::ostringstream oss;
    oss << "ep_" << std::setfill('0') << std::setw(6) << idx;
    return oss.str();
}

static std::string current_timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::gmtime(&t));
    return std::string(buf);
}

template<typename T>
static void write_scalar_attr(H5::H5Object& obj, const std::string& name, T val) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_INT, scalar);
    attr.write(H5::PredType::NATIVE_INT, &val);
}

static void write_string_attr(H5::H5Object& obj, const std::string& name,
                               const std::string& val) {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, str_type, scalar);
    attr.write(str_type, val);
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────

HDF5Writer::HDF5Writer(const std::string& filepath,
                       const std::string& robot,
                       const std::string& task,
                       int compress)
    : file_(filepath, H5F_ACC_TRUNC)
    , robot_(robot)
    , task_(task)
    , compress_(compress)
{
    // Create top-level groups
    file_.createGroup("/metadata");
    episodes_group_ = file_.createGroup("/episodes");

    write_metadata();
}

HDF5Writer::~HDF5Writer() {
    if (!closed_) close();
}

// ─── Metadata ─────────────────────────────────────────────────────────────────

void HDF5Writer::write_metadata() {
    H5::Group meta = file_.openGroup("/metadata");
    write_string_attr(meta, "robot",     robot_);
    write_string_attr(meta, "task",      task_);
    write_string_attr(meta, "created_at", current_timestamp());
    write_scalar_attr(meta, "total_episodes", 0);
    write_scalar_attr(meta, "compress_level", compress_);
}

void HDF5Writer::update_episode_count() {
    H5::Group meta = file_.openGroup("/metadata");
    // Delete old attribute and rewrite
    if (meta.attrExists("total_episodes"))
        meta.removeAttr("total_episodes");
    write_scalar_attr(meta, "total_episodes", episode_count_);
}

// ─── Dataset helpers ──────────────────────────────────────────────────────────

template<typename T>
void HDF5Writer::write_dataset(H5::Group& grp,
                                const std::string& name,
                                const std::vector<std::vector<T>>& data,
                                int rows, int cols) {
    // Flatten to contiguous
    std::vector<float> flat(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = static_cast<float>(data[i][j]);

    hsize_t dims[2] = {(hsize_t)rows, (hsize_t)cols};
    H5::DataSpace dspace(2, dims);

    H5::DSetCreatPropList prop;
    if (compress_ > 0) {
        hsize_t chunk[2] = {std::min((hsize_t)256, (hsize_t)rows), (hsize_t)cols};
        prop.setChunk(2, chunk);
        prop.setDeflate(compress_);
    }

    H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_FLOAT, dspace, prop);
    ds.write(flat.data(), H5::PredType::NATIVE_FLOAT);
}

void HDF5Writer::write_dataset_1d(H5::Group& grp,
                                   const std::string& name,
                                   const std::vector<float>& data) {
    hsize_t dim = data.size();
    H5::DataSpace dspace(1, &dim);

    H5::DSetCreatPropList prop;
    if (compress_ > 0 && dim > 0) {
        hsize_t chunk = std::min(dim, (hsize_t)1024);
        prop.setChunk(1, &chunk);
        prop.setDeflate(compress_);
    }

    H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_FLOAT, dspace, prop);
    ds.write(data.data(), H5::PredType::NATIVE_FLOAT);
}

void HDF5Writer::write_dataset_bool(H5::Group& grp,
                                     const std::string& name,
                                     const std::vector<bool>& data) {
    // Convert vector<bool> (bitfield) to uint8 array
    std::vector<uint8_t> buf(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        buf[i] = data[i] ? 1 : 0;

    hsize_t dim = buf.size();
    H5::DataSpace dspace(1, &dim);

    H5::DSetCreatPropList prop;
    if (compress_ > 0 && dim > 0) {
        hsize_t chunk = std::min(dim, (hsize_t)1024);
        prop.setChunk(1, &chunk);
        prop.setDeflate(compress_);
    }

    H5::DataSet ds = grp.createDataSet(name, H5::PredType::NATIVE_UINT8, dspace, prop);
    ds.write(buf.data(), H5::PredType::NATIVE_UINT8);
}

// ─── Public: write_episode ────────────────────────────────────────────────────

int HDF5Writer::write_episode(const EpisodeData& episode) {
    if (episode.length() == 0)
        throw std::invalid_argument("Cannot write empty episode");

    std::string ep_name = episode_name(episode_count_);
    H5::Group ep_grp = episodes_group_.createGroup(ep_name);

    int T        = episode.length();
    int obs_dim  = episode.obs_dim();
    int act_dim  = episode.act_dim();

    // 2D arrays
    write_dataset(ep_grp, "observations", episode.observations, T, obs_dim);
    write_dataset(ep_grp, "actions",      episode.actions,      T, act_dim);
    write_dataset(ep_grp, "next_obs",     episode.next_obs,     T, obs_dim);

    // 1D arrays
    write_dataset_1d(ep_grp, "rewards",   episode.rewards);
    write_dataset_1d(ep_grp, "values",    episode.values);
    write_dataset_1d(ep_grp, "log_probs", episode.log_probs);

    // Boolean
    write_dataset_bool(ep_grp, "dones", episode.dones);

    // Episode-level attributes
    float total_return = 0.0f;
    for (float r : episode.rewards) total_return += r;
    write_scalar_attr(ep_grp, "length",       T);
    write_scalar_attr(ep_grp, "total_return", (int)(total_return * 1000)); // store as milli

    episode_count_++;
    return episode_count_ - 1;
}

void HDF5Writer::flush() {
    update_episode_count();
    file_.flush(H5F_SCOPE_GLOBAL);
}

void HDF5Writer::close() {
    if (closed_) return;
    flush();
    episodes_group_.close();
    file_.close();
    closed_ = true;
}

} // namespace io
} // namespace humanoid_rl
