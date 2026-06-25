/**
 * bindings.cpp
 * pybind11 bindings exposing C++ core to Python.
 *
 * Exposed classes:
 *   humanoid_rl_cpp.RolloutBuffer  — lock-free ring buffer
 *   humanoid_rl_cpp.HDF5Writer     — fast episode dataset writer
 *   humanoid_rl_cpp.Transition     — single transition struct
 *   humanoid_rl_cpp.EpisodeData    — full episode struct
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "envpool/rollout_buffer.h"
#include "io/hdf5_writer.h"

namespace py = pybind11;
using namespace humanoid_rl;
using namespace humanoid_rl::io;

// ─── NumPy helpers ────────────────────────────────────────────────────────────

/**
 * Drain buffer contents into numpy arrays.
 * Returns dict: {"obs": np, "action": np, "reward": np, "done": np,
 *                "next_obs": np, "value": np, "log_prob": np}
 */
static py::dict drain_to_numpy(RolloutBuffer& buf) {
    std::size_t n = buf.size();
    if (n == 0) {
        std::vector<py::ssize_t> obs_shape  = {0, buf.obs_dim()};
        std::vector<py::ssize_t> act_shape  = {0, buf.act_dim()};
        return py::dict(
            py::arg("obs")      = py::array_t<float>(obs_shape),
            py::arg("action")   = py::array_t<float>(act_shape),
            py::arg("reward")   = py::array_t<float>(std::vector<py::ssize_t>{0}),
            py::arg("done")     = py::array_t<bool>(std::vector<py::ssize_t>{0}),
            py::arg("next_obs") = py::array_t<float>(obs_shape),
            py::arg("value")    = py::array_t<float>(std::vector<py::ssize_t>{0}),
            py::arg("log_prob") = py::array_t<float>(std::vector<py::ssize_t>{0})
        );
    }

    // Allocate numpy arrays
    std::vector<py::ssize_t> obs_sh  = {(py::ssize_t)n, (py::ssize_t)buf.obs_dim()};
    std::vector<py::ssize_t> act_sh  = {(py::ssize_t)n, (py::ssize_t)buf.act_dim()};
    std::vector<py::ssize_t> flat_sh = {(py::ssize_t)n};
    auto obs_arr  = py::array_t<float>(obs_sh);
    auto act_arr  = py::array_t<float>(act_sh);
    auto rew_arr  = py::array_t<float>(flat_sh);
    auto done_arr = py::array_t<bool>(flat_sh);
    auto nobs_arr = py::array_t<float>(obs_sh);
    auto val_arr  = py::array_t<float>(flat_sh);
    auto lp_arr   = py::array_t<float>(flat_sh);

    buf.drain(
        obs_arr.mutable_data(),
        act_arr.mutable_data(),
        rew_arr.mutable_data(),
        done_arr.mutable_data(),
        nobs_arr.mutable_data(),
        val_arr.mutable_data(),
        lp_arr.mutable_data()
    );

    return py::dict(
        py::arg("obs")      = obs_arr,
        py::arg("action")   = act_arr,
        py::arg("reward")   = rew_arr,
        py::arg("done")     = done_arr,
        py::arg("next_obs") = nobs_arr,
        py::arg("value")    = val_arr,
        py::arg("log_prob") = lp_arr
    );
}

// ─── Module definition ────────────────────────────────────────────────────────

PYBIND11_MODULE(humanoid_rl_cpp, m) {
    m.doc() = "Humanoid-RL C++ extensions: fast rollout buffer + HDF5 writer";

    // ── Transition ──────────────────────────────────────────────────────────
    py::class_<Transition>(m, "Transition")
        .def(py::init<>())
        .def_readwrite("obs",      &Transition::obs)
        .def_readwrite("action",   &Transition::action)
        .def_readwrite("reward",   &Transition::reward)
        .def_readwrite("done",     &Transition::done)
        .def_readwrite("next_obs", &Transition::next_obs)
        .def_readwrite("value",    &Transition::value)
        .def_readwrite("log_prob", &Transition::log_prob)
        .def("__repr__", [](const Transition& t) {
            return "<Transition reward=" + std::to_string(t.reward) +
                   " done=" + (t.done ? "True" : "False") + ">";
        });

    // ── RolloutBuffer ────────────────────────────────────────────────────────
    py::class_<RolloutBuffer>(m, "RolloutBuffer",
        R"doc(
        Lock-free ring buffer for storing RL transitions.

        Multiple producer threads (env workers) push transitions;
        the training thread drains them as numpy arrays.

        Args:
            capacity: max transitions (rounded up to power of 2)
            obs_dim:  observation dimension
            act_dim:  action dimension
        )doc")
        .def(py::init<std::size_t, int, int>(),
             py::arg("capacity"), py::arg("obs_dim"), py::arg("act_dim"))
        .def("push", &RolloutBuffer::push, py::arg("transition"),
             "Push one transition. Returns True on success, False if full.")
        .def("drain", &drain_to_numpy,
             "Drain all transitions into numpy arrays. Returns dict of arrays.")
        .def("clear",    &RolloutBuffer::clear)
        .def("size",     &RolloutBuffer::size)
        .def("empty",    &RolloutBuffer::empty)
        .def("capacity", &RolloutBuffer::capacity)
        .def("obs_dim",  &RolloutBuffer::obs_dim)
        .def("act_dim",  &RolloutBuffer::act_dim)
        .def("__len__",  &RolloutBuffer::size)
        .def("__repr__", [](const RolloutBuffer& b) {
            return "<RolloutBuffer size=" + std::to_string(b.size()) +
                   "/" + std::to_string(b.capacity()) + ">";
        });

    // ── EpisodeData ──────────────────────────────────────────────────────────
    py::class_<EpisodeData>(m, "EpisodeData",
        "Container for a complete episode (observations, actions, rewards, etc.)")
        .def(py::init<>())
        .def_readwrite("observations", &EpisodeData::observations)
        .def_readwrite("actions",      &EpisodeData::actions)
        .def_readwrite("rewards",      &EpisodeData::rewards)
        .def_readwrite("dones",        &EpisodeData::dones)
        .def_readwrite("next_obs",     &EpisodeData::next_obs)
        .def_readwrite("values",       &EpisodeData::values)
        .def_readwrite("log_probs",    &EpisodeData::log_probs)
        .def("length",   &EpisodeData::length)
        .def("obs_dim",  &EpisodeData::obs_dim)
        .def("act_dim",  &EpisodeData::act_dim);

    // ── HDF5Writer ───────────────────────────────────────────────────────────
    py::class_<HDF5Writer>(m, "HDF5Writer",
        R"doc(
        High-performance HDF5 dataset writer.

        Writes episodes with chunked GZIP compression.
        Each episode: observations, actions, rewards, dones,
                       next_obs, values, log_probs.

        Args:
            filepath: output .hdf5 file path
            robot:    robot name (e.g. "unitree_h1")
            task:     task name (e.g. "walk")
            compress: compression level 0-9 (default 4)
        )doc")
        .def(py::init<const std::string&, const std::string&,
                      const std::string&, int>(),
             py::arg("filepath"), py::arg("robot"), py::arg("task"),
             py::arg("compress") = 4)
        .def("write_episode",    &HDF5Writer::write_episode, py::arg("episode"),
             "Write one episode. Returns episode index.")
        .def("flush",            &HDF5Writer::flush,
             "Flush and update metadata.")
        .def("close",            &HDF5Writer::close,
             "Close the HDF5 file.")
        .def("total_episodes",   &HDF5Writer::total_episodes)
        .def("__repr__", [](const HDF5Writer& w) {
            return "<HDF5Writer episodes=" + std::to_string(w.total_episodes()) + ">";
        });
}
