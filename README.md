# Bevy MuJoCo

![image](https://github.com/stillonearth/bevy_rl/blob/main/img/dog.gif?raw=true)

_This is work in progress_

Import MJCF files into Bevy and run simulations with MuJoCo.

## Implementation Notes

MuJoCo has 2 modes with different coordinate systems for bodies

1. `paused` mode where all translations and rotations are extracted from `mj_Model` in `MuJoCo-Rust` as `body.pos`, `body.quat` in parent's body coordinate system. To make them work nice with bevy the body structure from mujoco has to be transformed to a tree structure with `body_tree()` call. Then `body_tree` is spawned into the bevy world recursively — a nice contraption to do it in `setup_mujoco`. 

2. `simulation` mode where translations are extracted from `sim.xpos()` and `sim.xquat()` — and this time they are in global frame. Since bodies are spawned hierarchically translations and rotations need to be converted to a parent coordinate system — it happens in `simulate_physics`.

## Getting Started

### Prerequisites

- `bevy` 0.9.0
- `MuJoCo` 2.3.0 installed in `~/.local/mujoco` for Linux or `C:/Program Files/Mujoco` for Windows
- *nightly* Rust. Compile with `cargo +nightly build`

### Usage

To run tests and example initialize [`mujoco_menagerie`](https://github.com/deepmind/mujoco_menagerie) submobule with

```bash
cd bevy_mujoco
git submodule init
git submodule update
```

See [example](https://github.com/stillonearth/bevy_quadruped_neural_control) for simulating Unitree A1 robot.
