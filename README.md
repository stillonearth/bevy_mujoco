# Bevy MuJoCo

[![Crates.io](https://img.shields.io/crates/v/bevy_mujoco.svg)](https://crates.io/crates/bevy_mujoco)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Crates.io](https://img.shields.io/crates/d/bevy_mujoco.svg)](https://crates.io/crates/bevy_mujoco)
[![Rust](https://github.com/stillonearth/bevy_mujoco/workflows/CI/badge.svg)](https://github.com/stillonearth/bevy_mujoco/actions)

##

![image](https://github.com/stillonearth/bevy_rl/blob/main/img/dog.gif?raw=true)

_This is work in progress_

Import MJCF files into Bevy and run simulations with MuJoCo.

## Getting Started

### Prerequisites

- `bevy` 0.9.0
- `MuJoCo` 2.3.0 installed in `~/.local/mujoco` for Linux or `C:/Program Files/Mujoco` for Windows

### Usage

To run tests and example initialize [`mujoco_menagerie`](https://github.com/deepmind/mujoco_menagerie) submobule with

```bash
cd bevy_mujoco
git submodule init
git submodule update
```

See [example](https://github.com/stillonearth/bevy_mujoco/blob/main/examples/scene.rs) for simulating Unitree A1 robot.
