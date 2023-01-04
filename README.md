# Bevy MuJoCo

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

See [example](https://github.com/stillonearth/bevy_quadruped_neural_control) for simulating Unitree A1 robot.
