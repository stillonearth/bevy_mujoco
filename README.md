# Bevy MuJoCo

[![Crates.io](https://img.shields.io/crates/v/bevy_mujoco.svg)](https://crates.io/crates/bevy_mujoco)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Crates.io](https://img.shields.io/crates/d/bevy_mujoco.svg)](https://crates.io/crates/bevy_mujoco)
[![Rust](https://github.com/stillonearth/bevy_mujoco/workflows/CI/badge.svg)](https://github.com/stillonearth/bevy_mujoco/actions)

https://user-images.githubusercontent.com/97428129/210613348-82a5e59d-96af-42a9-a94a-c47093eb8297.mp4

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

```rust
// 1. Import bevy_mujoco
use bevy_mujoco::*;
// 2. Setup bevy_mujoco Plugin. MuJoCo Plugin would spawn entities to the world
fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(MuJoCoPluginSettings {
            model_xml_path: "assets/unitree_a1/scene.xml".to_string(),
            pause_simulation: false,
            target_fps: 600.0, // this is not actual fps (bug in bevy_mujoco),
                               // the bigger the value, the slower the simulation
        })        
        .add_plugin(MuJoCoPlugin)
        .add_startup_system(setup)
        .add_system(robot_control_loop.after("mujoco_simulate"))
        .run();
}
// 3. You can control your robots here
fn robot_control_loop(mut mujoco_resources: ResMut<MuJoCoResources>) {
    // prepare simulation data for the NN
    let qpos = mujoco_resources.state.qpos.clone();
    let qvel = mujoco_resources.state.qvel.clone();
    let cfrc_ext = mujoco_resources.state.cfrc_ext.clone();

    // Compute input -> control values here and fill control
    // ...
    let mut control: Vec<f32> = Vec::new(); 

    mujoco_resources.control.data = input_vec;
}
```

**copy build.rs to root of your project to use in with Windows environments. it will copy mujoco.dll to a build dir of your application**

To run tests and example initialize [`mujoco_menagerie`](https://github.com/deepmind/mujoco_menagerie) submobule with

```bash
cd bevy_mujoco
git submodule init
git submodule update
```

See [example](https://github.com/stillonearth/bevy_quadruped_neural_control) for simulating Unitree A1 robot.
