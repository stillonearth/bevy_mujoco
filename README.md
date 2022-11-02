# Bevy MuJoCo

_This is work in progress_

Import MCJ files into Bevy and run simulations with MuJoCo.

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

**Example**

```rust
use bevy_mujoco::{MuJoCoPlugin, MuJoCoPluginSettings};

fn main (){
    App::new()
        .insert_resource(MuJoCoPluginSettings {
            model_xml_path: "assets/mujoco_menagerie/unitree_a1/scene.xml".to_string(),
            model_assets_path: "assets/mujoco_menagerie/unitree_a1/assets/".to_string(),
        })
        .add_plugin(MuJoCoPlugin)
        .run();
}
```
