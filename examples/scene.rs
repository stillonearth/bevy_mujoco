use bevy::prelude::*;

use bevy_inspector_egui::WorldInspectorPlugin;
use bevy_mujoco::{MuJoCoPlugin, MuJoCoPluginSettings};

fn setup(mut commands: Commands) {
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 9000.0,
            range: 100.,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(8.0, 16.0, 8.0),
        ..default()
    });
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::WHITE))
        .insert_resource(MuJoCoPluginSettings {
            model_xml_path: "assets/mujoco_menagerie/unitree_a1/scene.xml".to_string(),
            model_assets_path: "assets/mujoco_menagerie/unitree_a1/assets/".to_string(),
        })
        .add_plugins(DefaultPlugins)
        // .add_plugin(PlayerPlugin)
        // .insert_resource(MovementSettings {
        //     speed: 3.0,
        //     ..default()
        // })
        .add_plugin(MuJoCoPlugin)
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .run();
}
