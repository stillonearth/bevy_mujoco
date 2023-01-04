use bevy::prelude::*;
use bevy_flycam::*;
use bevy_inspector_egui::*;
use bevy_mujoco::*;

use rand::Rng;

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

    commands
        .spawn(Camera3dBundle {
            transform: Transform::from_xyz(0.0, 2.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        })
        .insert(FlyCam);
}

#[allow(unused_mut)]
fn robot_control_loop(mut mujoco_resources: ResMut<MuJoCoResources>) {
    let mut rng = rand::thread_rng();
    let mut control: Vec<f64> = vec![0.0; mujoco_resources.control.number_of_controls];
    for i in 0..mujoco_resources.control.number_of_controls {
        control[i] = rng.gen::<f64>();
    }
    mujoco_resources.control.data = control;
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(WorldInspectorPlugin::new())
        .insert_resource(MuJoCoPluginSettings {
            // model_xml_path: "assets/mjcf/simple_1.xml".to_string(),
            // * TODO: this example will crash
            // model_xml_path: "assets/mjcf/simple_2.xml".to_string(),
            // model_xml_path: "assets/mjcf/simple_3.xml".to_string(),
            // model_xml_path: "assets/mjcf/simple_4.xml".to_string(),
            model_xml_path: "assets/mujoco_menagerie/unitree_a1/scene.xml".to_string(),
            // model_xml_path: "assets/mujoco_menagerie/agility_cassie/scene.xml".to_string(),
            // pause_simulation: false,
            pause_simulation: false,
            // * TODO: FPS not correct / no synchronization with physics time
            target_fps: 600.0,
        })
        .add_plugin(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            speed: 1.0,
            ..default()
        })
        .add_plugin(MuJoCoPlugin)
        .add_startup_system(setup)
        .add_system(robot_control_loop)
        .run();
}
