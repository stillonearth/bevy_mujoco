// This example runs mujoco_menagerie/unitree_a1/scene.xml simulation with random control

use bevy::prelude::*;
use bevy_flycam::*;
use bevy_mujoco::*;

use rand::Rng;

fn setup(mut commands: Commands) {
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam,
    ));
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
        .insert_resource({
            let mut settings = MuJoCoPluginSettings::from_file("assets/unitree_a1/scene.xml");
            // Alternative paths for testing:
            // settings.model_source = ModelSource::FilePath("assets/mjcf/simple_1.xml".to_string());
            // settings.model_source = ModelSource::FilePath("assets/mjcf/simple_2.xml".to_string());
            // settings.model_source = ModelSource::FilePath("assets/mjcf/simple_3.xml".to_string());
            // settings.model_source = ModelSource::FilePath("assets/mjcf/simple_4.xml".to_string());
            // settings.model_source = ModelSource::FilePath("assets/mujoco_menagerie/agility_cassie/scene.xml".to_string());
            
            settings.pause_simulation = false;
            settings.target_fps = 600.0; // TODO: FPS not correct / no synchronization with physics time
            settings
        })
        .add_plugins(NoCameraPlayerPlugin)
        .insert_resource(MovementSettings {
            speed: 1.0,
            ..default()
        })
        .add_plugins(MuJoCoPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, robot_control_loop.after(simulate_physics))
        .run();
}
