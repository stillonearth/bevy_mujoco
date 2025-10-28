use bevy::prelude::*;
use bevy_mujoco::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(MuJoCoPluginSettings::from_file("assets/mjcf/simple_1.xml"))
        .add_plugins(MuJoCoPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, 3.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
    // Light
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
}
