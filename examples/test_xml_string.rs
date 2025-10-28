use bevy::prelude::*;
use bevy_mujoco::*;

const SIMPLE_MODEL_XML: &str = r#"
<mujoco>
  <worldbody>
    <light name="light0" diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
    
    <body name="box" pos="0 0 1">
      <joint name="free_joint" type="free"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
    
    <body name="sphere" pos="0.5 0 1.5">
      <joint name="free_joint2" type="free"/>
      <geom name="sphere_geom" type="sphere" size="0.15" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"#;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(MuJoCoPluginSettings::from_xml_string(SIMPLE_MODEL_XML))
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
