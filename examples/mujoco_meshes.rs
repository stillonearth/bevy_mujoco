use bevy::prelude::*;
use bevy_flycam::{MovementSettings, PlayerPlugin};
use bevy_inspector_egui::WorldInspectorPlugin;
use bevy_mujoco::MuJoCo;
use bevy_obj::*;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
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

    let model = &MuJoCo::new_from_xml("assets/mujoco_menagerie/unitree_a1/scene.xml");
    let mujoco_meshes = model.meshes();
    let mujoco_mesh = mujoco_meshes[0].clone();

    let path = format! {"mujoco_menagerie/unitree_a1/assets/{}.obj", mujoco_mesh.name};
    let cp_handle = asset_server.load(path);

    commands.spawn(PbrBundle {
        mesh: cp_handle,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        material: materials.add(StandardMaterial {
            base_color: Color::GRAY,
            ..Default::default()
        }),
        ..Default::default()
    });

    commands.spawn(PbrBundle {
        mesh: meshes.add(mujoco_mesh.mesh("mujoco_menagerie/unitree_a1/assets".to_string())),
        transform: Transform::from_xyz(0.2, 0.0, 0.0),
        material: materials.add(StandardMaterial {
            base_color: Color::GRAY,
            ..Default::default()
        }),
        ..default()
    });
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::WHITE))
        .add_plugins(DefaultPlugins)
        .add_plugin(ObjPlugin)
        .add_plugin(PlayerPlugin)
        .insert_resource(MovementSettings {
            speed: 3.0,
            ..default()
        })
        .add_plugin(WorldInspectorPlugin::new())
        .add_startup_system(setup)
        .run();
}
