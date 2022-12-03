mod mujoco_shape;
mod wrappers;

use bevy::{ecs::system::EntityCommands, prelude::*, render::mesh::Mesh};
use serde::Serialize;

use std::cell::RefCell;
use std::rc::Rc;

pub use crate::wrappers::*;

#[derive(Component)]
pub struct MuJoCoBody {
    pub id: i32,
}

#[derive(Component)]
pub struct MuJoCoMesh {
    pub id: i32,
}

#[derive(Resource, Default)]
pub struct MuJoCoPluginSettings {
    pub model_xml_path: String,
    pub model_assets_path: String,
    pub pause_simulation: bool,
}

#[derive(Resource, Default)]
pub struct MuJoCoResources {
    pub geoms: Vec<Geom>,
    pub bodies: Vec<Body>,

    pub state: MuJoCoState,
    pub control: MuJoCoControl,
}

#[derive(Default, Debug, Serialize, Clone)]
pub struct MuJoCoState {
    pub sensor_data: Vec<f64>,
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub cfrc_ext: Vec<[f64; 6]>,
}

#[derive(Default, Debug)]
pub struct MuJoCoControl {
    pub data: Vec<f64>,
    pub number_of_controls: usize,
}

pub struct MuJoCoPlugin;

impl Plugin for MuJoCoPlugin {
    fn build(&self, app: &mut App) {
        let map_plugin_settings = app.world.get_resource::<MuJoCoPluginSettings>().unwrap();
        let mujoco = MuJoCo::new_from_xml(map_plugin_settings.model_xml_path.as_str());

        app.insert_resource(mujoco);
        app.add_system(simulate_physics.label("mujoco_simulate"));
        app.add_startup_system(setup_mujoco);
    }
}

fn simulate_physics(
    mujoco: ResMut<MuJoCo>,
    settings: ResMut<MuJoCoPluginSettings>,
    mut bodies_query: Query<(Entity, &mut Transform, &MuJoCoBody)>,
    mut mujoco_resources: ResMut<MuJoCoResources>,
    mut mesh_query: Query<(Entity, &mut Transform, &MuJoCoMesh, Without<MuJoCoBody>)>,
) {
    if settings.pause_simulation {
        return;
    }

    // Set control data
    mujoco.control(&mujoco_resources.control.data);

    // Target 60 fps in simulation
    let sim_start = mujoco.time();
    while mujoco.time() - sim_start < 1.0 / 60000.0 {
        mujoco.step();
    }
    // mujoco.evaluate_sensors();

    // Read Sensor data
    mujoco_resources.state = MuJoCoState {
        sensor_data: mujoco.sensordata(),
        qpos: mujoco.qpos(),
        qvel: mujoco.qvel(),
        cfrc_ext: mujoco.cfrc_ext(),
    };

    let positions = mujoco.xpos();
    let rotations = mujoco.xquat();

    let find_geom = |body_id| -> &Geom {
        let predicate = mujoco_resources.geoms.iter().filter(|geom| {
            geom.body_id == body_id
                && (geom.geom_group == 2 || geom.geom_group == 0 || geom.geom_group == 1)
        });
        assert_eq!(predicate.clone().count(), 1);
        predicate.last().unwrap()
    };

    for (_, mut transform, body) in bodies_query.iter_mut() {
        let body_id = body.id as usize;
        if body_id >= positions.len() {
            continue;
        }

        let body = mujoco_resources.bodies[body_id].clone();
        let body_id = body.id;
        let parent_body_id = body.parent_id;
        let geom = find_geom(body_id);

        let pos = positions[body_id as usize];
        let parent_pos = positions[parent_body_id as usize];
        let rot = rotations[body_id as usize];
        let parent_rot = rotations[parent_body_id as usize];
        let translation = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
        let parent_translation = Vec3::new(
            parent_pos[0] as f32,
            parent_pos[1] as f32,
            parent_pos[2] as f32,
        );

        let rotation = Quat::from_xyzw(rot[1] as f32, rot[0] as f32, rot[2] as f32, -rot[3] as f32);
        let parent_rotation = Quat::from_xyzw(
            parent_rot[1] as f32,
            parent_rot[0] as f32,
            parent_rot[2] as f32,
            -parent_rot[3] as f32,
        );

        // Converting from MuJoCo to Bevy coordinate system
        let parent_rotation_inverse = parent_rotation.inverse();
        transform.translation = parent_rotation_inverse.mul_vec3(translation - parent_translation);
        transform.rotation = parent_rotation_inverse * rotation;
        // this fixes flipped roll
        // TODO: find better way
        let euler = transform.rotation.to_euler(EulerRot::XYZ);
        transform.rotation = Quat::from_euler(EulerRot::XYZ, -euler.0, -euler.1, euler.2);

        // Corrections due to way MuJoCo handles geometry
        match geom.geom_type {
            GeomType::MESH => {
                transform.translation.z *= -1.0;
            }
            _ => {
                transform.translation -= geom.correction();
            }
        }
    }

    // Different coordinate systems with model and data. Rotate model to match data.
    // TODO: this is a hack, find better invariant
    let model_rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    for (_, mut transform, _, _) in mesh_query.iter_mut() {
        transform.rotation = model_rotation;
    }
}

fn setup_mujoco(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<StandardMaterial>>,
    settings: Res<MuJoCoPluginSettings>,
    mujoco: Res<MuJoCo>,
) {
    let bodies = mujoco.bodies();
    let geoms = mujoco.geoms();

    commands.insert_resource(MuJoCoResources {
        geoms: geoms.clone(),
        bodies,
        control: MuJoCoControl {
            number_of_controls: mujoco.nu(),
            ..default()
        },
        ..default()
    });

    // This is a closure that can call itself recursively
    struct SpawnEntities<'s> {
        f: &'s dyn Fn(&SpawnEntities, BodyTree, &mut ChildBuilder, usize),
    }

    impl SpawnEntities<'_> {
        /// Spawn a bevy entity for MuJoCo body
        #[allow(clippy::too_many_arguments)]
        fn spawn_body(
            &self,
            child_builder: &mut ChildBuilder,
            body: &Body,
            geoms: &[Geom],
            settings: &Res<MuJoCoPluginSettings>,
            meshes: &Rc<RefCell<ResMut<Assets<Mesh>>>>,
            materials: &Rc<RefCell<ResMut<Assets<StandardMaterial>>>>,
            add_children: impl FnOnce(&mut ChildBuilder),
            depth: usize,
        ) {
            let geom = body.render_geom(geoms);
            if geom.is_none() {
                return;
            }
            let geom = geom.unwrap();
            let mesh = geom.mesh(settings.model_assets_path.clone());
            let body_transform = body.transform();
            let body_id = body.id;

            let mut binding: EntityCommands;
            {
                let mut materials = materials.borrow_mut();
                let mut meshes = meshes.borrow_mut();

                // Fixing coordinate system of MuJoCo for root body

                let mut transform = Transform {
                    translation: body_transform.translation,
                    ..default()
                };

                if depth == 0 {
                    let t_y = transform.translation.y;
                    let t_z = transform.translation.z;
                    transform.translation.z = t_y;
                    transform.translation.y = t_z;
                    transform.rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
                }

                binding = child_builder.spawn((
                    MuJoCoBody { id: body_id },
                    Name::new(format!("MuJoCo::body_{}", body.name)),
                    SpatialBundle {
                        transform,
                        ..default()
                    },
                ));

                binding.with_children(|children| {
                    let mut cmd = children.spawn(PbrBundle {
                        mesh: meshes.add(mesh),
                        material: materials.add(StandardMaterial {
                            base_color: Color::rgba(
                                geom.color[0],
                                geom.color[1],
                                geom.color[2],
                                geom.color[3],
                            ),
                            ..default()
                        }),
                        ..default()
                    });

                    cmd.insert(Name::new(format!("MuJoCo::mesh_{}", body.name)));
                    if geom.geom_type == GeomType::MESH {
                        cmd.insert(MuJoCoMesh { id: geom.id });
                    }
                });
            }

            binding.with_children(add_children);
        }
    }

    let meshes = Rc::new(RefCell::new(meshes));
    let materials = Rc::new(RefCell::new(materials));
    let commands = Rc::new(RefCell::new(commands));

    // closure implementation
    let spawn_entities = SpawnEntities {
        /// A function that spawn body into the current position in a tree
        f: &|func, body, child_builder, depth| {
            let root_leaf = body.data();

            let add_children = |child_builder: &mut ChildBuilder| {
                let mut body = body.clone();
                loop {
                    let leaf = body.pop_back();
                    if leaf.is_none() {
                        return;
                    }
                    (func.f)(func, BodyTree(leaf.unwrap()), child_builder, depth + 1);
                }
            };

            func.spawn_body(
                child_builder,
                root_leaf,
                &geoms,
                &settings,
                &meshes,
                &materials,
                add_children,
                depth,
            );
        },
    };

    let mut commands = commands.borrow_mut();
    // each mujoco body is defined as a tree
    commands
        .spawn((Name::new("MuJoCo::world"), SpatialBundle::default()))
        .with_children(|child_builder| {
            for body in mujoco.body_tree() {
                (spawn_entities.f)(&spawn_entities, body, child_builder, 0);
            }
        });
}
