mod adapters;
mod mujoco_shape;

use bevy::{ecs::system::EntityCommands, prelude::*, render::mesh::Mesh};
use serde::Serialize;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use mujoco_rust::{self, Body, Geom, GeomType};

use crate::adapters::*;

#[derive(Component)]
pub struct MuJoCoBody {
    pub id: i32,
    root_body: bool,
}

#[derive(Component)]
pub struct MuJoCoMesh {
    pub id: i32,
}

#[derive(Resource, Default)]
pub struct MuJoCoPluginSettings {
    pub model_xml_path: String,
    pub pause_simulation: bool,
    pub target_fps: f64,
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
        let model =
            mujoco_rust::Model::from_xml(map_plugin_settings.model_xml_path.as_str()).unwrap();

        let simulation = MuJoCoSimulation::new(model);

        app.insert_resource(simulation);
        app.add_system(simulate_physics.label("mujoco_simulate"));
        app.add_startup_system(setup_mujoco);
    }
}

#[derive(Deref, DerefMut, Resource)]
pub struct MuJoCoSimulation(Arc<Mutex<mujoco_rust::Simulation>>);

impl MuJoCoSimulation {
    pub fn new(model: mujoco_rust::Model) -> Self {
        let simulation = mujoco_rust::Simulation::new(model);
        MuJoCoSimulation(Arc::new(Mutex::new(simulation)))
    }
}

fn simulate_physics(
    mujoco: ResMut<MuJoCoSimulation>,
    settings: ResMut<MuJoCoPluginSettings>,
    mut bodies_query: Query<(Entity, &mut Transform, &MuJoCoBody)>,
    mut mujoco_resources: ResMut<MuJoCoResources>,
) {
    if settings.pause_simulation {
        return;
    }

    let mujoco = mujoco.lock().unwrap();

    // Set control data
    mujoco.control(&mujoco_resources.control.data);

    // Target 60 fps in simulation
    let sim_start = mujoco.state.time();
    while mujoco.state.time() - sim_start < 1.0 / settings.target_fps {
        mujoco.step();
    }

    let cfrc_ext = mujoco.cfrc_ext();
    let cfrc_ext: Vec<[f64; 6]> = cfrc_ext
        .iter()
        .map(|e| [e[0], e[1], e[2], e[3], e[4], e[5]])
        .collect();

    // Read Sensor data
    mujoco_resources.state = MuJoCoState {
        sensor_data: mujoco.sensordata(),
        qpos: mujoco.qpos(),
        qvel: mujoco.qvel(),
        cfrc_ext,
    };

    let positions = mujoco.xpos();
    let rotations = mujoco.xquat();
    let root_body_exists = bodies_query.iter_mut().any(|(_, _, body)| body.id == 0);

    for (_, mut transform, body) in bodies_query.iter_mut() {
        let mut body_id = body.id as usize;
        if !root_body_exists {
            body_id -= 1;
        }

        let mj_body = mujoco_resources.bodies[body_id].clone();
        let body_id = mj_body.id;
        let parent_body_id = mj_body.parent_id;

        let geom = mj_body.render_geom(&mujoco_resources.geoms);
        if geom.is_none() {
            continue;
        }
        let geom = geom.unwrap();

        let bpos = positions[body_id as usize];
        let ppos = positions[parent_body_id as usize];

        let brot = rotations[body_id as usize];
        let prot = rotations[parent_body_id as usize];

        let btran = Vec3::new(bpos[0] as f32, bpos[2] as f32, bpos[1] as f32)
            - geom_correction(&geom) * 3.0 / 4.0 / 2.0;
        let mut ptran = Vec3::new(ppos[0] as f32, ppos[2] as f32, ppos[1] as f32);

        let brot = quat_mujoco_2_bevy(brot);
        let mut prot = quat_mujoco_2_bevy(prot);
        if parent_body_id == 0 {
            prot = Quat::IDENTITY;
            ptran = Vec3::ZERO;
        }

        // Converting from MuJoCo to Bevy coordinate system
        let parent_rotation_inverse = prot.inverse();
        transform.translation = parent_rotation_inverse.mul_vec3(btran - ptran);

        if geom.geom_type == GeomType::MESH {
            transform.translation =
                Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2).mul_vec3(transform.translation);
        }
        transform.rotation = parent_rotation_inverse * brot;

        // Corrections due to way MuJoCo handles geometry
        // TODO: find a nicer way to handle this
        transform.translation.z *= -1.0;
        match geom.geom_type {
            GeomType::MESH => {
                let euler = transform.rotation.to_euler(EulerRot::XYZ);
                // transform.rotation = Quat::from_euler(EulerRot::XZY, -euler.0, -euler.1, -euler.2);
            }
            _ => {
                let correction = (geom_correction(&geom));
                transform.translation += correction;
            }
        }
        if body.root_body && geom.geom_type == GeomType::MESH {
            let correction = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            transform.translation = correction.mul_vec3(transform.translation);
            transform.rotation = correction * transform.rotation;
        }
    }
}

fn setup_mujoco(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<StandardMaterial>>,
    mujoco: ResMut<MuJoCoSimulation>,
) {
    let mujoco = mujoco.lock().unwrap();
    let bodies = mujoco.model.bodies();
    let geoms = mujoco.model.geoms();

    commands.insert_resource(MuJoCoResources {
        geoms: geoms.clone(),
        bodies: bodies.clone(),
        control: MuJoCoControl {
            number_of_controls: mujoco.model.nu(),
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
            meshes: &Rc<RefCell<ResMut<Assets<Mesh>>>>,
            materials: &Rc<RefCell<ResMut<Assets<StandardMaterial>>>>,
            add_children: impl FnOnce(&mut ChildBuilder),
            depth: usize,
        ) {
            let geom = body.render_geom(geoms);
            if geom.is_none() {
                return;
            }
            let geom = &geom.unwrap();
            let mesh = geom_mesh(geom);
            let mut body_transform = body_transform(body);
            let geom_transform = geom_transform(geom);

            let mut binding: EntityCommands;
            {
                let mut materials = materials.borrow_mut();
                let mut meshes = meshes.borrow_mut();

                // Fixing coordinate system of MuJoCo for root body
                if depth == 0 {
                    let t_y = body_transform.translation.y;
                    let t_z = body_transform.translation.z;
                    body_transform.translation.z = t_y;
                    body_transform.translation.y = t_z;

                    if geom.geom_type == GeomType::MESH {
                        body_transform.rotation =
                            Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
                    }
                }

                binding = child_builder.spawn((
                    MuJoCoBody {
                        id: body.id,
                        root_body: depth == 0,
                    },
                    Name::new(format!("MuJoCo::body_{}", body.name)),
                    SpatialBundle {
                        transform: body_transform,
                        ..default()
                    },
                ));

                binding.with_children(|children| {
                    let mut cmd = children.spawn(PbrBundle {
                        mesh: meshes.add(mesh),
                        material: materials.add(geom_material(geom)),
                        transform: geom_transform,
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
                &meshes,
                &materials,
                add_children,
                depth,
            );
        },
    };

    let mut commands = commands.borrow_mut();
    let body_tree = body_tree(&bodies);
    // each mujoco body is defined as a tree
    commands
        .spawn((Name::new("MuJoCo::world"), SpatialBundle::default()))
        .with_children(|child_builder| {
            for body in body_tree {
                (spawn_entities.f)(&spawn_entities, body, child_builder, 0);
            }
        });
}
