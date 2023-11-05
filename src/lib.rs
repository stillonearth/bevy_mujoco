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
    pub root_body: bool,
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
        let mj_plugin_settings = app.world.get_resource::<MuJoCoPluginSettings>().unwrap();

        let model =
            mujoco_rust::Model::from_xml(mj_plugin_settings.model_xml_path.as_str()).unwrap();

        let simulation = MuJoCoSimulation::new(model);

        app.insert_resource(simulation);
        app.add_systems(Update, simulate_physics);
        app.add_systems(Startup, setup_mujoco);
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

pub fn simulate_physics(
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

    for (_, mut transform, body) in bodies_query.iter_mut() {
        let body_id = body.id as usize;
        let mj_body = mujoco_resources.bodies[body_id].clone();
        let parent_body_id = mj_body.parent_id as usize;

        let geom = mj_body.render_geom(&mujoco_resources.geoms);
        if geom.is_none() {
            continue;
        }
        let geom = geom.unwrap();

        let (body_pos, parent_body_pos) = (positions[body_id], positions[parent_body_id]);
        let (body_rot, parent_prot) = (rotations[body_id], rotations[parent_body_id]);

        let (body_translation, parent_body_translation) = (
            vec3_mujoco_2_bevy(body_pos),
            vec3_mujoco_2_bevy(parent_body_pos),
        );

        let (body_rot, parent_body_rot) = (
            quat_mujoco_2_bevy(body_rot),
            quat_mujoco_2_bevy(parent_prot),
        );

        // Converting from MuJoCo to Bevy coordinate system
        let parent_rotation_inverse = parent_body_rot.inverse();
        transform.translation =
            parent_rotation_inverse.mul_vec3(body_translation - parent_body_translation);

        transform.rotation = parent_rotation_inverse * body_rot;

        if body.root_body {
            let correction = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            transform.translation = correction.mul_vec3(transform.translation);
            transform.rotation = correction * transform.rotation;
        }

        if geom.geom_type != GeomType::MESH {
            transform.translation -= geom_correction(&geom);
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

                if depth == 0 {
                    let correction = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
                    body_transform.translation = correction.mul_vec3(body_transform.translation);
                    body_transform.rotation = correction * body_transform.rotation;
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
        // A function that spawn body into the current position in a tree
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
