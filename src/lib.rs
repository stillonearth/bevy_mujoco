mod mujoco_shape;
mod wrappers;

use bevy::{ecs::system::EntityCommands, prelude::*, render::mesh::Mesh};

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

#[derive(Resource)]
pub struct MuJoCoResources {
    pub geoms: Vec<Geom>,
    pub bodies: Vec<Body>,
}

pub struct MuJoCoPlugin;

impl Plugin for MuJoCoPlugin {
    fn build(&self, app: &mut App) {
        let map_plugin_settings = app.world.get_resource::<MuJoCoPluginSettings>().unwrap();
        let mujoco = MuJoCo::new_from_xml(map_plugin_settings.model_xml_path.as_str());

        app.insert_resource(mujoco);
        app.add_system(simulate_physics);
        app.add_startup_system(setup_mujoco);
    }
}

fn simulate_physics(
    mujoco: ResMut<MuJoCo>,
    settings: ResMut<MuJoCoPluginSettings>,
    mut bodies_query: Query<(Entity, &mut Transform, &MuJoCoBody)>,
    mujoco_resources: Res<MuJoCoResources>,
) {
    if settings.pause_simulation {
        return;
    }

    // Target 60 fps in simulation
    let sim_start = mujoco.time();
    while mujoco.time() - sim_start < 1.0 / 120.0 {
        mujoco.step();
    }

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
        transform.translation = parent_rotation
            .inverse()
            .mul_vec3(translation - parent_translation);

        transform.rotation = parent_rotation.inverse() * rotation;

        if geom.geom_type != GeomType::MESH {
            transform.translation -= geom.correction();
        } else {
            transform.translation.z *= -1.0;
        }
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
        bodies: bodies.clone(),
    });

    // this is a closure that can call itself recursively
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
            _bodies: &[Body],
            settings: &Res<MuJoCoPluginSettings>,
            meshes: &Rc<RefCell<ResMut<Assets<Mesh>>>>,
            materials: &Rc<RefCell<ResMut<Assets<StandardMaterial>>>>,
            add_children: impl FnOnce(&mut ChildBuilder),
            _depth: usize,
        ) {
            let geom = body.render_geom(geoms);
            if geom.is_none() {
                return;
            }
            let geom = geom.unwrap();
            let mesh = geom.mesh(settings.model_assets_path.clone());
            let body_transform = body.transform();
            let body_id = body.id;
            // let geom_transform = geom.transform();

            let mut binding: EntityCommands;
            {
                let mut materials = materials.borrow_mut();
                let mut meshes = meshes.borrow_mut();

                binding = child_builder.spawn((
                    MuJoCoBody { id: body_id },
                    Name::new(format!("MuJoCo::body_{}", body.name)),
                    SpatialBundle {
                        transform: body_transform,
                        ..Default::default()
                    },
                ));

                // println!(
                //     "body.name: {}, body.quat:{:?}, geom.quat: {:?}, geom.pos: {:?}, body.pos: {:?}",
                //     body.name, body.quat, geom.quat, geom.pos, body.pos
                // );

                let mut rotation = Quat::IDENTITY;
                if geom.geom_type == GeomType::MESH {
                    rotation = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
                } else {
                    // rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
                }

                binding.with_children(|children| {
                    children
                        .spawn(PbrBundle {
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
                            transform: Transform {
                                rotation,
                                ..default()
                            },
                            ..default()
                        })
                        .insert(MuJoCoMesh { id: geom.id })
                        .insert(Name::new(format!("MuJoCo::mesh_{}", body.name)));
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
                &bodies,
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
