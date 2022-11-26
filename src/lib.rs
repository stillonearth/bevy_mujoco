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

#[derive(Resource)]
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
    settings: Res<MuJoCoPluginSettings>,
    mut bodies_query: Query<(Entity, &mut Transform, &MuJoCoBody)>,
    mujoco_resources: Res<MuJoCoResources>,
) {
    if settings.pause_simulation {
        return;
    }

    // Target 60 fps in simulation
    let sim_start = mujoco.time();
    while mujoco.time() - sim_start < 1.0 / 60.0 {
        mujoco.step();
    }

    let positions = mujoco.xpos();
    let rotations = mujoco.xquat();

    for (_, mut transform, body) in bodies_query.iter_mut() {
        let body_id = body.id as usize;
        if body_id >= positions.len() {
            continue;
        }

        let pos = positions[body_id];
        let rot = rotations[body_id];

        let dynamic_translation = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
        let body = mujoco_resources.bodies[body_id].clone();
        let body_id = body.id;
        let predicate = mujoco_resources.geoms.iter().filter(|geom| {
            geom.body_id == body_id && (geom.geom_group == 2 || geom.geom_group == 0)
        });
        assert_eq!(predicate.clone().count(), 1);
        let geom = predicate.last().unwrap();

        // let geom_translation =
        //     Vec3::new(geom.pos[0] as f32, geom.pos[1] as f32, geom.pos[2] as f32);
        // let body_translation =
        //     Vec3::new(body.pos[0] as f32, body.pos[1] as f32, body.pos[2] as f32);

        transform.translation = dynamic_translation - geom.correction();

        let dynamic_rotation =
            Quat::from_xyzw(rot[0] as f32, rot[1] as f32, rot[2] as f32, rot[3] as f32);
        let body_rotation = Quat::from_xyzw(
            body.quat[0] as f32,
            body.quat[1] as f32,
            body.quat[2] as f32,
            body.quat[3] as f32,
        );

        let simulation_rotation = body_rotation * dynamic_rotation;

        transform.rotation = simulation_rotation;
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
    });

    // this is a closure that can call itself recursively
    struct SpawnEntities<'s> {
        f: &'s dyn Fn(&SpawnEntities, BodyTree, &mut ChildBuilder),
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
        ) {
            let body_id = body.id;

            let predicate = geoms.iter().filter(|geom| {
                geom.body_id == body_id && (geom.geom_group == 2 || geom.geom_group == 0)
            });

            if predicate.clone().count() == 0 {
                return;
            }

            assert_eq!(predicate.clone().count(), 1);
            let geom = predicate.last().unwrap();
            // Extracting mesh from mujoco object doesn't work correctly
            // Insted load the obj directly with `bevy_obj`
            let mesh = geom.mesh(settings.model_assets_path.clone());

            let body_transform = body.transform();
            let geom_transform = geom.clone().transform();

            // deref
            // let mut entity_commands = entity_commands.borrow_mut();
            let mut binding: EntityCommands;
            {
                let mut materials = materials.borrow_mut();
                let mut meshes = meshes.borrow_mut();

                // let mut binding = entity_commands.borrow_mut();
                binding = child_builder.spawn(PbrBundle {
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
                        translation: body_transform.translation + geom_transform.translation,
                        rotation: body_transform.rotation * geom_transform.rotation,
                        ..default()
                    },
                    ..default()
                });
            }

            binding
                .insert(MuJoCoBody { id: body_id })
                .insert(Name::new(format!("MuJoCo::body_{}", body.name)));

            binding.with_children(add_children);

            // binding
            // child_entity_commands = binding;
        }
    }

    let meshes = Rc::new(RefCell::new(meshes));
    let materials = Rc::new(RefCell::new(materials));
    let commands = Rc::new(RefCell::new(commands));

    // closure implementation
    let spawn_entities = SpawnEntities {
        /// A function that spawn body into the current position in a tree
        f: &|func, body, child_builder| {
            let root_leaf = body.data();
            // TODO: spawn_body inserts a new entity and return a cursor to a leaf in a tree
            // It does not return entity_commands

            let add_children = |child_builder: &mut ChildBuilder| {
                let mut body = body.clone();
                loop {
                    let leaf = body.pop_back();
                    if leaf.is_none() {
                        return;
                    }
                    (func.f)(func, BodyTree(leaf.unwrap()), child_builder);
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
            );
        },
    };

    let mut commands = commands.borrow_mut();
    // each mujoco body is defined as a tree
    commands
        .spawn((Name::new("MuJoCo::world"), SpatialBundle::default()))
        .with_children(|child_builder| {
            for body in mujoco.body_tree() {
                (spawn_entities.f)(&spawn_entities, body, child_builder);
                // return;
            }
        });
}
