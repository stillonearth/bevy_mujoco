// Updated bevy_mujoco with mujoco-rs 2.0 + Bevy 0.17.2
//
// This is a working implementation that provides:
// - MuJoCo simulation integration with Bevy
// - State tracking (qpos, qvel, cfrc_ext, sensor data)
// - Control interface for actuators
// - Body transform synchronization
// - Geom-based visual representation with proper hierarchy
// - ECS-first physics API
//
mod adapters;
mod body;
pub mod builder;
mod geom;
pub mod physics;
pub mod query_extensions;
pub mod systems;

use bevy::prelude::*;
pub use body::{Body, BodyNode};
pub use geom::{Geom, GeomType};
use mujoco_rs::mujoco_c::{mjData, mjModel};
use mujoco_rs::prelude::*;
use serde::Serialize;
use std::sync::{Arc, Mutex};

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

/// Source for loading a MuJoCo model
#[derive(Clone)]
pub enum ModelSource {
    /// Load from a file path
    FilePath(String),
    /// Load from an XML string (uses VFS internally)
    XmlString(String),
}

impl Default for ModelSource {
    fn default() -> Self {
        ModelSource::FilePath(String::new())
    }
}

#[derive(Resource, Default)]
pub struct MuJoCoPluginSettings {
    pub model_source: ModelSource,
    pub pause_simulation: bool,
    pub target_fps: f64,
}

impl MuJoCoPluginSettings {
    /// Create settings for loading from a file path
    pub fn from_file(path: impl Into<String>) -> Self {
        Self {
            model_source: ModelSource::FilePath(path.into()),
            pause_simulation: false,
            target_fps: 60.0,
        }
    }

    /// Create settings for loading from an XML string
    pub fn from_xml_string(xml: impl Into<String>) -> Self {
        Self {
            model_source: ModelSource::XmlString(xml.into()),
            pause_simulation: false,
            target_fps: 60.0,
        }
    }

    /// Backwards compatibility: load from model_xml_path
    pub fn with_path(path: impl Into<String>) -> Self {
        Self::from_file(path)
    }
}

#[derive(Resource, Default)]
pub struct MuJoCoResources {
    pub geoms: Vec<Geom>,
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

#[derive(Resource)]
pub struct MuJoCoSimulation {
    pub model: Arc<MjModel>,
    pub data: Arc<Mutex<MjData<Arc<MjModel>>>>,
}

impl MuJoCoSimulation {
    /// Create a new simulation from a model
    pub fn new(model: MjModel) -> Self {
        let model = Arc::new(model);
        let data = MjData::new(model.clone());
        let data = Arc::new(Mutex::new(data));

        MuJoCoSimulation { model, data }
    }

    /// Create a simulation by loading a model from a file path
    pub fn from_file(path: impl AsRef<str>) -> Result<Self, String> {
        let model = MjModel::from_xml(path.as_ref())
            .map_err(|e| format!("Failed to load model from file: {}", e))?;
        Ok(Self::new(model))
    }

    /// Create a simulation by loading a model from an XML string
    pub fn from_xml_string(xml: impl AsRef<str>) -> Result<Self, String> {
        let model = MjModel::from_xml_string(xml.as_ref())
            .map_err(|e| format!("Failed to load model from XML string: {}", e))?;
        Ok(Self::new(model))
    }

    /// Set control values (actuator inputs)
    pub fn set_control(&self, control_data: &[f64]) {
        let mut data = self.data.lock().unwrap();
        let ctrl_slice =
            unsafe { std::slice::from_raw_parts_mut(data.ffi_mut().ctrl, control_data.len()) };
        ctrl_slice.copy_from_slice(control_data);
    }

    /// Step the simulation forward by one timestep
    pub fn step(&self) {
        let mut data = self.data.lock().unwrap();
        data.step();
    }

    /// Get current simulation time
    pub fn time(&self) -> f64 {
        let data = self.data.lock().unwrap();
        data.ffi().time
    }

    /// Get body positions (xpos)
    pub fn xpos(&self) -> Vec<[f64; 3]> {
        let data = self.data.lock().unwrap();
        let ffi = data.ffi();
        unsafe {
            std::slice::from_raw_parts(ffi.xpos as *const [f64; 3], self.model.ffi().nbody as usize)
        }
        .to_vec()
    }

    /// Get body quaternions (xquat)
    pub fn xquat(&self) -> Vec<[f64; 4]> {
        let data = self.data.lock().unwrap();
        let ffi = data.ffi();
        unsafe {
            std::slice::from_raw_parts(
                ffi.xquat as *const [f64; 4],
                self.model.ffi().nbody as usize,
            )
        }
        .to_vec()
    }

    /// Get sensor data
    pub fn sensordata(&self) -> Vec<f64> {
        let data = self.data.lock().unwrap();
        let ffi = data.ffi();
        unsafe { std::slice::from_raw_parts(ffi.sensordata, self.model.ffi().nsensordata as usize) }
            .to_vec()
    }

    /// Get generalized positions (qpos)
    pub fn qpos(&self) -> Vec<f64> {
        let data = self.data.lock().unwrap();
        let ffi = data.ffi();
        unsafe { std::slice::from_raw_parts(ffi.qpos, self.model.ffi().nq as usize) }.to_vec()
    }

    /// Get generalized velocities (qvel)
    pub fn qvel(&self) -> Vec<f64> {
        let data = self.data.lock().unwrap();
        let ffi = data.ffi();
        unsafe { std::slice::from_raw_parts(ffi.qvel, self.model.ffi().nv as usize) }.to_vec()
    }

    /// Get contact forces (cfrc_ext)
    pub fn cfrc_ext(&self) -> Vec<[f64; 6]> {
        let data = self.data.lock().unwrap();
        let ffi = data.ffi();
        unsafe {
            std::slice::from_raw_parts(
                ffi.cfrc_ext as *const [f64; 6],
                self.model.ffi().nbody as usize,
            )
        }
        .to_vec()
    }

    /// Direct access to the model FFI (for advanced use)
    pub fn model_ffi(&self) -> &mjModel {
        self.model.ffi()
    }

    /// Direct access to the data FFI (for advanced use, requires locking)
    pub fn with_data<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mjData) -> R,
    {
        let data = self.data.lock().unwrap();
        f(data.ffi())
    }

    /// Direct mutable access to the data FFI (for advanced use, requires locking)
    pub fn with_data_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut mjData) -> R,
    {
        let mut data = self.data.lock().unwrap();
        unsafe { f(data.ffi_mut()) }
    }
}

pub struct MuJoCoPlugin;

impl Plugin for MuJoCoPlugin {
    fn build(&self, app: &mut App) {
        let mj_plugin_settings = app
            .world()
            .get_resource::<MuJoCoPluginSettings>()
            .expect("MuJoCoPluginSettings resource must be added before MuJoCoPlugin");

        // Load the model based on the source type
        let model = match &mj_plugin_settings.model_source {
            ModelSource::FilePath(path) => {
                MjModel::from_xml(path.as_str()).expect("Failed to load MuJoCo model from file")
            }
            ModelSource::XmlString(xml) => MjModel::from_xml_string(xml.as_str())
                .expect("Failed to load MuJoCo model from XML string"),
        };

        // Debug: Check immediately after loading
        println!("🔍 Immediately after loading:");
        println!("   model.ffi().ngeom = {}", model.ffi().ngeom);
        println!("   model.ffi().nbody = {}", model.ffi().nbody);
        println!("   model.ffi().nmesh = {}", model.ffi().nmesh);
        println!("   model.ffi().njnt = {}", model.ffi().njnt);

        // Check if geom_type pointer is valid
        unsafe {
            if !model.ffi().geom_type.is_null() {
                println!("   geom_type pointer: valid");
            } else {
                println!("   geom_type pointer: NULL!");
            }
        }

        let simulation = MuJoCoSimulation::new(model);

        app.insert_resource(simulation);
        app.add_systems(Update, simulate_physics);
        app.add_systems(Startup, setup_mujoco);
    }
}

pub fn simulate_physics(
    simulation: Res<MuJoCoSimulation>,
    settings: Res<MuJoCoPluginSettings>,
    mut bodies_query: Query<(Entity, &mut Transform, &MuJoCoBody)>,
    mut mujoco_resources: ResMut<MuJoCoResources>,
) {
    if settings.pause_simulation {
        return;
    }

    // Set control data using convenience method
    if !mujoco_resources.control.data.is_empty() {
        simulation.set_control(&mujoco_resources.control.data);
    }

    // Target fps in simulation - step until we've advanced enough time
    let sim_start = simulation.time();
    while simulation.time() - sim_start < 1.0 / settings.target_fps {
        simulation.step();
    }

    // Read state data using convenience methods
    mujoco_resources.state = MuJoCoState {
        sensor_data: simulation.sensordata(),
        qpos: simulation.qpos(),
        qvel: simulation.qvel(),
        cfrc_ext: simulation.cfrc_ext(),
    };

    // Get body transforms
    let xpos = simulation.xpos();
    let xquat = simulation.xquat();
    let model_ffi = simulation.model_ffi();

    // Get body parent IDs for transform calculations
    let body_parentid =
        unsafe { std::slice::from_raw_parts(model_ffi.body_parentid, model_ffi.nbody as usize) };

    // Update body transforms
    for (_, mut transform, body) in bodies_query.iter_mut() {
        let body_id = body.id as usize;
        let parent_id = body_parentid[body_id] as usize;

        let body_pos = xpos[body_id];
        let parent_pos = xpos[parent_id];

        let body_quat = xquat[body_id];
        let parent_quat = xquat[parent_id];

        // Convert to Bevy types
        let body_translation = Vec3::new(
            body_pos[0] as f32,
            body_pos[2] as f32, // Swap Y and Z for Bevy
            -body_pos[1] as f32,
        );

        let parent_translation = Vec3::new(
            parent_pos[0] as f32,
            parent_pos[2] as f32,
            -parent_pos[1] as f32,
        );

        // MuJoCo quaternions are [w, x, y, z]
        let body_rot = Quat::from_xyzw(
            body_quat[1] as f32,
            body_quat[3] as f32, // Swap Y and Z
            -body_quat[2] as f32,
            body_quat[0] as f32,
        );

        let parent_rot = Quat::from_xyzw(
            parent_quat[1] as f32,
            parent_quat[3] as f32,
            -parent_quat[2] as f32,
            parent_quat[0] as f32,
        );

        // Convert to parent-relative transform
        let parent_rot_inv = parent_rot.inverse();
        transform.translation = parent_rot_inv * (body_translation - parent_translation);
        transform.rotation = parent_rot_inv * body_rot;

        // Apply root body correction if needed
        if body.root_body {
            let correction = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            transform.translation = correction * transform.translation;
            transform.rotation = correction * transform.rotation;
        }
    }
}

fn setup_mujoco(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    simulation: Res<MuJoCoSimulation>,
) {
    let _data = simulation.data.lock().unwrap();
    let model_ffi = simulation.model.ffi();

    println!("🔍 MuJoCo model FFI info:");
    println!("   nbody: {}", model_ffi.nbody);
    println!("   ngeom: {}", model_ffi.ngeom);
    println!("   njnt: {}", model_ffi.njnt);

    // Get number of controls
    let num_controls = model_ffi.nu as usize;

    // Extract all bodies and geoms from model
    let bodies = Body::extract_all_bodies(model_ffi);
    let geoms = Geom::extract_all_geoms(model_ffi);

    println!(
        "🔧 Setup MuJoCo: {} bodies, {} geoms extracted",
        bodies.len(),
        geoms.len()
    );
    for (i, geom) in geoms.iter().enumerate() {
        println!(
            "   Geom {}: type={:?}, body_id={}",
            i, geom.geom_type, geom.body_id
        );
    }

    // Build body hierarchy
    let body_hierarchy = body::build_body_hierarchy(&bodies, &geoms);
    println!(
        "🌳 Built hierarchy with {} root nodes",
        body_hierarchy.len()
    );

    // Helper function to recursively spawn bodies with proper hierarchy
    // We collect spawn operations into a flat list with depth information
    #[derive(Clone)]
    struct SpawnOp {
        node: BodyNode,
        depth: usize,
    }

    fn collect_spawn_ops(node: &BodyNode, depth: usize) -> Vec<SpawnOp> {
        let mut ops = vec![SpawnOp {
            node: node.clone(),
            depth,
        }];

        for child in &node.children {
            ops.extend(collect_spawn_ops(child, depth + 1));
        }

        ops
    }

    // Collect all spawn operations in a flat list
    let mut all_spawn_ops = Vec::new();
    for root_node in &body_hierarchy {
        all_spawn_ops.extend(collect_spawn_ops(root_node, 0));
    }

    // Spawn world entity
    let world_entity = commands
        .spawn((
            Name::new("MuJoCo::World"),
            Transform::default(),
            Visibility::default(),
        ))
        .id();

    // Now spawn entities level by level to maintain proper parent-child relationships
    // Map body ID to entity ID
    use std::collections::HashMap;
    let mut body_to_entity: HashMap<i32, Entity> = HashMap::new();
    body_to_entity.insert(0, world_entity); // World body maps to world entity

    // Process spawn operations in order (they're already sorted by depth via collect_spawn_ops)
    for spawn_op in all_spawn_ops {
        let node = &spawn_op.node;
        let body = &node.body;
        let body_id = body.id;
        let depth = spawn_op.depth;

        // Get the transform for this body
        let mut body_transform = body_transform(body);

        // Apply root body correction for world-attached bodies (depth 0)
        let is_root = depth == 0;
        if is_root {
            let correction = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            body_transform.translation = correction * body_transform.translation;
            body_transform.rotation = correction * body_transform.rotation;
        }

        // Find parent entity
        let parent_entity = body_to_entity
            .get(&body.parent_id)
            .copied()
            .unwrap_or(world_entity);

        // Spawn the body entity
        let body_entity = commands
            .spawn((
                Name::new(format!("MuJoCo::Body::{}", body.name)),
                body_transform,
                Visibility::default(),
                MuJoCoBody {
                    id: body_id,
                    root_body: is_root,
                },
            ))
            .id();

        // Set parent relationship
        commands.entity(parent_entity).add_child(body_entity);

        // Store this mapping for children
        body_to_entity.insert(body_id, body_entity);

        // Spawn geoms as children of this body
        for geom in &node.geoms {
            let mesh = geom.to_mesh(Some(model_ffi));
            let geom_transform = geom_transform(geom);
            let material = geom_material(geom);

            let mesh_handle = meshes.add(mesh);
            let material_handle = materials.add(material);

            let geom_entity = commands
                .spawn((
                    Name::new(format!("MuJoCo::Geom::{}", geom.id)),
                    Mesh3d(mesh_handle),
                    MeshMaterial3d(material_handle),
                    geom_transform,
                ))
                .id();

            commands.entity(body_entity).add_child(geom_entity);
        }
    }

    // Insert resources after spawning
    commands.insert_resource(MuJoCoResources {
        geoms,
        control: MuJoCoControl {
            number_of_controls: num_controls,
            data: vec![0.0; num_controls],
        },
        ..default()
    });
}
