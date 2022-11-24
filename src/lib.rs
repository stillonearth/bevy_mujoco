use arrayvec::ArrayVec;
use serde::{Serialize, Serializer};
use trees::Tree;

use bevy::{
    ecs::system::EntityCommands,
    prelude::*,
    render::{
        mesh::{Indices, Mesh},
        render_resource::PrimitiveTopology,
    },
};
use bevy_obj::load_obj_from_bytes;

use mujoco_rs_sys::mjData;

use std::{cell::RefCell, io::Read};
use std::{
    fs::{self, File},
    rc::Rc,
};

mod mujoco_shape;

/// A component that holds a reference to a MuJoCo model, it needs to be wrapped to
/// implement Send and Sync for thread safety.
#[derive(Debug, Deref, DerefMut)]
struct ModelWrapper(mujoco_rust::Model);

unsafe impl Send for ModelWrapper {}
unsafe impl Sync for ModelWrapper {}

#[derive(Debug, Deref, DerefMut)]
struct DataWrapper(*mut mjData);

unsafe impl Send for DataWrapper {}
unsafe impl Sync for DataWrapper {}

/// A wrapped mujoco model
#[derive(Debug, Resource)]
pub struct MuJoCo {
    mj_model: ModelWrapper,
    mj_data: DataWrapper,
}

#[derive(Component)]
pub struct MuJoCoBody {
    pub id: i32,
}

#[derive(Debug, Clone)]
pub enum GeomType {
    PLANE = 0,   // plane
    HFIELD,      // height field
    SPHERE,      // sphere
    CAPSULE,     // capsule
    ELLIPSOID,   // ellipsoid
    CYLINDER,    // cylinder
    BOX,         // box
    MESH,        // mesh
    NONE = 1001, // missing geom type
}

impl GeomType {
    fn from<T>(val: usize) -> GeomType {
        match val {
            0 => GeomType::PLANE,
            1 => GeomType::HFIELD,
            2 => GeomType::SPHERE,
            3 => GeomType::CAPSULE,
            4 => GeomType::ELLIPSOID,
            5 => GeomType::CYLINDER,
            6 => GeomType::BOX,
            7 => GeomType::MESH,
            1001 => GeomType::NONE,
            _ => panic!("Invalid value for GeomType"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Geom {
    pub id: i32,
    pub name: String,
    pub geom_type: GeomType,
    pub body_id: i32,
    pub parent_id: i32,
    pub pos: [f32; 3],
    pub quat: [f32; 4],
    pub size: [f32; 3],
    pub color: [f32; 4],
    pub mesh: Option<MuJoCoMesh>,
}

impl Geom {
    /// Get bevy mesh for the body
    pub fn mesh(&self, assets_path: String) -> Mesh {
        match self.geom_type {
            GeomType::PLANE => Mesh::from(shape::Box::new(
                self.size[0] as f32,
                self.size[1] as f32,
                self.size[2] as f32,
            )),
            GeomType::BOX => Mesh::from(shape::Box::new(
                self.size[0] as f32,
                self.size[1] as f32,
                self.size[2] as f32,
            )),
            GeomType::SPHERE => Mesh::from(shape::Icosphere {
                radius: self.size[0] as f32,
                ..default()
            }),
            GeomType::CAPSULE => Mesh::from(shape::Capsule {
                radius: self.size[0] as f32,
                depth: (self.size[1] * 2.0) as f32,
                ..default()
            }),
            GeomType::ELLIPSOID => todo!(),
            GeomType::CYLINDER => Mesh::from(mujoco_shape::Cylinder {
                radius: self.size[0] as f32,
                height: (self.size[2] * 2.0) as f32,
                ..default()
            }),

            GeomType::MESH => self.mesh.clone().unwrap().mesh(assets_path),
            // --- NOT IMPLEMENTED ---
            GeomType::NONE => todo!(),
            GeomType::HFIELD => todo!(),
        }
    }

    pub fn rotation(&self) -> Quat {
        Quat::from_xyzw(self.quat[0], self.quat[1], self.quat[2], self.quat[3])
    }

    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.pos[0], self.pos[1], self.pos[2])
    }

    pub fn transform(&self) -> Transform {
        Transform {
            translation: self.translation(),
            rotation: self.rotation(),
            ..default()
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Body {
    pub id: i32,
    pub name: String,
    pub parent_id: i32,
    pub root_id: i32,
    pub geom_n: i32,
    pub simple: u8,
    pub pos: [f32; 3],
    pub quat: [f32; 4],
    pub mass: f32,
}

impl Body {
    pub fn rotation(&self) -> Quat {
        Quat::from_xyzw(self.quat[0], self.quat[1], self.quat[2], self.quat[3])
    }

    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.pos[0], self.pos[1], self.pos[2])
    }

    pub fn transform(&self) -> Transform {
        // print!("{:?}", self.rotation());

        Transform {
            translation: self.translation(),
            rotation: self.rotation() * self.rotation(),
            ..default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct MuJoCoMesh {
    pub vertices: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub name: String,
}

impl MuJoCoMesh {
    /// This does not work correctly yet
    pub fn _mesh(self) -> Mesh {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        // let mut uvs: Vec<[f32; 2]> = vec![];
        mesh.set_indices(Some(Indices::U32(self.indices)));
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.vertices);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh
    }

    pub fn mesh(self, assets_path: String) -> Mesh {
        // load_obj_from_bytes

        let filename = format!("{}/{}.obj", assets_path, self.name);
        let metadata = fs::metadata(&filename).expect("unable to read metadata");

        let mut f = File::open(&filename).expect("no file found");
        let mut buffer = vec![0; metadata.len() as usize];
        f.read_exact(&mut buffer).expect("buffer overflow");

        load_obj_from_bytes(&buffer).unwrap()
    }
}

pub struct Local<T>(T);

pub trait LocalFloat {
    fn to_f32(&self) -> f32;
}
impl LocalFloat for Local<f32> {
    fn to_f32(&self) -> f32 {
        self.0
    }
}

impl LocalFloat for Local<f64> {
    fn to_f32(&self) -> f32 {
        self.0 as f32
    }
}

impl LocalFloat for Local<i32> {
    fn to_f32(&self) -> f32 {
        self.0 as f32
    }
}

#[derive(Deref, DerefMut)]
pub struct BodyTree(pub Tree<Body>);

impl Serialize for BodyTree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct BodyTreeSerialized {
            pub body: Body,
            pub children: Vec<BodyTreeSerialized>,
        }

        fn collect_children(tree: &Tree<Body>) -> Vec<BodyTreeSerialized> {
            let mut children = vec![];
            let mut tree = tree.clone();
            loop {
                let leaf = tree.pop_back();
                if leaf.is_none() {
                    break;
                }
                let leaf = leaf.unwrap();
                children.push(BodyTreeSerialized {
                    body: leaf.data().clone(),
                    children: collect_children(&leaf),
                });
            }
            children
        }

        let item = Some(BodyTreeSerialized {
            body: self.0.data().clone(),
            children: collect_children(&self.0),
        });

        match item {
            Some(item) => item.serialize(serializer),
            None => serializer.serialize_none(),
        }
    }
}

/// Copy MuJoCo array into Vec<f64>
fn extract_vector_float<T>(vec: *mut T, element_size: usize, n_entries: usize) -> Vec<f32>
where
    T: LocalFloat,
{
    let mut result_vec: Vec<f32> = Vec::new();

    unsafe {
        let entries = vec;
        for i in 0..n_entries {
            for j in 0..element_size {
                result_vec.push((*entries.add(i * element_size + j)).to_f32());
            }
        }
    }

    result_vec
}

/// Get String from an pointer to a null-terminated char array pointer
fn extract_string(array: *mut i8) -> String {
    let mut name = String::new();
    let mut i = 0;
    loop {
        let char = unsafe { *array.offset(i) } as i32;
        if char == 0 {
            break;
        }
        name.push(char::from_u32(char as u32).unwrap());
        i += 1;
    }
    name
}

/// Extract vertices and normals from a MuJoCo mesh
fn extract_mesh_attribute(array: *mut f32, offset: usize, count: usize) -> Vec<[f32; 3]> {
    let mut points: Vec<[f32; 3]> = vec![];

    let point_array =
        unsafe { extract_vector_float(array.add(offset) as *mut Local<f32>, 3, count) };
    for p in point_array.chunks(3) {
        let p: [f32; 3] = p.try_into().unwrap();
        points.push(p);
    }

    points
}

/// Extract normals from a MuJoCo mesh
fn extract_indices(array: *mut i32, face_addr: usize, face_num: usize) -> Vec<u32> {
    let mut indices: Vec<u32> = vec![];
    for j in face_addr..face_addr + face_num {
        unsafe {
            let face = array.add(j);
            indices.push(*face.add(0) as u32);
            indices.push(*face.add(1) as u32);
            indices.push(*face.add(2) as u32);
        }
    }
    indices
}

/// Adjust format of 3-Vec from Mujoco to bevy
fn replace_values_vec3(arr: &mut [f32; 3], i1: usize, i2: usize) {
    let c_1 = arr[i1];
    let c_2 = arr[i2];
    arr[i1] = c_2;
    arr[i2] = c_1;
}

/// Adjust format of quaternion from Mujoco to bevy
fn replace_values_vec4(arr: &mut [f32; 4], i1: usize, i2: usize) {
    let c_1 = arr[i1];
    let c_2 = arr[i2];
    arr[i1] = c_2;
    arr[i2] = c_1;
}

fn collect_children(parent_leaf: &mut Tree<Body>, bodies: &Vec<Body>) {
    let parent_id = parent_leaf.data().id;
    let children: Vec<Body> = bodies
        .iter()
        .filter(|child| {
            // this will make world plane and bodies different entities
            // otherwise this would be a tree with one root — world plane
            child.parent_id == parent_id && child.id != child.parent_id && child.parent_id != 0
        })
        .cloned()
        .collect();

    for child in children {
        let mut leaf = Tree::<Body>::new(child.clone());
        collect_children(&mut leaf, bodies);
        parent_leaf.push_back(leaf);
    }
}

impl MuJoCo {
    /// Creates a wrapped mujoco model from an MJCF file
    pub fn new_from_xml(path: &str) -> Self {
        let mj_model = ModelWrapper(mujoco_rust::Model::from_xml(path, true).unwrap());
        let mj_data = unsafe { mujoco_rs_sys::no_render::mj_makeData(mj_model.ptr()) };
        let mj_data = DataWrapper(mj_data);

        Self { mj_model, mj_data }
    }

    /// Returns positions of bodies
    pub fn xpos(&self) -> Vec<[f32; 3]> {
        let mj_data = self.mj_data.0;
        let raw_vec = unsafe { (*mj_data).xpos };

        let raw_xpos = extract_vector_float(raw_vec as *mut Local<f64>, 3, self.ngeom());

        let mut xpos: Vec<[f32; 3]> = Vec::new();

        for i in 0..self.ngeom() {
            let entry: [f32; 3] = [raw_xpos[i * 3], raw_xpos[i * 3 + 1], raw_xpos[i * 3 + 2]];
            xpos.push(entry);
        }

        xpos
    }

    /// Returns rotations of bodies
    pub fn xquat(&self) -> Vec<[f32; 4]> {
        let mj_data = self.mj_data.0;
        let raw_vec = unsafe { (*mj_data).xquat };
        let raw_quat = extract_vector_float(raw_vec as *mut Local<f64>, 4, self.ngeom());
        let mut xquat: Vec<[f32; 4]> = Vec::new();

        for i in 0..self.ngeom() {
            let entry: [f32; 4] = [
                raw_quat[i * 4 + 3],
                raw_quat[i * 4 + 1],
                raw_quat[i * 4 + 2],
                raw_quat[i * 4],
            ];
            xquat.push(entry);
        }

        xquat
    }

    /// Returns a number of geoms in the model
    pub fn ngeom(&self) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*mj_model.ptr()).ngeom as usize
        }
    }

    /// Returns a number of bodies in the model
    pub fn nbody(&self) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*mj_model.ptr()).nbody as usize
        }
    }

    /// Returns a number of meshes
    pub fn nmesh(&self) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*mj_model.ptr()).nmesh as usize
        }
    }

    /// Returns a number of vertices in all meshes
    pub fn nmeshvert(&self) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*mj_model.ptr()).nmeshvert as usize
        }
    }

    // Returns start index of vertices for a given mesh
    pub fn mesh_vertadr(&self, mesh_id: usize) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*(*mj_model.ptr()).mesh_vertadr.add(mesh_id)) as usize
        }
    }

    // Returns number of vertices for a given mesh
    pub fn mesh_vertnum(&self, mesh_id: usize) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*(*mj_model.ptr()).mesh_vertnum.add(mesh_id)) as usize
        }
    }

    // Returns start index of faces for a given mesh
    pub fn mesh_faceadr(&self, mesh_id: usize) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*(*mj_model.ptr()).mesh_faceadr.add(mesh_id)) as usize
        }
    }

    // Returns number of faces for a given mesh
    pub fn mesh_facenum(&self, mesh_id: usize) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*(*mj_model.ptr()).mesh_facenum.add(mesh_id)) as usize
        }
    }

    /// Returns a number of triangular faces in all meshes
    pub fn nmeshface(&self) -> usize {
        unsafe {
            let mj_model = &self.mj_model;
            (*mj_model.ptr()).nmeshface as usize
        }
    }

    /// Advance simulation by one step
    pub fn step(&self) {
        unsafe {
            let mj_model = &self.mj_model;
            let mj_data = self.mj_data.0;
            mujoco_rs_sys::no_render::mj_step(mj_model.ptr(), mj_data);
        };
    }

    /// Simulation time in seconds
    pub fn time(&self) -> f64 {
        unsafe {
            let mj_data = self.mj_data.0;
            (*mj_data).time
        }
    }

    pub fn names(&self) -> Vec<String> {
        let mj_model = &self.mj_model;
        let mj_model = unsafe { *mj_model.ptr() };

        let mut names: Vec<String> = Vec::new();
        let mut j: isize = 0;
        for _ in 0..mj_model.nnames {
            names.push(String::new());

            loop {
                let char = unsafe { *mj_model.names.offset(j as isize) };
                if char == 0 {
                    j += 1;
                    break;
                }
                let idx = names.len() - 1;
                if char != 0 {
                    names[idx] = format!("{}{}", names[names.len() - 1], (char as u8) as char);
                }
                j += 1;
            }
        }

        names
    }

    /// Get geoms of the model
    pub fn geoms(&self) -> Vec<Geom> {
        let mj_model = &self.mj_model;
        let mj_model = unsafe { *mj_model.ptr() };
        let n_geom = self.ngeom();

        let mut geoms: Vec<Geom> = Vec::new();
        let meshes = self.meshes();

        let body_pos_vec: Vec<f32> =
            extract_vector_float(mj_model.geom_pos as *mut Local<f64>, 3, n_geom);
        let body_quat_vec: Vec<f32> =
            extract_vector_float(mj_model.geom_quat as *mut Local<f64>, 4, n_geom);
        let body_size_vec: Vec<f32> =
            extract_vector_float(mj_model.geom_size as *mut Local<f64>, 4, n_geom);
        let body_rgba_vec: Vec<f32> =
            extract_vector_float(mj_model.geom_rgba as *mut Local<f32>, 4, n_geom);

        for i in 0..n_geom {
            // position
            let pos_array = body_pos_vec[i * 3..i * 3 + 3].to_vec();
            let pos_array: ArrayVec<f32, 3> = pos_array.into_iter().collect();
            let pos_array: [f32; 3] = pos_array.into_inner().unwrap();

            // quaternion
            let quat_array = body_quat_vec[i * 4..i * 4 + 4].to_vec();
            let quat_array: ArrayVec<f32, 4> = quat_array.into_iter().collect();
            let quat_array: [f32; 4] = quat_array.into_inner().unwrap();

            // size
            let size_array = body_size_vec[i * 3..i * 3 + 3].to_vec();
            let size_array: ArrayVec<f32, 3> = size_array.into_iter().collect();
            let size_array: [f32; 3] = size_array.into_inner().unwrap();

            // color
            let color_array = body_rgba_vec[i * 4..i * 4 + 4].to_vec();
            let color_array: ArrayVec<f32, 4> = color_array.into_iter().collect();
            let color_array: [f32; 4] = color_array.into_inner().unwrap();

            let mut mesh: Option<MuJoCoMesh> = None;
            let mesh_id = unsafe { *mj_model.geom_dataid.add(i) };
            if mesh_id != -1 {
                mesh = Some(meshes[mesh_id as usize].clone());
            }

            // name
            let geom_name_idx = unsafe { *mj_model.name_geomadr.add(i) as usize };

            let mut geom_body = unsafe {
                Geom {
                    id: i as i32,
                    geom_type: GeomType::from::<usize>(*mj_model.geom_type.add(i) as usize),
                    body_id: *mj_model.geom_bodyid.add(i) as i32,
                    parent_id: *mj_model.body_parentid.add(i) as i32,
                    pos: pos_array,
                    quat: quat_array,
                    size: size_array,
                    color: color_array,
                    mesh,
                    name: extract_string(mj_model.names.add(geom_name_idx)),
                }
            };

            // bevy uses different coordinate system than mujoco so we need to swap y and z
            // position, and adjust quaternion
            replace_values_vec3(&mut geom_body.pos, 1, 2);
            replace_values_vec3(&mut geom_body.size, 1, 2);
            replace_values_vec4(&mut geom_body.quat, 0, 3);

            geoms.push(geom_body);
        }
        geoms
    }

    /// Get bodies of the model
    pub fn bodies(&self) -> Vec<Body> {
        let mj_model = &self.mj_model;
        let mj_model = unsafe { *mj_model.ptr() };
        let n_body = self.nbody();

        let body_pos_vec: Vec<f32> =
            extract_vector_float(mj_model.body_pos as *mut Local<f64>, 3, n_body);
        let body_quat_vec: Vec<f32> =
            extract_vector_float(mj_model.body_quat as *mut Local<f64>, 4, n_body);

        let mut bodies: Vec<Body> = Vec::new();
        for i in 0..n_body {
            // position
            let pos_array = body_pos_vec[i * 3..i * 3 + 3].to_vec();
            let pos_array: ArrayVec<f32, 3> = pos_array.into_iter().collect();
            let pos_array: [f32; 3] = pos_array.into_inner().unwrap();

            // quaternion
            let quat_array = body_quat_vec[i * 4..i * 4 + 4].to_vec();
            let quat_array: ArrayVec<f32, 4> = quat_array.into_iter().collect();
            let quat_array: [f32; 4] = quat_array.into_inner().unwrap();

            // metadata
            let name_idx = unsafe { *mj_model.name_bodyadr.add(i) as usize };

            let mut geom_body = unsafe {
                Body {
                    id: i as i32,
                    parent_id: *mj_model.body_parentid.add(i) as i32,
                    root_id: *mj_model.body_rootid.add(i) as i32,
                    geom_n: *mj_model.body_geomnum.add(i) as i32,
                    simple: *mj_model.body_simple.add(i) as u8,
                    pos: pos_array,
                    quat: quat_array,
                    mass: *mj_model.body_mass.add(i) as f32,
                    name: extract_string(mj_model.names.add(name_idx)),
                }
            };

            replace_values_vec3(&mut geom_body.pos, 1, 2);
            replace_values_vec4(&mut geom_body.quat, 0, 3);

            bodies.push(geom_body);
        }
        bodies
    }

    pub fn body_tree(&self) -> Vec<BodyTree> {
        let mut trees: Vec<BodyTree> = vec![];
        let bodies = self.bodies();

        let root_bodies = bodies.iter().filter(|body| body.parent_id == 0);

        for body in root_bodies {
            let mut root_leaf: BodyTree = BodyTree(Tree::new(body.clone()));
            collect_children(&mut root_leaf.0, &bodies);

            trees.push(root_leaf);
        }

        trees
    }

    /// Returns list of meshes in a scene
    pub fn meshes(&self) -> Vec<MuJoCoMesh> {
        let mut meshes = Vec::new();
        let mj_model = &self.mj_model;
        let mj_model = unsafe { *mj_model.ptr() };

        for i in 0..self.nmesh() {
            let vertadr = self.mesh_vertadr(i);
            let vertnum = self.mesh_vertnum(i);
            let faceadr = self.mesh_faceadr(i);
            let facenum = self.mesh_facenum(i);

            // mesh data
            let vertices = extract_mesh_attribute(mj_model.mesh_vert, vertadr, vertnum);
            let normals = extract_mesh_attribute(mj_model.mesh_normal, vertadr, vertnum);
            let indices = extract_indices(mj_model.mesh_face, faceadr, facenum);

            // metadata
            let mesh_name_idx = unsafe { *mj_model.name_meshadr.add(i) as usize };
            let name = unsafe { extract_string(mj_model.names.add(mesh_name_idx)) };

            let mujoco_mesh = MuJoCoMesh {
                vertices,
                normals,
                indices,
                name,
            };

            meshes.push(mujoco_mesh);
        }
        meshes
    }
}

#[derive(Resource)]
pub struct MuJoCoPluginSettings {
    pub model_xml_path: String,
    pub model_assets_path: String,
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

#[allow(unused_mut)]
#[allow(unused_variables)]
#[allow(unreachable_code)]
fn simulate_physics(
    mujoco: ResMut<MuJoCo>,
    mut bodies_query: Query<(Entity, &mut Transform, &MuJoCoBody)>,
    mujoco_resources: Res<MuJoCoResources>,
) {
    return;

    // Target 60 fps in simulation
    let _sim_start = mujoco.time();
    // while mujoco.time() - sim_start < 1.0 / 60.0 {
    //     mujoco.step();
    // }

    let positions = mujoco.xpos();
    let rotations = mujoco.xquat();

    for (_, mut transform, body) in bodies_query.iter_mut() {
        let body_id = body.id as usize;
        if body_id >= positions.len() {
            continue;
        }

        let pos = positions[body_id];
        let rot = rotations[body_id];

        let dynamic_translation = Vec3::from_array(pos);
        let geom_translation = Vec3::from_array(mujoco_resources.geoms[body_id].pos);
        let _body_translation = Vec3::from_array(mujoco_resources.bodies[body_id].pos);

        transform.translation = geom_translation + dynamic_translation;
        // if let GeomType::PLANE = mujoco_resources.geoms[body_id].geom_type {
        //     transform.translation.y += mujoco_resources.geoms[body_id].size[1];
        // }

        let dynamic_rotation = Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
        let body_rotation = Quat::from_xyzw(
            mujoco_resources.geoms[body_id].quat[0],
            mujoco_resources.geoms[body_id].quat[1],
            mujoco_resources.geoms[body_id].quat[2],
            mujoco_resources.geoms[body_id].quat[3],
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
        bodies: bodies,
    });

    // this is a closure that can call itself recursively
    struct SpawnEntities<'s> {
        f: &'s dyn Fn(&SpawnEntities, BodyTree, &mut EntityCommands),
    }

    impl SpawnEntities<'_> {
        /// Spawn a bevy entity for MuJoCo body
        fn spawn_body<'s>(
            &self,
            entity_commands: &mut EntityCommands,
            body: &Body,
            geoms: &Vec<Geom>,
            settings: &Res<MuJoCoPluginSettings>,
            meshes: &Rc<RefCell<ResMut<Assets<Mesh>>>>,
            materials: &Rc<RefCell<ResMut<Assets<StandardMaterial>>>>,
        ) {
            let body_id = body.id;
            let geom = geoms.iter().find(|geom| geom.body_id == body_id).unwrap();

            // Extracting mesh from mujoco object doesn't work correctly
            // Insted load the obj directly with `bevy_obj`
            let mesh = geom.mesh(settings.model_assets_path.clone());

            let body_transform = body.transform();
            let geom_transform = geom.clone().transform();

            // deref
            // let mut entity_commands = entity_commands.borrow_mut();
            let mut meshes = meshes.borrow_mut();
            let mut materials = materials.borrow_mut();

            // let mut binding = entity_commands.borrow_mut();
            let mut binding = entity_commands.commands().spawn(PbrBundle {
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
                    rotation: (body_transform.rotation * geom_transform.rotation)
                        * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)
                        * Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
                    ..default()
                },
                ..default()
            });

            binding
                .insert(MuJoCoBody { id: body_id })
                .insert(Name::new(format!("MuJoCo::body_{}", body.name)));

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
        f: &|func, mut body, entity_commands| {
            let root_leaf = body.data();
            // TODO: spawn_body inserts a new entity and return a cursor to a leaf in a tree
            // It does not return entity_commands
            func.spawn_body(
                entity_commands,
                root_leaf,
                &geoms,
                &settings,
                &meshes,
                &materials,
            );

            entity_commands.with_children(|children| loop {
                let leaf = body.pop_back();
                if leaf.is_none() {
                    break;
                }

                // ---
                // spawn_body should return entity_commands
                // entity_commands is supposted to point to a position in entity tree
                // ---

                let mut entity_commands = children.spawn_empty();
                (func.f)(func, BodyTree(leaf.unwrap()), &mut entity_commands);
            });
        },
    };

    let mut commands = commands.borrow_mut();
    // each mujoco body is defined as a tree
    for body in mujoco.body_tree() {
        let entity_commands = &mut commands.spawn_empty();
        (spawn_entities.f)(&spawn_entities, body, entity_commands);
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::{assert_eq, assert_ne};
    use ron::ser::{to_string_pretty, PrettyConfig};

    use super::*;

    #[test]
    fn geoms() {
        let model = &MuJoCo::new_from_xml("assets/mjcf/simple_1.xml");
        let geoms = model.bodies();

        assert!(!geoms.is_empty());
    }

    #[test]
    fn bodies() {
        let model = &MuJoCo::new_from_xml("assets/mujoco_menagerie/unitree_a1/scene.xml");
        let bodies = model.bodies();

        assert!(bodies.len() == 14);

        let body_names = bodies
            .iter()
            .map(|body| body.name.clone())
            .collect::<Vec<String>>();

        assert!(
            body_names
                == vec![
                    "world", "trunk", "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh",
                    "FL_calf", "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf",
                ]
        );
    }

    #[test]
    fn bodies_tree() {
        let model = &MuJoCo::new_from_xml("assets/mujoco_menagerie/unitree_a1/scene.xml");
        let bodies_tree = model.body_tree();

        assert!(bodies_tree.len() == 2);

        fn print_body_tree_level(mut tree: Tree<Body>, level: usize) -> String {
            let mut ret = format!("{}{}\n", "--".repeat(level), tree.data().name);

            loop {
                let leaf = tree.pop_back();
                if leaf.is_none() {
                    break;
                }
                ret = format!("{}{}", ret, print_body_tree_level(leaf.unwrap(), level + 1));
            }

            ret
        }

        let tree_structure = print_body_tree_level(bodies_tree[1].clone(), 0);
        let exprected_tree_structure = r"trunk
--RL_hip
----RL_thigh
------RL_calf
--RR_hip
----RR_thigh
------RR_calf
--FL_hip
----FL_thigh
------FL_calf
--FR_hip
----FR_thigh
------FR_calf
";

        assert_eq!(tree_structure, exprected_tree_structure);
    }

    #[test]
    fn meshes() {
        let model = &MuJoCo::new_from_xml("assets/mujoco_menagerie/unitree_a1/scene.xml");
        let meshes = model.meshes();

        assert!(meshes.len() == model.nmesh());
    }

    #[test]
    fn names() {
        let model = &MuJoCo::new_from_xml("assets/mujoco_menagerie/unitree_a1/scene.xml");
        let names = model.names();

        assert!(!names.is_empty());
    }
}
