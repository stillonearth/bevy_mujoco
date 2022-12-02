use arrayvec::ArrayVec;
use itertools::Itertools;
use serde::{Serialize, Serializer};
use trees::Tree;

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, Mesh},
        render_resource::PrimitiveTopology,
    },
};
use bevy_obj::load_obj_from_bytes;

use mujoco_rs_sys::mjData;

use std::io::Read;
use std::{
    cmp::Ordering,
    fs::{self, File},
};

use crate::mujoco_shape;

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub size: [f64; 3],
    pub color: [f32; 4],
    pub mesh: Option<MuJoCoMesh>,
    pub geom_group: i32,
    pub geom_contype: i32,
}

impl Geom {
    /// Get bevy mesh for the body
    pub fn mesh(&self, assets_path: String) -> Mesh {
        match self.geom_type {
            GeomType::PLANE => {
                let plane_size = if self.size[0] > 0.0 {
                    self.size[0]
                } else {
                    // MuJoCo size 0 means infinite
                    1e6
                };
                Mesh::from(shape::Plane {
                    size: plane_size as f32,
                })
            }
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
                depth: (self.size[2] * 2.0) as f32,
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
        Quat::from_xyzw(
            self.quat[1] as f32,
            self.quat[0] as f32,
            self.quat[2] as f32,
            -self.quat[3] as f32,
        )
    }

    /// bevy and mujoco treat object frame differently, this function converts
    pub fn correction(&self) -> Vec3 {
        match self.geom_type {
            GeomType::BOX => Vec3::new(0.0, (self.size[1] / 2.0) as f32, 0.0),
            GeomType::CAPSULE => Vec3::new(0.0, (self.size[1] * 2.0) as f32, 0.0),
            GeomType::CYLINDER => Vec3::new(0.0, (self.size[2] * 2.0) as f32, 0.0),
            _ => Vec3::ZERO,
        }
    }

    pub fn translation(&self) -> Vec3 {
        let translation = Vec3::new(self.pos[0] as f32, self.pos[2] as f32, self.pos[1] as f32);
        translation - self.correction()
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
    pub geom_n: i32,
    pub geom_addr: i32,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
}

impl Body {
    pub fn rotation(&self) -> Quat {
        Quat::from_xyzw(
            self.quat[1] as f32,
            self.quat[0] as f32,
            self.quat[2] as f32,
            -self.quat[3] as f32,
        )
    }

    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.pos[0] as f32, self.pos[2] as f32, self.pos[1] as f32)
    }

    pub fn transform(&self) -> Transform {
        Transform {
            translation: self.translation(),
            rotation: self.rotation(),
            ..default()
        }
    }

    pub fn geoms(&self, geoms: &[Geom]) -> Vec<Geom> {
        let mut body_geoms = Vec::new();
        for i in 0..self.geom_n {
            let geom = &geoms[(self.geom_addr + i) as usize];
            body_geoms.push(geom.clone());
        }
        body_geoms
    }

    pub fn render_geom(&self, geoms: &[Geom]) -> Option<Geom> {
        let geom_query = geoms.iter().filter(|g| g.body_id == self.id);
        if geom_query.clone().count() == 1 {
            return Some(geom_query.clone().last().unwrap().clone());
        }

        let geoms = self.geoms(geoms);

        // This is questionable, but it seems to work
        let geom = geoms
            .iter()
            .filter(|g| g.geom_group < 3)
            .sorted_by(|g1, g2| {
                if g1.geom_type == GeomType::MESH && g2.geom_type != GeomType::MESH {
                    return Ordering::Less;
                }
                if g1.geom_type != GeomType::MESH && g2.geom_type == GeomType::MESH {
                    return Ordering::Greater;
                }

                if g1.geom_group < g2.geom_group {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
            .last()
            .cloned();

        geom
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
    fn to_f64(&self) -> f64;
    fn to_f32(&self) -> f32;
}
impl LocalFloat for Local<f32> {
    fn to_f32(&self) -> f32 {
        self.0
    }
    fn to_f64(&self) -> f64 {
        self.0 as f64
    }
}

impl LocalFloat for Local<f64> {
    fn to_f32(&self) -> f32 {
        self.0 as f32
    }

    fn to_f64(&self) -> f64 {
        self.0
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
fn extract_vector_float_f64<T>(vec: *mut T, element_size: usize, n_entries: usize) -> Vec<f64>
where
    T: LocalFloat,
{
    let mut result_vec: Vec<f64> = Vec::new();

    unsafe {
        let entries = vec;
        for i in 0..n_entries {
            for j in 0..element_size {
                result_vec.push((*entries.add(i * element_size + j)).to_f64());
            }
        }
    }

    result_vec
}

/// Copy MuJoCo array into Vec<f64>
fn extract_vector_float_f32<T>(vec: *mut T, element_size: usize, n_entries: usize) -> Vec<f32>
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
        unsafe { extract_vector_float_f32(array.add(offset) as *mut Local<f32>, 3, count) };
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
fn replace_values_vec3(arr: &mut [f64; 3], i1: usize, i2: usize) {
    let c_1 = arr[i1];
    let c_2 = arr[i2];
    arr[i1] = c_2;
    arr[i2] = c_1;
}

/// Adjust format of quaternion from Mujoco to bevy
fn replace_values_vec4(arr: &mut [f64; 4], i1: usize, i2: usize) {
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

impl MuJoCo {
    /// Creates a wrapped mujoco model from an MJCF file
    pub fn new_from_xml(path: &str) -> Self {
        let mj_model = ModelWrapper(mujoco_rust::Model::from_xml(path, true).unwrap());
        let mj_data = unsafe { mujoco_rs_sys::no_render::mj_makeData(mj_model.ptr()) };
        let mj_data = DataWrapper(mj_data);

        Self { mj_model, mj_data }
    }

    /// Returns positions of bodies in inertial frame
    pub fn xpos(&self) -> Vec<[f64; 3]> {
        let mj_data = self.mj_data.0;
        let raw_vec = unsafe { (*mj_data).xpos };

        let raw_xpos = extract_vector_float_f64(raw_vec as *mut Local<f64>, 3, self.ngeom());

        let mut xpos: Vec<[f64; 3]> = Vec::new();

        for i in 0..self.ngeom() {
            let entry: [f64; 3] = [raw_xpos[i * 3], raw_xpos[i * 3 + 2], raw_xpos[i * 3 + 1]];
            xpos.push(entry);
        }

        xpos
    }

    /// Returns rotations of bodies
    pub fn xquat(&self) -> Vec<[f64; 4]> {
        let mj_data = self.mj_data.0;
        let raw_vec = unsafe { (*mj_data).xquat };
        let raw_quat = extract_vector_float_f64(raw_vec as *mut Local<f64>, 4, self.ngeom());
        let mut xquat: Vec<[f64; 4]> = Vec::new();

        for i in 0..self.ngeom() {
            let entry: [f64; 4] = [
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
                let char = unsafe { *mj_model.names.offset(j) };
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

        let body_pos_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.geom_pos as *mut Local<f64>, 3, n_geom);
        let body_quat_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.geom_quat as *mut Local<f64>, 4, n_geom);
        let body_size_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.geom_size as *mut Local<f64>, 4, n_geom);
        let body_rgba_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.geom_rgba as *mut Local<f32>, 4, n_geom);

        for i in 0..n_geom {
            // position
            let pos_array = body_pos_vec[i * 3..i * 3 + 3].to_vec();
            let pos_array: ArrayVec<f64, 3> = pos_array.into_iter().collect();
            let mut pos_array: [f64; 3] = pos_array.into_inner().unwrap();

            // quaternion
            let quat_array = body_quat_vec[i * 4..i * 4 + 4].to_vec();
            let quat_array: ArrayVec<f64, 4> = quat_array.into_iter().collect();
            let quat_array: [f64; 4] = quat_array.into_inner().unwrap();

            // size
            let size_array = body_size_vec[i * 3..i * 3 + 3].to_vec();
            let size_array: ArrayVec<f64, 3> = size_array.into_iter().collect();
            let size_array: [f64; 3] = size_array.into_inner().unwrap();

            // color
            let color_array = body_rgba_vec[i * 4..i * 4 + 4].to_vec();
            let color_array: ArrayVec<f64, 4> = color_array.into_iter().collect();
            let color_array: [f64; 4] = color_array.into_inner().unwrap();
            let color_array: [f32; 4] = [
                color_array[0] as f32,
                color_array[1] as f32,
                color_array[2] as f32,
                color_array[3] as f32,
            ];

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
                    body_id: *mj_model.geom_bodyid.add(i),
                    geom_group: *mj_model.geom_group.add(i),
                    geom_contype: *mj_model.geom_contype.add(i),
                    pos: pos_array,
                    quat: quat_array,
                    size: size_array,
                    color: color_array,
                    mesh,
                    name: extract_string(mj_model.names.add(geom_name_idx)),
                }
            };

            if geom_body.geom_type == GeomType::PLANE {
                geom_body.color = [0.0, 0.0, 0.5, 0.0];
            }

            // bevy uses different coordinate system than mujoco so we need to swap y and z
            // position, and adjust quaternion
            replace_values_vec3(&mut geom_body.pos, 1, 2);
            replace_values_vec3(&mut geom_body.size, 1, 2);
            replace_values_vec4(&mut geom_body.quat, 0, 3);

            // bevy's origin lower face while mujoco's origin is upper face
            pos_array[2] -= size_array[2];

            geoms.push(geom_body);
        }
        geoms
    }

    /// Get bodies of the model
    pub fn bodies(&self) -> Vec<Body> {
        let mj_model = &self.mj_model;
        let mj_model = unsafe { *mj_model.ptr() };
        let n_body = self.nbody();

        let body_pos_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.body_pos as *mut Local<f64>, 3, n_body);
        let body_quat_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.body_quat as *mut Local<f64>, 4, n_body);

        let _body_ipos_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.body_ipos as *mut Local<f64>, 3, n_body);
        let _body_iquat_vec: Vec<f64> =
            extract_vector_float_f64(mj_model.body_iquat as *mut Local<f64>, 4, n_body);

        let mut bodies: Vec<Body> = Vec::new();
        for i in 0..n_body {
            // position
            let pos_array = body_pos_vec[i * 3..i * 3 + 3].to_vec();
            let pos_array: ArrayVec<f64, 3> = pos_array.into_iter().collect();
            let pos_array: [f64; 3] = pos_array.into_inner().unwrap();

            // quaternion
            let quat_array = body_quat_vec[i * 4..i * 4 + 4].to_vec();
            let quat_array: ArrayVec<f64, 4> = quat_array.into_iter().collect();
            let quat_array: [f64; 4] = quat_array.into_inner().unwrap();

            // metadata
            let name_idx = unsafe { *mj_model.name_bodyadr.add(i) as usize };

            let mut geom_body = unsafe {
                Body {
                    id: i as i32,
                    parent_id: *mj_model.body_parentid.add(i),
                    geom_n: *mj_model.body_geomnum.add(i),
                    geom_addr: *mj_model.body_geomadr.add(i),
                    pos: pos_array,
                    quat: quat_array,
                    name: extract_string(mj_model.names.add(name_idx)),
                }
            };

            replace_values_vec3(&mut geom_body.pos, 1, 2);
            replace_values_vec4(&mut geom_body.quat, 0, 3);

            bodies.push(geom_body);
        }
        bodies
    }

    /// Returns tree of bodies
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

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
