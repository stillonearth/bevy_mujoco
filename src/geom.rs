use bevy::prelude::*;
use mujoco_rs::mujoco_c::*;

/// Geom types in MuJoCo
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomType {
    Plane,
    HField,
    Sphere,
    Capsule,
    Ellipsoid,
    Cylinder,
    Box,
    Mesh,
    Sdf,
    None,
}

impl From<i32> for GeomType {
    fn from(value: i32) -> Self {
        match value {
            0 => GeomType::Plane,
            1 => GeomType::HField,
            2 => GeomType::Sphere,
            3 => GeomType::Capsule,
            4 => GeomType::Ellipsoid,
            5 => GeomType::Cylinder,
            6 => GeomType::Box,
            7 => GeomType::Mesh,
            8 => GeomType::Sdf,
            1001 => GeomType::None,
            _ => GeomType::None,
        }
    }
}

impl From<mjtGeom_> for GeomType {
    fn from(value: mjtGeom_) -> Self {
        (value as i32).into()
    }
}

/// Geom data structure extracted from MuJoCo model
#[derive(Debug, Clone)]
pub struct Geom {
    pub id: i32,
    pub geom_type: GeomType,
    pub body_id: i32,
    pub size: [f64; 3],
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub rgba: [f32; 4],
    pub dataid: i32, // For meshes
    pub group: i32,  // Geom group for rendering selection
}

impl Geom {
    /// Extract geom data from MuJoCo model FFI
    pub fn from_model_ffi(model_ffi: &mjModel, geom_id: usize) -> Self {
        unsafe {
            let geom_type_slice =
                std::slice::from_raw_parts(model_ffi.geom_type, model_ffi.ngeom as usize);
            let geom_type = GeomType::from(geom_type_slice[geom_id] as i32);

            let body_id_slice =
                std::slice::from_raw_parts(model_ffi.geom_bodyid, model_ffi.ngeom as usize);
            let body_id = body_id_slice[geom_id];

            let size_slice = std::slice::from_raw_parts(
                model_ffi.geom_size as *const [f64; 3],
                model_ffi.ngeom as usize,
            );
            let size = size_slice[geom_id];

            let pos_slice = std::slice::from_raw_parts(
                model_ffi.geom_pos as *const [f64; 3],
                model_ffi.ngeom as usize,
            );
            let pos = pos_slice[geom_id];

            let quat_slice = std::slice::from_raw_parts(
                model_ffi.geom_quat as *const [f64; 4],
                model_ffi.ngeom as usize,
            );
            let quat = quat_slice[geom_id];

            let rgba_slice = std::slice::from_raw_parts(
                model_ffi.geom_rgba as *const [f32; 4],
                model_ffi.ngeom as usize,
            );
            let rgba = rgba_slice[geom_id];

            let dataid_slice =
                std::slice::from_raw_parts(model_ffi.geom_dataid, model_ffi.ngeom as usize);
            let dataid = dataid_slice[geom_id];

            let group_slice =
                std::slice::from_raw_parts(model_ffi.geom_group, model_ffi.ngeom as usize);
            let group = group_slice[geom_id];

            Geom {
                id: geom_id as i32,
                geom_type,
                body_id,
                size,
                pos,
                quat,
                rgba,
                dataid,
                group,
            }
        }
    }

    /// Extract all geoms from MuJoCo model
    pub fn extract_all_geoms(model_ffi: &mjModel) -> Vec<Self> {
        let ngeom = model_ffi.ngeom as usize;
        (0..ngeom)
            .map(|i| Self::from_model_ffi(model_ffi, i))
            .collect()
    }

    /// Generate a Bevy mesh from this geom
    /// For mesh types, pass the model_ffi to extract mesh data
    pub fn to_mesh(&self, model_ffi: Option<&mjModel>) -> Mesh {
        match self.geom_type {
            GeomType::Box => create_box_mesh(self.size),
            GeomType::Sphere => create_sphere_mesh(self.size[0]),
            GeomType::Cylinder => create_cylinder_mesh(self.size[0], self.size[1]),
            GeomType::Capsule => create_capsule_mesh(self.size[0], self.size[1]),
            GeomType::Ellipsoid => create_ellipsoid_mesh(self.size),
            GeomType::Plane => create_plane_mesh(self.size),
            GeomType::Mesh => {
                if let Some(model_ffi) = model_ffi {
                    if self.dataid >= 0 {
                        create_mesh_from_mujoco(model_ffi, self.dataid as usize)
                    } else {
                        // No mesh data available, use fallback
                        create_box_mesh([0.1, 0.1, 0.1])
                    }
                } else {
                    // No model available, use fallback
                    create_box_mesh([0.1, 0.1, 0.1])
                }
            }
            _ => {
                // Default to a small box for unsupported types
                create_box_mesh([0.1, 0.1, 0.1])
            }
        }
    }
}

/// Create a box mesh with MuJoCo-style sizing (half-extents)
fn create_box_mesh(size: [f64; 3]) -> Mesh {
    let size = Vec3::new(
        size[0] as f32 * 2.0,
        size[1] as f32 * 2.0,
        size[2] as f32 * 2.0,
    );
    Mesh::from(Cuboid::from_size(size))
}

/// Create a sphere mesh
fn create_sphere_mesh(radius: f64) -> Mesh {
    Mesh::from(Sphere::new(radius as f32).mesh().ico(3).unwrap())
}

/// Create a cylinder mesh
fn create_cylinder_mesh(radius: f64, half_height: f64) -> Mesh {
    Mesh::from(Cylinder {
        radius: radius as f32,
        half_height: half_height as f32,
    })
}

/// Create a capsule mesh (cylinder with hemispherical caps)
fn create_capsule_mesh(radius: f64, half_height: f64) -> Mesh {
    Mesh::from(Capsule3d {
        radius: radius as f32,
        half_length: half_height as f32,
    })
}

/// Create an ellipsoid mesh (scaled sphere)  
fn create_ellipsoid_mesh(size: [f64; 3]) -> Mesh {
    // For now, use a sphere and scale the transform instead
    // TODO: Implement proper vertex scaling
    create_sphere_mesh(size[0].max(size[1]).max(size[2]))
}

/// Create a plane mesh
fn create_plane_mesh(size: [f64; 3]) -> Mesh {
    let size_x = size[0] as f32 * 10.0; // MuJoCo planes are usually large
    let size_z = size[1] as f32 * 10.0;

    Mesh::from(Plane3d {
        normal: Dir3::Y,
        half_size: Vec2::new(size_x, size_z),
    })
}

/// Create a mesh from MuJoCo mesh data
fn create_mesh_from_mujoco(model_ffi: &mjModel, mesh_id: usize) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::Indices;
    use bevy::render::render_resource::PrimitiveTopology;

    unsafe {
        // Get mesh metadata
        let mesh_vertadr =
            std::slice::from_raw_parts(model_ffi.mesh_vertadr, model_ffi.nmesh as usize);
        let mesh_vertnum =
            std::slice::from_raw_parts(model_ffi.mesh_vertnum, model_ffi.nmesh as usize);
        let mesh_faceadr =
            std::slice::from_raw_parts(model_ffi.mesh_faceadr, model_ffi.nmesh as usize);
        let mesh_facenum =
            std::slice::from_raw_parts(model_ffi.mesh_facenum, model_ffi.nmesh as usize);
        let vert_start = mesh_vertadr[mesh_id] as usize;
        let vert_count = mesh_vertnum[mesh_id] as usize;
        let face_start = mesh_faceadr[mesh_id] as usize;
        let face_count = mesh_facenum[mesh_id] as usize;

        println!("🔍 Mesh {} extraction:", mesh_id);
        println!("   vert_start={}, vert_count={}", vert_start, vert_count);
        println!("   face_start={}, face_count={}", face_start, face_count);

        // Extract vertex positions
        let mesh_vert = std::slice::from_raw_parts(
            model_ffi.mesh_vert as *const [f32; 3],
            model_ffi.nmeshvert as usize,
        );
        let vertices: Vec<[f32; 3]> = mesh_vert[vert_start..vert_start + vert_count]
            .iter()
            .copied()
            .collect();

        println!("   Extracted {} vertices", vertices.len());

        // Extract normals (normals use the same indexing as vertices)
        let mesh_normal = std::slice::from_raw_parts(
            model_ffi.mesh_normal as *const [f32; 3],
            model_ffi.nmeshnormal as usize,
        );
        let normals: Vec<[f32; 3]> = mesh_normal[vert_start..vert_start + vert_count]
            .iter()
            .copied()
            .collect();

        println!("   Extracted {} normals", normals.len());

        // Extract face indices (triangles)
        // NOTE: mesh_face contains absolute vertex indices, so we need to make them relative
        let mesh_face = std::slice::from_raw_parts(
            model_ffi.mesh_face as *const i32,
            model_ffi.nmeshface as usize * 3,
        );

        let mut indices: Vec<u32> = Vec::with_capacity(face_count * 3);
        for i in 0..face_count {
            let face_idx = (face_start + i) * 3;
            let idx0 = mesh_face[face_idx] as i32;
            let idx1 = mesh_face[face_idx + 1] as i32;
            let idx2 = mesh_face[face_idx + 2] as i32;

            // Make indices relative to this mesh's vertex array
            indices.push((idx0 - vert_start as i32) as u32);
            indices.push((idx1 - vert_start as i32) as u32);
            indices.push((idx2 - vert_start as i32) as u32);
        }

        println!(
            "   Extracted {} indices ({} faces)",
            indices.len(),
            face_count
        );
        if !indices.is_empty() {
            println!(
                "   First face indices: [{}, {}, {}]",
                indices[0], indices[1], indices[2]
            );
        }

        // Create Bevy mesh
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );

        mesh.insert_indices(Indices::U32(indices));
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertices);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);

        mesh
    }
}
