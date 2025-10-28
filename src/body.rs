use bevy::prelude::*;
use mujoco_rs::mujoco_c::*;

use crate::geom::{Geom, GeomType};

/// Body data structure extracted from MuJoCo model
#[derive(Debug, Clone)]
pub struct Body {
    pub id: i32,
    pub name: String,
    pub parent_id: i32,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
}

impl Body {
    /// Extract body data from MuJoCo model FFI
    pub fn from_model_ffi(model_ffi: &mjModel, body_id: usize) -> Self {
        unsafe {
            // Get parent ID
            let parent_id_slice =
                std::slice::from_raw_parts(model_ffi.body_parentid, model_ffi.nbody as usize);
            let parent_id = parent_id_slice[body_id];

            // Get body name
            let name_adr =
                std::slice::from_raw_parts(model_ffi.name_bodyadr, model_ffi.nbody as usize);
            let names_ptr = model_ffi.names;
            let name_start = name_adr[body_id] as isize;
            let name_cstr = std::ffi::CStr::from_ptr(names_ptr.offset(name_start));
            let name = name_cstr.to_string_lossy().to_string();

            // Get body position
            let pos_slice = std::slice::from_raw_parts(
                model_ffi.body_pos as *const [f64; 3],
                model_ffi.nbody as usize,
            );
            let pos = pos_slice[body_id];

            // Get body quaternion
            let quat_slice = std::slice::from_raw_parts(
                model_ffi.body_quat as *const [f64; 4],
                model_ffi.nbody as usize,
            );
            let quat = quat_slice[body_id];

            Body {
                id: body_id as i32,
                name,
                parent_id,
                pos,
                quat,
            }
        }
    }

    /// Extract all bodies from MuJoCo model
    pub fn extract_all_bodies(model_ffi: &mjModel) -> Vec<Self> {
        let nbody = model_ffi.nbody as usize;
        (0..nbody)
            .map(|i| Self::from_model_ffi(model_ffi, i))
            .collect()
    }

    /// Convert MuJoCo position to Bevy Vec3 (with coordinate system conversion)
    pub fn position_to_bevy(&self) -> Vec3 {
        Vec3::new(
            self.pos[0] as f32,
            self.pos[2] as f32, // Swap Y and Z
            -self.pos[1] as f32,
        )
    }

    /// Convert MuJoCo quaternion to Bevy Quat (with coordinate system conversion)
    /// MuJoCo quaternions are [w, x, y, z]
    pub fn rotation_to_bevy(&self) -> Quat {
        Quat::from_xyzw(
            self.quat[1] as f32,
            self.quat[3] as f32, // Swap Y and Z
            -self.quat[2] as f32,
            self.quat[0] as f32,
        )
    }

    /// Get all geoms associated with this body
    pub fn geoms(&self, all_geoms: &[Geom]) -> Vec<Geom> {
        all_geoms
            .iter()
            .filter(|g| g.body_id == self.id)
            .cloned()
            .collect()
    }

    /// Find the best geom for rendering this body
    /// This implements the same logic as the original mujoco-rust version:
    /// - If there's only one geom for this body, use it
    /// - Otherwise, prefer geoms in groups 0-2 (visual groups)
    /// - Prioritize MESH type geoms over others
    /// - Among remaining geoms, prefer lower group numbers
    pub fn render_geom(&self, all_geoms: &[Geom]) -> Option<Geom> {
        let body_geoms: Vec<&Geom> = all_geoms.iter().filter(|g| g.body_id == self.id).collect();

        // If there's only one geom, return it
        if body_geoms.len() == 1 {
            return Some(body_geoms[0].clone());
        }

        if body_geoms.is_empty() {
            return None;
        }

        // Filter to visual groups (0-2)
        let visual_geoms: Vec<&Geom> = body_geoms.iter().filter(|g| g.group < 3).copied().collect();

        if visual_geoms.is_empty() {
            // No visual geoms, just return the first one
            return Some(body_geoms[0].clone());
        }

        // Sort by priority: MESH types first, then by lower group number
        let best_geom = visual_geoms.iter().max_by(|a, b| {
            // Prefer MESH type
            let a_is_mesh = a.geom_type == GeomType::Mesh;
            let b_is_mesh = b.geom_type == GeomType::Mesh;

            if a_is_mesh && !b_is_mesh {
                return std::cmp::Ordering::Greater;
            }
            if !a_is_mesh && b_is_mesh {
                return std::cmp::Ordering::Less;
            }

            // If both or neither are meshes, prefer lower group number
            b.group.cmp(&a.group)
        });

        best_geom.map(|g| (*g).clone())
    }
}

/// Build a hierarchical tree of bodies
pub fn build_body_hierarchy(bodies: &[Body], geoms: &[Geom]) -> Vec<BodyNode> {
    // Find root bodies (those whose parent is world body 0, but not world itself)
    bodies
        .iter()
        .filter(|b| b.id != 0 && b.parent_id == 0)
        .map(|b| build_node(b, bodies, geoms))
        .collect()
}

fn build_node(body: &Body, all_bodies: &[Body], geoms: &[Geom]) -> BodyNode {
    // Find child bodies
    let children: Vec<BodyNode> = all_bodies
        .iter()
        .filter(|b| b.parent_id == body.id)
        .map(|b| build_node(b, all_bodies, geoms))
        .collect();

    // Find geoms for this body
    let body_geoms: Vec<Geom> = geoms
        .iter()
        .filter(|g| g.body_id == body.id)
        .cloned()
        .collect();

    BodyNode {
        body: body.clone(),
        geoms: body_geoms,
        children,
    }
}

/// A node in the body hierarchy tree
#[derive(Debug, Clone)]
pub struct BodyNode {
    pub body: Body,
    pub geoms: Vec<Geom>,
    pub children: Vec<BodyNode>,
}
