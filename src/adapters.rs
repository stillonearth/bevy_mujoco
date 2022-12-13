use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
};
use mujoco_rust::{Body, Geom, GeomType};
use trees::Tree;

use crate::mujoco_shape;

#[derive(Deref, DerefMut)]
pub struct BodyTree(pub Tree<Body>);

fn collect_children(parent_leaf: &mut Tree<Body>, bodies: &[Body]) {
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

/// Returns tree of bodies
pub fn body_tree(bodies: &[Body]) -> Vec<BodyTree> {
    let mut trees: Vec<BodyTree> = vec![];

    let root_bodies = bodies.iter().filter(|body| body.parent_id == 0);

    for body in root_bodies {
        let mut root_leaf: BodyTree = BodyTree(Tree::new(body.clone()));
        collect_children(&mut root_leaf.0, bodies);

        trees.push(root_leaf);
    }

    trees
}

pub fn mujoco_mesh_2_bevy(mj_mesh: mujoco_rust::Mesh) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.set_indices(Some(Indices::U32(mj_mesh.indices)));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, mj_mesh.vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, mj_mesh.normals);
    mesh
}

/// Adjust format of 3-Vec from Mujoco to bevy
fn replace_values_vec3(arr: &[f64; 3], i1: usize, i2: usize) -> [f64; 3] {
    let mut out = *arr;
    let c_1 = arr[i1];
    let c_2 = arr[i2];
    out[i1] = c_2;
    out[i2] = c_1;
    out
}

pub fn geom_mesh(geom: &Geom) -> Mesh {
    let size = replace_values_vec3(&geom.size, 1, 2);

    match geom.geom_type {
        GeomType::PLANE => {
            let plane_size = if size[0] > 0.0 {
                size[0]
            } else {
                // MuJoCo size 0 means infinite
                1e6
            };
            Mesh::from(shape::Plane {
                size: plane_size as f32,
            })
        }
        GeomType::BOX => Mesh::from(shape::Box::new(
            size[0] as f32,
            size[1] as f32,
            size[2] as f32,
        )),
        GeomType::SPHERE => Mesh::from(shape::Icosphere {
            radius: size[0] as f32,
            ..default()
        }),
        GeomType::CAPSULE => Mesh::from(shape::Capsule {
            radius: size[0] as f32,
            depth: (size[2] * 2.0) as f32,
            ..default()
        }),
        GeomType::ELLIPSOID => todo!(),
        GeomType::CYLINDER => Mesh::from(mujoco_shape::Cylinder {
            radius: size[0] as f32,
            height: (size[2] * 2.0) as f32,
            ..default()
        }),

        GeomType::MESH => mujoco_mesh_2_bevy(geom.mesh.clone().unwrap()),
        // --- NOT IMPLEMENTED ---
        _ => todo!(),
    }
}

pub fn geom_rotation(geom: &Geom) -> Quat {
    match geom.geom_type {
        GeomType::MESH => Quat::from_xyzw(
            geom.quat[1] as f32,
            geom.quat[2] as f32,
            geom.quat[3] as f32,
            geom.quat[0] as f32,
        ),
        _ => Quat::from_xyzw(
            geom.quat[1] as f32,
            geom.quat[3] as f32,
            geom.quat[2] as f32,
            -geom.quat[0] as f32,
        ),
    }
}

/// bevy and mujoco treat object frame differently, this function converts
pub fn geom_correction(geom: &Geom) -> Vec3 {
    let size = replace_values_vec3(&geom.size, 1, 2);
    match geom.geom_type {
        GeomType::BOX => Vec3::new(0.0, (size[1] * 2.0) as f32, 0.0),
        GeomType::CAPSULE => Vec3::new(0.0, (size[1] * 2.0) as f32, 0.0),
        GeomType::CYLINDER => Vec3::new(0.0, (size[2] * 2.0) as f32, 0.0),
        _ => Vec3::ZERO,
    }
}

pub fn geom_translation(geom: &Geom) -> Vec3 {
    Vec3::new(geom.pos[0] as f32, geom.pos[1] as f32, geom.pos[2] as f32) - geom_correction(geom)
}

pub fn geom_transform(geom: &Geom) -> Transform {
    Transform {
        translation: geom_translation(geom),
        rotation: geom_rotation(geom),
        ..default()
    }
}

pub fn body_rotation(body: &Body) -> Quat {
    Quat::from_xyzw(
        body.quat[1] as f32,
        body.quat[3] as f32,
        body.quat[2] as f32,
        -body.quat[0] as f32,
    )
}

pub fn body_translation(body: &Body) -> Vec3 {
    Vec3::new(body.pos[0] as f32, body.pos[1] as f32, body.pos[2] as f32)
}

pub fn body_transform(body: &Body) -> Transform {
    Transform {
        translation: body_translation(body),
        rotation: body_rotation(body),
        ..default()
    }
}
