use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
};
use mujoco_rust::{Body, Geom, GeomType};
use nalgebra::Quaternion;
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

pub fn mesh_mujoco_2_bevy(mj_mesh: mujoco_rust::Mesh) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.set_indices(Some(Indices::U32(mj_mesh.indices)));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, mj_mesh.vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, mj_mesh.normals);
    mesh
}

pub fn quat_mujoco_2_bevy(quat: Quaternion<f64>) -> Quat {
    Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)
        * Quat::from_xyzw(
            quat[0] as f32,
            quat[2] as f32,
            quat[1] as f32,
            -quat[3] as f32,
        )
}

pub fn geom_material(geom: &Geom) -> StandardMaterial {
    let mut material = StandardMaterial::default();

    // Override color for ground plane
    if geom.body_id == 0 {
        material.base_color = Color::rgb(0.8, 0.4, 0.4);
    } else {
        material.base_color =
            Color::rgba(geom.color[0], geom.color[1], geom.color[2], geom.color[3]);
    }

    material
}

pub fn geom_mesh(geom: &Geom) -> Mesh {
    let size = &mut [geom.size.x, geom.size.y, geom.size.z];
    size.swap(1, 2);

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

        GeomType::MESH => mesh_mujoco_2_bevy(geom.mesh.clone().unwrap()),
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
    let size = &mut [geom.size.x, geom.size.y, geom.size.z];
    size.swap(1, 2);
    match geom.geom_type {
        GeomType::BOX => Vec3::new(
            0.0, // (size[0] * 2.0) as f32,
            (size[2] * 2.0) as f32,
            0.0, // (size[1] * 2.0) as f32,
        ),
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
        body.quat[2] as f32,
        body.quat[3] as f32,
        body.quat[0] as f32,
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
