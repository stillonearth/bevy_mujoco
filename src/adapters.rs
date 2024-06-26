use bevy::{
    prelude::*,
    render::{mesh::Indices, render_asset::RenderAssetUsages, render_resource::PrimitiveTopology},
};
use mujoco_rust::{Body, Geom, GeomType};
use nalgebra::{ArrayStorage, Const, Matrix, Quaternion};
use trees::Tree;

// use crate::mujoco_shape;

/// BodyTree restructures bodie list into a tree structure
/// All translations and quaternions are relative to the parent body
#[derive(Deref, DerefMut)]
pub struct BodyTree(pub Tree<Body>);

/// Returns tree of bodies
pub(crate) fn body_tree(bodies: &[Body]) -> Vec<BodyTree> {
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

    let mut trees: Vec<BodyTree> = vec![];
    let root_bodies = bodies.iter().filter(|body| body.parent_id == 0);
    for body in root_bodies {
        let mut root_leaf: BodyTree = BodyTree(Tree::new(body.clone()));
        collect_children(&mut root_leaf.0, bodies);
        trees.push(root_leaf);
    }

    trees
}

/// Make a bevy mesh from exported MuJoCo mesh
pub(crate) fn mesh_mujoco_2_bevy(mj_mesh: mujoco_rust::Mesh) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_indices(Indices::U32(mj_mesh.indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, mj_mesh.vertices);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, mj_mesh.normals);
    mesh
}

/// Make bevy quaternion from MuJoCo quaternion
pub(crate) fn quat_mujoco_2_bevy(quat: Quaternion<f64>) -> Quat {
    Quat::from_xyzw(quat.i as f32, quat.j as f32, quat.k as f32, quat.w as f32)
}

/// Make bevy vector from MuJoCo vector
pub(crate) fn vec3_mujoco_2_bevy(
    vec: Matrix<f64, Const<3>, Const<1>, ArrayStorage<f64, 3, 1>>,
) -> Vec3 {
    Vec3::new(vec.x as f32, vec.y as f32, vec.z as f32)
}

/// Make bevy material from MuJoCo description
pub(crate) fn geom_material(geom: &Geom) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgba(geom.color[0], geom.color[1], geom.color[2], geom.color[3]),
        ..default()
    }
}

/// Return mesh for a given geometry (PLANE; BOX; SPHERE; CAPSULE; ELLIPSOID; CYLINDER; MESH)
pub(crate) fn geom_mesh(geom: &Geom) -> Mesh {
    let size = &mut [geom.size.x as f32, geom.size.z as f32, geom.size.y as f32];

    match geom.geom_type {
        GeomType::PLANE => {
            let _plane_size = if size[0] > 0.0 {
                size[0]
            } else {
                // MuJoCo size 0 means infinite
                1e6
            };
            Mesh::from(Plane3d::default())
        }
        GeomType::BOX => Mesh::from(Cuboid::new(size[0], size[1], size[2])),
        GeomType::SPHERE => Mesh::from(Sphere {
            radius: size[0],
            ..default()
        }),
        GeomType::CAPSULE => Mesh::from(Capsule3d {
            radius: size[0],
            half_length: size[2],
            ..default()
        }),
        GeomType::ELLIPSOID => todo!(),
        GeomType::CYLINDER => Mesh::from(Cylinder {
            radius: size[0],
            half_height: size[2],
            ..default()
        }),

        GeomType::MESH => mesh_mujoco_2_bevy(geom.mesh.clone().unwrap()),
        // --- NOT IMPLEMENTED ---
        _ => todo!(),
    }
}

/// bevy and mujoco treat object frame differently, this function converts
pub(crate) fn geom_correction(geom: &Geom) -> Vec3 {
    let size = &mut [geom.size.x, geom.size.z, geom.size.y];

    match geom.geom_type {
        GeomType::BOX => Vec3::new(0.0, (size[1] / 2.0) as f32, 0.0),
        GeomType::CAPSULE => Vec3::new(0.0, (size[1] * 2.0) as f32, 0.0),
        GeomType::CYLINDER => Vec3::new(0.0, (size[2] * 2.0) as f32, 0.0),

        _ => Vec3::ZERO,
    }
}

/// Return bevy transform for a given geometry from MuJoCo model
pub(crate) fn geom_transform(geom: &Geom) -> Transform {
    let mut transform = Transform {
        translation: vec3_mujoco_2_bevy(geom.pos),

        rotation: quat_mujoco_2_bevy(geom.quat),
        ..default()
    };

    if geom.geom_type != GeomType::MESH {
        transform.rotation *= Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    }

    transform
}

/// Return bevy transform for a given body from MuJoCo model
pub(crate) fn body_transform(body: &Body) -> Transform {
    Transform {
        translation: vec3_mujoco_2_bevy(body.pos),
        rotation: quat_mujoco_2_bevy(body.quat),
        ..default()
    }
}
