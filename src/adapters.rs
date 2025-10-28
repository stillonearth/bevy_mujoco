use bevy::prelude::*;

use crate::{Body, Geom, GeomType};

/// Make bevy quaternion from MuJoCo quaternion [w, x, y, z]
pub(crate) fn quat_mujoco_2_bevy(quat: [f64; 4]) -> Quat {
    // MuJoCo quaternions are [w, x, y, z], Bevy uses [x, y, z, w]
    Quat::from_xyzw(quat[1] as f32, quat[2] as f32, quat[3] as f32, quat[0] as f32)
}

/// Make bevy vector from MuJoCo vector
pub(crate) fn vec3_mujoco_2_bevy(vec: [f64; 3]) -> Vec3 {
    Vec3::new(vec[0] as f32, vec[1] as f32, vec[2] as f32)
}

/// Make bevy material from MuJoCo geom description
pub(crate) fn geom_material(geom: &Geom) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgba(
            geom.rgba[0],
            geom.rgba[1],
            geom.rgba[2],
            geom.rgba[3],
        ),
        ..default()
    }
}

/// Return bevy transform for a given geometry from MuJoCo model
pub(crate) fn geom_transform(geom: &Geom) -> Transform {
    let mut transform = Transform {
        translation: vec3_mujoco_2_bevy(geom.pos),
        rotation: quat_mujoco_2_bevy(geom.quat),
        ..default()
    };

    // Apply coordinate system conversion for non-mesh geoms
    if geom.geom_type != GeomType::Mesh {
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

/// Bevy and MuJoCo treat object frames differently, this function provides correction
#[allow(dead_code)]
pub(crate) fn geom_correction(geom: &Geom) -> Vec3 {
    let size = geom.size;
    
    match geom.geom_type {
        GeomType::Box => Vec3::new(0.0, (size[1] / 2.0) as f32, 0.0),
        GeomType::Capsule => Vec3::new(0.0, (size[1] * 2.0) as f32, 0.0),
        GeomType::Cylinder => Vec3::new(0.0, (size[2] * 2.0) as f32, 0.0),
        _ => Vec3::ZERO,
    }
}
