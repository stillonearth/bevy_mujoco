//! ECS systems for MuJoCo integration

use bevy::prelude::*;
use crate::physics::*;
use crate::{MuJoCoSimulation, MuJoCoResources};

/// System to sync physics state from MuJoCo to Bevy components
pub fn sync_physics_state(
    simulation: Res<MuJoCoSimulation>,
    mut query: Query<(Entity, &PhysicsBody, &mut Transform, Option<&mut Velocity>)>,
) {
    let mujoco = simulation.data.lock().unwrap();
    let positions = mujoco.xpos();
    let rotations = mujoco.xquat();
    let velocities = mujoco.qvel();
    
    for (_entity, physics_body, mut transform, velocity) in query.iter_mut() {
        let body_id = physics_body.body_id as usize;
        
        // Update transform
        let pos = positions[body_id];
        let quat = rotations[body_id];
        
        transform.translation = Vec3::new(pos[0] as f32, pos[2] as f32, -pos[1] as f32);
        transform.rotation = Quat::from_xyzw(
            quat[1] as f32,
            quat[3] as f32,
            -quat[2] as f32,
            quat[0] as f32,
        );
        
        // Update velocity if component exists
        if let Some(mut vel) = velocity {
            // TODO: Extract actual velocities from qvel based on joint configuration
            vel.linear = Vec3::ZERO;
            vel.angular = Vec3::ZERO;
        }
    }
}

/// System to apply forces/torques to MuJoCo bodies
pub fn apply_forces(
    simulation: Res<MuJoCoSimulation>,
    mut query: Query<(&PhysicsBody, &mut ExternalForce)>,
) {
    let mut mujoco = simulation.data.lock().unwrap();
    
    for (physics_body, mut force) in query.iter_mut() {
        if force.force.length_squared() > 0.0 || force.torque.length_squared() > 0.0 {
            // Apply force to MuJoCo (requires access to xfrc_applied)
            let body_id = physics_body.body_id as usize;
            
            // TODO: Access xfrc_applied and apply forces
            // For now, log the intent
            debug!(
                "Applying force {:?} and torque {:?} to body {}",
                force.force, force.torque, body_id
            );
            
            // Reset forces after applying
            force.force = Vec3::ZERO;
            force.torque = Vec3::ZERO;
        }
    }
}

/// System to handle impulses (one-time forces)
pub fn apply_impulses(
    mut commands: Commands,
    simulation: Res<MuJoCoSimulation>,
    query: Query<(Entity, &PhysicsBody, &Impulse)>,
) {
    let mut mujoco = simulation.data.lock().unwrap();
    
    for (entity, physics_body, impulse) in query.iter() {
        let body_id = physics_body.body_id as usize;
        
        // Apply impulse to MuJoCo
        debug!(
            "Applying impulse {:?} to body {}",
            impulse.linear, body_id
        );
        
        // TODO: Apply impulse to qvel
        
        // Remove impulse component after applying
        commands.entity(entity).remove::<Impulse>();
    }
}

/// System to control actuators
pub fn control_actuators(
    simulation: Res<MuJoCoSimulation>,
    mut resources: ResMut<MuJoCoResources>,
    query: Query<&Actuator, Changed<Actuator>>,
) {
    for actuator in query.iter() {
        let actuator_id = actuator.actuator_id as usize;
        
        // Update control value in MuJoCo
        if actuator_id < resources.control.data.len() {
            resources.control.data[actuator_id] = actuator.control;
        }
    }
}

/// System to read sensor data
pub fn read_sensors(
    simulation: Res<MuJoCoSimulation>,
    resources: Res<MuJoCoResources>,
    mut query: Query<&mut Sensor>,
) {
    let sensor_data = &resources.state.sensor_data;
    
    for mut sensor in query.iter_mut() {
        let sensor_id = sensor.sensor_id as usize;
        
        // Update sensor value from MuJoCo
        if sensor_id < sensor_data.len() {
            // Sensors can have multiple dimensions
            sensor.value = vec![sensor_data[sensor_id]];
        }
    }
}

/// System to detect and emit collision events
pub fn detect_collisions(
    _simulation: Res<MuJoCoSimulation>,
    mut _collision_events: Commands,
    _query: Query<(Entity, &PhysicsBody, Option<&mut Contacts>)>,
) {
    // TODO: Access MuJoCo contact data
    // For now, this is a placeholder
    
    // Example of how to emit events:
    // collision_events.send(CollisionEvent {
    //     entity_a: entity1,
    //     entity_b: entity2,
    //     contact: Contact { ... },
    // });
}

/// System to update physics statistics
pub fn update_physics_stats(
    simulation: Res<MuJoCoSimulation>,
    mut stats: ResMut<PhysicsStats>,
) {
    let mujoco = simulation.data.lock().unwrap();
    stats.simulation_time = mujoco.time();
    // TODO: Add more detailed stats
}

/// Helper system to visualize forces (debug)
#[cfg(feature = "debug")]
pub fn debug_forces(
    mut gizmos: Gizmos,
    query: Query<(&Transform, &ExternalForce)>,
) {
    for (transform, force) in query.iter() {
        if force.force.length_squared() > 0.0 {
            gizmos.arrow(
                transform.translation,
                transform.translation + force.force * 0.1,
                Color::srgb(1.0, 0.0, 0.0),
            );
        }
        
        if force.torque.length_squared() > 0.0 {
            gizmos.circle(
                transform.translation,
                Direction3d::Y,
                force.torque.length() * 0.1,
                Color::srgb(0.0, 1.0, 0.0),
            );
        }
    }
}
