//! Physics components and systems for MuJoCo ECS integration
//! 
//! This module provides a natural Bevy-style interface for MuJoCo physics.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Marker component for physics-enabled entities
#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct PhysicsBody {
    /// MuJoCo body ID
    pub body_id: i32,
}

/// Velocity component (automatically synced from MuJoCo)
#[derive(Component, Default, Reflect, Debug, Clone)]
#[reflect(Component)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Mass properties (read-only, from MuJoCo model)
#[derive(Component, Default, Reflect, Debug, Clone)]
#[reflect(Component)]
pub struct Mass {
    pub value: f32,
    pub center_of_mass: Vec3,
    pub inertia: Mat3,
}

/// Apply forces to bodies
#[derive(Component, Default, Reflect, Debug)]
#[reflect(Component)]
pub struct ExternalForce {
    pub force: Vec3,
    pub torque: Vec3,
    /// If true, forces are applied in world space; otherwise in local space
    pub world_space: bool,
}

/// Apply impulses (one-time forces)
#[derive(Component, Default, Debug)]
pub struct Impulse {
    pub linear: Vec3,
    pub angular: Vec3,
    pub world_space: bool,
}

/// Joint component for articulated bodies
#[derive(Component, Reflect, Debug, Clone)]
#[reflect(Component)]
pub struct Joint {
    pub joint_id: i32,
    pub joint_type: JointType,
    /// Current joint position(s)
    pub position: Vec<f64>,
    /// Current joint velocity(s)  
    pub velocity: Vec<f64>,
    /// Joint limits [min, max] for each DOF
    pub limits: Vec<[f64; 2]>,
}

#[derive(Reflect, Debug, Clone, Copy, PartialEq)]
pub enum JointType {
    Free,      // 6 DOF
    Ball,      // 3 DOF rotation
    Slide,     // 1 DOF translation
    Hinge,     // 1 DOF rotation
}

/// Actuator control component
#[derive(Component, Default, Reflect, Debug)]
#[reflect(Component)]
pub struct Actuator {
    pub actuator_id: i32,
    /// Desired control value
    pub control: f64,
    /// Current actuator force/torque output
    pub force: f64,
}

/// Sensor data from MuJoCo
#[derive(Component, Default, Reflect, Debug, Clone)]
#[reflect(Component)]
pub struct Sensor {
    pub sensor_id: i32,
    pub sensor_type: SensorType,
    pub value: Vec<f64>,
}

#[derive(Reflect, Default, Debug, Clone, Copy, PartialEq)]
pub enum SensorType {
    #[default]
    Touch,
    Accelerometer,
    Velocimeter,
    Gyro,
    Force,
    Torque,
    Magnetometer,
    Rangefinder,
    JointPos,
    JointVel,
    ActuatorPos,
    ActuatorVel,
    ActuatorForce,
    FramePos,
    FrameQuat,
    FrameLinVel,
    FrameAngVel,
}

/// Collision detection component
#[derive(Component, Default, Reflect, Debug)]
#[reflect(Component)]
pub struct Collider {
    pub geom_id: i32,
    /// Collision filter group
    pub collision_group: u32,
    /// Which groups this collider can collide with
    pub collision_mask: u32,
}

/// Contact information (automatically populated)
#[derive(Component, Default, Debug, Clone)]
pub struct Contacts {
    pub contacts: Vec<Contact>,
}

#[derive(Debug, Clone)]
pub struct Contact {
    pub other_entity: Option<Entity>,
    pub position: Vec3,
    pub normal: Vec3,
    pub penetration: f32,
    pub force: f32,
}

/// Events for physics interactions
#[derive(Debug, Clone)]
pub struct CollisionEvent {
    pub entity_a: Entity,
    pub entity_b: Entity,
    pub contact: Contact,
}

#[derive(Debug, Clone)]
pub struct ContactStarted {
    pub entity_a: Entity,
    pub entity_b: Entity,
}

#[derive(Debug, Clone)]
pub struct ContactEnded {
    pub entity_a: Entity,
    pub entity_b: Entity,
}

/// Physics material properties
#[derive(Component, Reflect, Debug, Clone)]
#[reflect(Component)]
pub struct PhysicsMaterial {
    pub friction: f32,
    pub restitution: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.0,
        }
    }
}

/// Control mode for joints/actuators
#[derive(Component, Reflect, Debug, Clone)]
#[reflect(Component)]
pub struct ControlMode {
    pub mode: ControlType,
    pub target: f64,
    pub gain: f64,
}

#[derive(Reflect, Debug, Clone, Copy, PartialEq)]
pub enum ControlType {
    Position,
    Velocity,
    Force,
}

/// Physics configuration
#[derive(Resource, Reflect, Debug, Clone)]
#[reflect(Resource)]
pub struct PhysicsConfig {
    /// Simulation timestep
    pub timestep: f64,
    /// Number of substeps per frame
    pub substeps: u32,
    /// Gravity vector
    pub gravity: Vec3,
    /// Pause simulation
    pub paused: bool,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            timestep: 0.002,  // 2ms = 500 Hz
            substeps: 1,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            paused: false,
        }
    }
}

/// Statistics from last physics step
#[derive(Resource, Default, Reflect, Debug, Clone)]
#[reflect(Resource)]
pub struct PhysicsStats {
    pub contacts: usize,
    pub step_time_us: u128,
    pub simulation_time: f64,
}

/// Query helpers for common patterns
pub trait PhysicsQuery {
    fn apply_force_at_point(&mut self, force: Vec3, point: Vec3);
    fn apply_central_force(&mut self, force: Vec3);
    fn apply_torque(&mut self, torque: Vec3);
}

impl PhysicsQuery for (Mut<'_, ExternalForce>, &Transform) {
    fn apply_force_at_point(&mut self, force: Vec3, point: Vec3) {
        let (ext_force, transform) = (self.0.as_mut(), self.1);
        ext_force.force += force;
        let r = point - transform.translation;
        ext_force.torque += r.cross(force);
    }

    fn apply_central_force(&mut self, force: Vec3) {
        self.0.force += force;
    }

    fn apply_torque(&mut self, torque: Vec3) {
        self.0.torque += torque;
    }
}
