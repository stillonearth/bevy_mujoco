//! Builder API for spawning physics entities declaratively

use bevy::prelude::*;
use crate::physics::*;

/// Builder for creating physics-enabled entities
pub struct PhysicsEntityBuilder {
    commands: EntityCommands<'static>,
}

impl PhysicsEntityBuilder {
    pub fn new(commands: EntityCommands<'static>) -> Self {
        Self { commands }
    }
    
    /// Add a physics body
    pub fn with_physics_body(mut self, body_id: i32) -> Self {
        self.commands.insert(PhysicsBody { body_id });
        self
    }
    
    /// Add velocity tracking
    pub fn with_velocity(mut self) -> Self {
        self.commands.insert(Velocity::default());
        self
    }
    
    /// Add mass properties
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.commands.insert(Mass {
            value: mass,
            ..default()
        });
        self
    }
    
    /// Add external force capability
    pub fn with_forces(mut self) -> Self {
        self.commands.insert(ExternalForce::default());
        self
    }
    
    /// Add a joint
    pub fn with_joint(mut self, joint_id: i32, joint_type: JointType) -> Self {
        self.commands.insert(Joint {
            joint_id,
            joint_type,
            position: vec![],
            velocity: vec![],
            limits: vec![],
        });
        self
    }
    
    /// Add an actuator
    pub fn with_actuator(mut self, actuator_id: i32) -> Self {
        self.commands.insert(Actuator {
            actuator_id,
            control: 0.0,
            force: 0.0,
        });
        self
    }
    
    /// Add a sensor
    pub fn with_sensor(mut self, sensor_id: i32, sensor_type: SensorType) -> Self {
        self.commands.insert(Sensor {
            sensor_id,
            sensor_type,
            value: vec![],
        });
        self
    }
    
    /// Add collision detection
    pub fn with_collider(mut self, geom_id: i32) -> Self {
        self.commands.insert(Collider {
            geom_id,
            collision_group: 1,
            collision_mask: !0,
        });
        self.commands.insert(Contacts::default());
        self
    }
    
    /// Add physics material
    pub fn with_material(mut self, friction: f32, restitution: f32) -> Self {
        self.commands.insert(PhysicsMaterial {
            friction,
            restitution,
        });
        self
    }
    
    /// Finish building and return entity ID
    pub fn build(self) -> Entity {
        self.commands.id()
    }
}

/// Extension trait for Commands to spawn physics entities easily
pub trait PhysicsEntitySpawner {
    fn spawn_physics_entity(&mut self) -> PhysicsEntityBuilder;
    fn spawn_physics_body(&mut self, body_id: i32) -> Entity;
}

impl PhysicsEntitySpawner for Commands<'_, '_> {
    fn spawn_physics_entity(&mut self) -> PhysicsEntityBuilder {
        // Note: This is a simplified version. In practice, you'd need to handle
        // the lifetime more carefully.
        todo!("Implement proper builder with correct lifetime handling")
    }
    
    fn spawn_physics_body(&mut self, body_id: i32) -> Entity {
        self.spawn((
            PhysicsBody { body_id },
            Transform::default(),
            Visibility::default(),
        )).id()
    }
}

/// Helper macro for spawning physics entities
#[macro_export]
macro_rules! spawn_physics {
    ($commands:expr, $body_id:expr, { $($component:expr),* $(,)? }) => {
        $commands.spawn((
            PhysicsBody { body_id: $body_id },
            Transform::default(),
            Visibility::default(),
            $($component),*
        ))
    };
}
