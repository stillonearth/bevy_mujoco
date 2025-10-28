//! Query extensions for common physics operations

use bevy::prelude::*;
use crate::physics::*;

/// Extension trait for physics queries
pub trait PhysicsQueryExt {
    /// Get the linear velocity of a physics body
    fn linear_velocity(&self) -> Vec3;
    
    /// Get the angular velocity of a physics body
    fn angular_velocity(&self) -> Vec3;
    
    /// Check if body is moving
    fn is_moving(&self, threshold: f32) -> bool;
    
    /// Get kinetic energy
    fn kinetic_energy(&self) -> f32;
}

impl PhysicsQueryExt for (&Velocity, &Mass) {
    fn linear_velocity(&self) -> Vec3 {
        self.0.linear
    }
    
    fn angular_velocity(&self) -> Vec3 {
        self.0.angular
    }
    
    fn is_moving(&self, threshold: f32) -> bool {
        self.0.linear.length_squared() > threshold * threshold
    }
    
    fn kinetic_energy(&self) -> f32 {
        let (vel, mass) = (self.0, self.1);
        0.5 * mass.value * vel.linear.length_squared()
    }
}

/// Extension for joint queries
pub trait JointQueryExt {
    /// Check if joint is at limit
    fn at_limit(&self, threshold: f64) -> bool;
    
    /// Get normalized position in joint range [0, 1]
    fn normalized_position(&self) -> Vec<f64>;
}

impl JointQueryExt for &Joint {
    fn at_limit(&self, threshold: f64) -> bool {
        for (i, pos) in self.position.iter().enumerate() {
            if i < self.limits.len() {
                let [min, max] = self.limits[i];
                if (*pos - min).abs() < threshold || (*pos - max).abs() < threshold {
                    return true;
                }
            }
        }
        false
    }
    
    fn normalized_position(&self) -> Vec<f64> {
        self.position.iter().enumerate().map(|(i, pos)| {
            if i < self.limits.len() {
                let [min, max] = self.limits[i];
                if (max - min).abs() > 1e-6 {
                    return (pos - min) / (max - min);
                }
            }
            *pos
        }).collect()
    }
}

/// Query for actuator state
pub trait ActuatorQueryExt {
    /// Check if actuator is saturated
    fn is_saturated(&self) -> bool;
    
    /// Get control error (for position control)
    fn control_error(&self, target: f64) -> f64;
}

impl ActuatorQueryExt for &Actuator {
    fn is_saturated(&self) -> bool {
        // TODO: Check against actuator force limits
        self.force.abs() > 1000.0
    }
    
    fn control_error(&self, target: f64) -> f64 {
        target - self.control
    }
}

/// Helper for collision queries
pub trait CollisionQueryExt {
    /// Check if entity is in contact with anything
    fn has_contacts(&self) -> bool;
    
    /// Get total contact force
    fn total_contact_force(&self) -> f32;
    
    /// Check if in contact with specific entity
    fn in_contact_with(&self, entity: Entity) -> bool;
}

impl CollisionQueryExt for &Contacts {
    fn has_contacts(&self) -> bool {
        !self.contacts.is_empty()
    }
    
    fn total_contact_force(&self) -> f32 {
        self.contacts.iter().map(|c| c.force).sum()
    }
    
    fn in_contact_with(&self, entity: Entity) -> bool {
        self.contacts.iter().any(|c| c.other_entity == Some(entity))
    }
}
