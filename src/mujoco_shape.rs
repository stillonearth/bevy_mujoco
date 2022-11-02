// Originally from https://github.com/bevyengine/bevy/pull/305/files

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, Mesh},
        render_resource::PrimitiveTopology,
    },
};

/// A cylinder which stands on the XZ plane
pub struct Cylinder {
    /// Radius of the cylinder (X&Z axis)
    pub radius: f32,
    /// Height of the cylinder (Y axis)
    pub height: f32,
    /// Number of vertices around each horizontal slice of the cylinder
    pub resolution: u32,
    // Number of vertical subdivisionss
    pub subdivisions: u32,
}

impl Default for Cylinder {
    fn default() -> Self {
        Self {
            radius: 0.5,
            height: 1.0,
            resolution: 20,
            subdivisions: 4,
        }
    }
}

impl From<Cylinder> for Mesh {
    fn from(c: Cylinder) -> Self {
        assert!(c.radius > 0.0 && c.height > 0.0 && c.resolution > 0 && c.subdivisions > 0);

        let count = (c.resolution * (c.subdivisions + 3) + 2) as usize;
        let mut positions = Vec::with_capacity(count);
        let step = std::f32::consts::PI * 2.0 / c.resolution as f32;
        let mut add_ring = |height, with_center| {
            if with_center {
                positions.push([0.0, height, 0.0]);
            }
            for j in 0..c.resolution {
                let theta = step * j as f32;
                positions.push([theta.cos() * c.radius, height, theta.sin() * c.radius]);
            }
        };

        // Shaft vertices
        let h_step = c.height / c.subdivisions as f32;
        for i in 0..(c.subdivisions + 1) {
            add_ring(c.height * 0.5 - h_step * i as f32, false);
        }

        // Top vertices
        let top_offset = c.resolution * (c.subdivisions + 1);
        add_ring(c.height * 0.5, true);

        // Bottom vertices
        let bottom_offset = top_offset + c.resolution + 1;
        add_ring(-c.height * 0.5, true);
        assert_eq!(positions.len(), count);

        // Index buffer
        let index_count = ((6 * c.subdivisions * c.resolution) + 6 * c.resolution) as usize;
        let mut indices = Vec::with_capacity(index_count);
        // Shaft quads
        for i in 0..c.subdivisions {
            let base1 = c.resolution * i;
            let base2 = base1 + c.resolution;
            for j in 0..c.resolution {
                let j1 = (j + 1) % c.resolution;
                indices.extend([base2 + j, base1 + j1, base1 + j].iter().copied());
                indices.extend([base2 + j, base2 + j1, base1 + j1].iter().copied());
            }
        }

        // Top circle triangles
        for j in 0..c.resolution {
            let j1 = (j + 1) % c.resolution;
            let base = top_offset + 1;
            indices.extend([top_offset, base + j, base + j1].iter().copied());
        }
        // Bottom circle triangles
        for j in 0..c.resolution {
            let j1 = (j + 1) % c.resolution;
            let base = bottom_offset + 1;
            indices.extend([bottom_offset, base + j1, base + j].iter().copied());
        }
        assert_eq!(indices.len(), index_count);

        // Shaft normals are their positions X&Z
        let mut normals = positions
            .iter()
            .map(|&p| {
                (Vec3::from(p) * Vec3::new(1.0, 0.0, 1.0))
                    .normalize()
                    .into()
            })
            .collect::<Vec<[f32; 3]>>();

        // Give the top and bottom of the cylinder a clear up/down normal
        for i in top_offset..bottom_offset {
            normals[i as usize] = [0.0, 1.0, 0.0];
        }
        for i in bottom_offset..count as u32 {
            normals[i as usize] = [0.0, -1.0, 0.0];
        }

        let uvs: Vec<[f32; 2]> = positions
            .iter()
            .map(|&p| [p[0] / c.radius, (p[1] + c.height) / (c.height * 2.0)])
            .collect();

        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_indices(Some(Indices::U32(indices)));
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

        mesh
    }
}
