use bevy::prelude::*;

use std::f32::consts::{PI, TAU};

pub struct IsoCameraPlugin;

impl Plugin for IsoCameraPlugin {
  fn build(&self, app: &mut App) {
    app.register_type::<IsoCameraState>();
    app.add_systems(Update, update_camera);
  }
}

#[derive(Default, Clone, Copy, Debug, Reflect)]
pub enum CardinalDirection {
  NW = 3,  N = 2,  NE = 1,
   W = 4, #[default]E = 0,
  SW = 5,  S = 6,  SE = 7,
}

impl CardinalDirection {
  pub fn from(n: i32) -> CardinalDirection {
    match n {
      0 => CardinalDirection::E,
      1 => CardinalDirection::NE,
      2 => CardinalDirection::N,
      3 => CardinalDirection::NW,
      4 => CardinalDirection::W,
      5 => CardinalDirection::SW,
      6 => CardinalDirection::S,
      7 => CardinalDirection::SE,
      _ => CardinalDirection::E,
    }
  }
}

#[derive(Default, Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)] 
pub struct IsoCameraState {
  pub target: Vec3,
  pub distance: f32,
  pub angle: f32,
  pub direction: CardinalDirection,
}

#[derive(Bundle)]
struct CameraBundle {
  camera: Camera3dBundle,
  state: IsoCameraState,
}

fn lerp_angle(from: f32, to: f32, t: f32) -> f32 {
  let mut angle = to - from;
  while angle > PI {
    angle -= TAU;
  }
  while angle < -PI {
    angle += TAU;
  }
  if (angle).abs() < 0.01 {
    return to;
  }
  from + angle * t
}

fn update_camera(
  mut camera: Query<(&mut IsoCameraState, &mut Transform)>,
  time: Res<Time>,
) {
  let mut camera = camera.single_mut();
  let state = &mut camera.0;
  let transform = &mut camera.1;

  let angle = state.direction as i32 as f32 / 8.0 * TAU;
  state.angle = lerp_angle(state.angle, angle, time.delta_seconds() * 10.0);

  let tf = Transform::from_xyz(15.0 * state.angle.sin(), 8.66025, 15.0 * state.angle.cos()).looking_at(Vec3::ZERO, Vec3::Y);
  transform.translation = tf.translation;
  transform.rotation = tf.rotation;
}