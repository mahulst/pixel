use bevy::{
  prelude::*,
  render::camera::ScalingMode,
  core_pipeline::{
    tonemapping::DebandDither,
    prepass::{
      DepthPrepass,
      NormalPrepass,
      MotionVectorPrepass,
      DeferredPrepass,
    },
  },
};
use leafwing_input_manager::prelude::*;
use crate::{isocam::{
  IsoCameraState,
  CardinalDirection,
}, postpixel::PostPixelSettings};

use std::f32::consts::PI;

pub struct CameraSetupPlugin;

impl Plugin for CameraSetupPlugin {
  fn build(&self, app: &mut App) {
    app
      .register_type::<IsoCameraState>()
      .add_systems(Startup, setup)
      .add_systems(Update, (
        move_camera,
      ))
      .add_plugins(InputManagerPlugin::<Action>::default())
    ;
  }
}

#[derive(Actionlike, PartialEq, Eq, Hash, Clone, Copy, Debug, Reflect)]
pub enum Action {
  CameraLeft,
  CameraRight,
}

pub fn move_camera(
  query: Query<&ActionState<Action>, With<Camera>>,
  mut camera: Query<&mut IsoCameraState>,
) {
  let mut camera = camera.single_mut();

  let action_state = query.single();

  let mut new_dir = camera.direction as i32;
  
  if action_state.just_pressed(Action::CameraLeft) {
    new_dir += 1;
  }
  if action_state.just_pressed(Action::CameraRight) {
    new_dir -= 1;
  }

  if new_dir < 0 {
    new_dir = 7;
  } else if new_dir > 7 {
    new_dir = 0;
  }
  
  camera.direction = CardinalDirection::from(new_dir);
}

/// set up a simple 3D scene
pub fn setup(
  mut commands: Commands,
) {
  commands.spawn((
    Camera3dBundle {
      projection: OrthographicProjection {
        scale: 3.0,
        scaling_mode: ScalingMode::FixedVertical(2.0),
        near: 0.1,
        far: 32.0,
        ..default()
      }
      .into(),
      dither: DebandDither::Disabled,
      transform: Transform::from_xyz(15.0 * (PI/4.0_f32).sin(), 8.66025, 15.0 * (PI/4.0_f32).cos()).looking_at(Vec3::ZERO, Vec3::Y),
      ..default()
    },
    InputManagerBundle {
      action_state: ActionState::<Action>::default(),
      input_map: InputMap::new([(KeyCode::A, Action::CameraLeft), (KeyCode::D, Action::CameraRight)]),
    },
    // ScreenSpaceAmbientOcclusionBundle::default(),
    // TemporalAntiAliasBundle::default(),
    IsoCameraState {
      target: Vec3::ZERO,
      distance: 8.66,
      angle: 45.0 / 360.0 * std::f32::consts::TAU,
      direction: CardinalDirection::NW,
    },
    PostPixelSettings {
      pixel_scale: 4.0,
      ..default()
    },
    DepthPrepass,
    NormalPrepass,
    MotionVectorPrepass,
    DeferredPrepass,
  ));
}