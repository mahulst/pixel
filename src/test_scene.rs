use bevy::prelude::*;
use crate::transform_animations::{
  AnimatedTransform,
  AnimatedTransformPart,
  TransformAnimation
};

pub struct TestScenePlugin;

impl Plugin for TestScenePlugin {
  fn build(&self, app: &mut App) {
    app
      .add_systems(Startup, setup)
      .add_systems(Update, animate_sun)
    ;
  }
}

#[derive(Component)]
struct SunLight;

pub fn setup(
  mut commands: Commands,
  // mut materials: ResMut<Assets<StandardMaterial>>,
  asset_server: Res<AssetServer>,
) {
  // Point lights
  commands.spawn((PointLightBundle {
    transform: Transform::from_xyz(1.0, 0.75, -2.0),
    point_light: PointLight {
      color: Color::rgba_u8(244, 177, 114, 255),
      intensity: 200.0,
      range: 8.0,
      shadows_enabled: true,
      ..default()
    },
    ..default()
  },
  AnimatedTransform {
    origin: Transform::from_xyz(1.0, 0.5, -2.0),
    animations: vec![
      AnimatedTransformPart {
        animation: TransformAnimation::Circle,
        speed: 0.2,
        scale: 1.0,
      },
      AnimatedTransformPart {
        animation: TransformAnimation::Bob,
        speed: 4.7,
        scale: 0.1,
      },
      AnimatedTransformPart {
        animation: TransformAnimation::Bob,
        speed: 1.0,
        scale: 0.33,
      },
    ],
  }));
  // Directional light
  commands.spawn((DirectionalLightBundle {
    directional_light: DirectionalLight {
      color: Color::rgb(0.8, 0.7, 0.6),
      illuminance: 40000.0,
      shadows_enabled: true,
      ..default()
    },
    transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 3.0)),
    ..default()
  }, SunLight));

  // commands.insert_resource(Animations(vec![
  //   asset_server.load("models/dungeon_test.glb#Idle"),
  // ]));

  // Load in models/dungeon_test.glb
  commands.spawn(SceneBundle {
    scene: asset_server.load("models/dungeon_test.glb#Scene0"),
    ..default()
  });
}

fn animate_sun(time: Res<Time>, mut sun: Query<&mut Transform, With<SunLight>>) {
  let mut transform = sun.single_mut();
  *transform = Transform::from_rotation(Quat::from_rotation_x(
    -std::f32::consts::PI / 3.0 + time.elapsed_seconds() as f32 / 10.0,
  ));
}