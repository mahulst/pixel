//! Shows how to create a 3D orthographic view (for isometric-look games or CAD applications).

pub mod postprocessing;

use std::f32::consts::PI;

use bevy::{
    prelude::*,
    render::camera::ScalingMode,
    input::keyboard::KeyboardInput,
    core_pipeline::{
        fxaa::Fxaa,
        prepass::{DepthPrepass, MotionVectorPrepass, DeferredPrepass, NormalPrepass},
    },
    pbr::{
        DefaultOpaqueRendererMethod,
        NotShadowCaster,
        OpaqueRendererMethod,
    },
};

use crate::postprocessing::postprocessing::{PostProcessPlugin, PostProcessSettings};

fn main() {
    App::new()
        // .insert_resource(Msaa::Off)
        // .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins(DefaultPlugins)
        .add_plugins(PostProcessPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, process_input)
        .add_systems(Update, process_input)
        .insert_resource(InputState::default())
        .run();
}

#[derive(Resource)]
struct QuitCounter {
    count: u32,
}

impl Default for QuitCounter {
    fn default() -> Self {
        QuitCounter {
            count: 10,
        }
    }
}

#[derive(Default, Clone, Copy, Debug, Reflect)]
enum CameraDirection {
    NW = 3,  N = 2,  NE = 1,
     W = 4, #[default]E = 0,
    SW = 5,  S = 6,  SE = 7,
}

#[derive(Default, Component, Clone, Copy, Debug, Reflect)]
struct CameraState {
    target: Vec3,
    distance: f32,
    angle: f32,
    direction: CameraDirection,
}

#[derive(Bundle)]
struct CameraBundle {
    camera: Camera3dBundle,
    state: CameraState,
}

#[derive(Reflect, Clone, Copy, Debug)]
struct InputElement<T> {
    value: T,
    previous: T,
    duration: f32,
    consumed: bool,
}

impl Default for InputElement<bool> {
    fn default() -> Self {
        InputElement {
            value: false,
            previous: false,
            duration: 0.0,
            consumed: false,
        }
    }
}
impl Default for InputElement<Vec2> {
    fn default() -> Self {
        InputElement {
            value: Vec2::ZERO,
            previous: Vec2::ZERO,
            duration: 0.0,
            consumed: false,
        }
    }
}

impl InputElement<bool> {
    fn set(&mut self, value: bool) {
        self.previous = self.value;
        self.value = value;
        if self.value != self.previous {
            self.duration = 0.0;
        }
        self.consumed = false;
    }
}
impl InputElement<Vec2> {
    fn set(&mut self, value: Vec2) {
        self.previous = self.value;
        self.value = value;
        if self.value.x != self.previous.x || self.value.y != self.previous.y {
            // Maybe deadzone check instead here.
            self.duration = 0.0;
        }
        self.consumed = false;
    }
}

#[derive(Resource, Reflect, Clone, Copy, Debug)]
struct InputState {
    movement: InputElement<Vec2>,
    action: InputElement<bool>,
    fire: InputElement<bool>,
    melee: InputElement<bool>,
    dodge: InputElement<bool>,
}

impl Default for InputState {
    fn default() -> Self {
        InputState {
            movement: InputElement::default(),
            action: InputElement::default(),
            fire: InputElement::default(),
            melee: InputElement::default(),
            dodge: InputElement::default(),
        }
    }
}

fn process_input(
    mut kb_events: EventReader<KeyboardInput>,
    mut inputs: ResMut<InputState>,
    time: Res<Time>,
) {
    // Update each input's timer.
    inputs.dodge.duration += time.delta_seconds();
    inputs.fire.duration += time.delta_seconds();
    inputs.melee.duration += time.delta_seconds();
    inputs.action.duration += time.delta_seconds();
    inputs.movement.duration += time.delta_seconds();

    for event in kb_events.read() {
        let mut movement: Vec2 = Vec2::ZERO;
        match event.key_code {
            Some(key) => {
                match key {
                    KeyCode::W | KeyCode::Up => {
                        movement.y += 1.0;
                    },
                    KeyCode::A | KeyCode::Left => {
                        movement.x -= 1.0;
                    },
                    KeyCode::S | KeyCode::Down => {
                        movement.y -= 1.0;
                    },
                    KeyCode::D | KeyCode::Right => {
                        movement.x += 1.0;
                    },
                    _ => {}
                }
            },
            None => {}
        }
        inputs.movement.value = movement;
        inputs.movement.duration = 0.0;

    }
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut inputs: ResMut<InputState>,
) {
    // camera
    // commands.spawn(Camera3dBundle {
    //     projection: OrthographicProjection {
    //         scale: 3.0,
    //         scaling_mode: ScalingMode::FixedVertical(2.0),
    //         ..default()
    //     }
    //     .into(),
    //     transform: Transform::from_xyz(5.0, 5.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    //     ..default()
    // });

    let resolution = Vec2::new(640.0, 480.0);

    commands.spawn((
        Camera3dBundle {
            projection: OrthographicProjection {
                scale: 3.0,
                scaling_mode: ScalingMode::FixedVertical(2.0),
                ..default()
            }
            .into(),
            transform: Transform::from_xyz(5.0 * (PI/4.0_f32).sin(), 2.88675, 5.0 * (PI/4.0_f32).cos()).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        CameraState {
            target: Vec3::ZERO,
            distance: 8.66,
            angle: 45.0 / 360.0 * std::f32::consts::TAU,
            direction: CameraDirection::NW,
        },
        PostProcessSettings {
            intensity: 0.02,
            resolution: resolution,
            pixel_scale: 4.0,
            ..default()
        },
        DepthPrepass,
        MotionVectorPrepass,
    ));

    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(5.0).into()),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });
    // cubes
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(1.5, 0.5, 1.5),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(1.5, 0.5, -1.5),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(-1.5, 0.5, 1.5),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(-1.5, 0.5, -1.5),
        ..default()
    });
    // light
    commands.spawn(PointLightBundle {
        transform: Transform::from_xyz(3.0, 8.0, 5.0),
        ..default()
    });
}