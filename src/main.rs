//! Shows how to create a 3D orthographic view (for isometric-look games or CAD applications).

pub mod postprocessing;

use std::f32::consts::PI;

use bevy::{
    prelude::*,
    render::camera::ScalingMode,
    core_pipeline::{
        prepass::{DepthPrepass, MotionVectorPrepass, DeferredPrepass, NormalPrepass},
    },
};

use leafwing_input_manager::prelude::*;

use bevy_inspector_egui:: {
    prelude::*,
    quick::{
        WorldInspectorPlugin,
    }
};

use crate::postprocessing::postprocessing::{PostProcessPlugin, PostProcessSettings};

fn main() {
    let mut app = App::new();
    app
        .insert_resource(Msaa::Off)
        .register_type::<CameraState>()
        // .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .add_plugins(DefaultPlugins)
        .add_plugins(PostProcessPlugin)
        .add_plugins(WorldInspectorPlugin::new())
        .add_plugins(InputManagerPlugin::<Action>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, move_camera)
        .add_systems(Update, update_camera)
        .run();
    bevy_mod_debugdump::print_render_graph(&mut app);
}

#[derive(Default, Clone, Copy, Debug, Reflect)]
enum CameraDirection {
    NW = 3,  N = 2,  NE = 1,
     W = 4, #[default]E = 0,
    SW = 5,  S = 6,  SE = 7,
}

impl CameraDirection {
    fn from(n: i32) -> CameraDirection {
        match n {
            0 => CameraDirection::E,
            1 => CameraDirection::NE,
            2 => CameraDirection::N,
            3 => CameraDirection::NW,
            4 => CameraDirection::W,
            5 => CameraDirection::SW,
            6 => CameraDirection::S,
            7 => CameraDirection::SE,
            _ => CameraDirection::E,
        }
    }
}

#[derive(Default, Component, Clone, Copy, Debug, Reflect, InspectorOptions)]
#[reflect(Component, InspectorOptions)] 
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

fn lerp_angle(from: f32, to: f32, t: f32) -> f32 {
    let mut angle = to - from;
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    from + angle * t
}

fn update_camera(
    mut camera: Query<(&mut CameraState, &mut Transform)>,
    time: Res<Time>,
) {
    let mut camera = camera.single_mut();
    let state = &mut camera.0;
    let transform = &mut camera.1;

    let angle = state.direction as i32 as f32 / 8.0 * std::f32::consts::TAU;
    state.angle = lerp_angle(state.angle, angle, time.delta_seconds() * 10.0);

    let tf = Transform::from_xyz(5.0 * state.angle.sin(), 2.88675, 5.0 * state.angle.cos()).looking_at(Vec3::ZERO, Vec3::Y);
    transform.translation = tf.translation;
    transform.rotation = tf.rotation;
        // Transform::from_xyz(state.distance * (PI/4.0_f32).sin(), state.distance * (PI/6.0_f32.sin()), state.distance * (PI/4.0_f32).cos()).looking_at(Vec3::ZERO, Vec3::Y));
}


#[derive(Actionlike, PartialEq, Eq, Hash, Clone, Copy, Debug, Reflect)]
enum Action {
    CameraLeft,
    CameraRight,
}

fn move_camera(
    query: Query<&ActionState<Action>, With<Camera>>,
    mut camera: Query<&mut CameraState>,
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
    
    camera.direction = CameraDirection::from(new_dir);
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // let resolution = Vec2::new(640.0, 480.0);

    commands.spawn((
        Camera3dBundle {
            projection: OrthographicProjection {
                scale: 3.0,
                scaling_mode: ScalingMode::FixedVertical(2.0),
                near: 0.1,
                far: 16.0,
                ..default()
            }
            .into(),
            transform: Transform::from_xyz(5.0 * (PI/4.0_f32).sin(), 2.88675, 5.0 * (PI/4.0_f32).cos()).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        InputManagerBundle {
            action_state: ActionState::<Action>::default(),
            input_map: InputMap::new([(KeyCode::A, Action::CameraLeft), (KeyCode::D, Action::CameraRight)]),
        },
        CameraState {
            target: Vec3::ZERO,
            distance: 8.66,
            angle: 45.0 / 360.0 * std::f32::consts::TAU,
            direction: CameraDirection::NW,
        },
        PostProcessSettings {
            pixel_scale: 4.0,
            ..default()
        },
        DepthPrepass,
        NormalPrepass,
        MotionVectorPrepass,
        DeferredPrepass,
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
        mesh: meshes.add(Mesh::from(shape::UVSphere { radius: 0.5, sectors: 32, stacks: 16 })),
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