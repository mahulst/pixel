//! Shows how to create a 3D orthographic view (for isometric-look games or CAD applications).

mod postpixel;
mod transform_animations;
mod isocam;

mod test_scene;
mod camera_setup;

use bevy::{
  prelude::*,
  window::WindowResized,
};

use bevy_editor_pls::EditorPlugin;

use transform_animations::TransformAnimationPlugin;

use crate::postpixel::{
  PostPixelPlugin,
  PostPixelSettings
};

use crate::isocam::IsoCameraPlugin;

fn main() {
  App::new()
    .insert_resource(Msaa::Off)
    // .insert_resource(DefaultOpaqueRendererMethod::deferred())
    .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
    .insert_resource(AmbientLight {
      color: Color::rgb(0.066, 0.066, 0.066),
      brightness: 0.5,
     })
    .add_plugins(DefaultPlugins)
    .add_plugins(PostPixelPlugin)
    // .add_plugins(ScreenSpaceAmbientOcclusionPlugin)

    .add_plugins(EditorPlugin::default())
    // .add_plugins(EditorPlugin::in_new_window(EditorPlugin::default(), Window::default()))
    // .add_plugins(WorldInspectorPlugin::new())

    .add_plugins(IsoCameraPlugin)
    .add_plugins(TransformAnimationPlugin)

    .add_plugins((
      camera_setup::CameraSetupPlugin,
      test_scene::TestScenePlugin,
    ))

    .add_systems(Update, resize_window)
    .run();
}

fn resize_window(
  mut resize_event: EventReader<WindowResized>,
  mut pps: Query<&mut PostPixelSettings>,
) {
  // let mut pps = pps.single_mut();
  for mut pps in pps.iter_mut() {
    for e in resize_event.read() {
      pps.pixel_scale = (e.height as f32 / 480.0).ceil();
    }
  }
}
