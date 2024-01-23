use bevy::prelude::*;
pub struct TransformAnimationPlugin;

impl Plugin for TransformAnimationPlugin {
  fn build(&self, app: &mut App) {
    app.register_type::<AnimatedTransform>();
    app.register_type::<AnimatedTransformPart>();

    app.add_systems(Update, process_animations);

  }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Reflect)]
pub enum TransformAnimation {
  None,
  Circle,
  Bob,
}

#[derive(Copy, Clone, Debug, Reflect)]
pub struct AnimatedTransformPart {
  pub animation: TransformAnimation,
  pub speed: f32,
  pub scale: f32,
}

#[derive(Component, Clone, Debug, Reflect)]
pub struct AnimatedTransform {
  pub origin: Transform,
  pub animations: Vec<AnimatedTransformPart>,
}

impl AnimatedTransform {
  fn animate(&mut self, time: &Time) -> Transform {
    let mut transform = self.origin.clone();
    for animation in &self.animations {
      transform = match animation.animation {
        TransformAnimation::None => self.origin,
        TransformAnimation::Circle => {
          AnimatedTransform::animate_circle(transform, *animation, time)
        }
        TransformAnimation::Bob => {
          AnimatedTransform::animate_bob(transform, *animation, time)
        }
      }
    }
    transform
  }

  fn animate_circle(
    transform: Transform,
    animation: AnimatedTransformPart,
    time: &Time,
  ) -> Transform {
    let mut transform = transform;
    transform.translation.x +=
      (animation.speed * time.elapsed_seconds() as f32).sin() * animation.scale;
    transform.translation.z +=
      (animation.speed * time.elapsed_seconds() as f32).cos() * animation.scale;
    transform
  }

  fn animate_bob(
    transform: Transform,
    animation: AnimatedTransformPart,
    time: &Time,
  ) -> Transform {
    let mut transform = transform;
    transform.translation.y +=
      (animation.speed * time.elapsed_seconds() as f32).sin() * animation.scale;
    transform
  }
}

fn process_animations(time: Res<Time>, mut anims: Query<(&mut Transform, &mut AnimatedTransform)>) {
  for (mut transform, mut animation) in anims.iter_mut() {
    *transform = animation.animate(&time);
  }
}
