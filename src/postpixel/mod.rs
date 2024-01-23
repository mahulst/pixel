use bevy::{
  core_pipeline::{
    core_3d,
    fullscreen_vertex_shader::fullscreen_shader_vertex_state, prepass::ViewPrepassTextures,
  },
  ecs::query::QueryItem,
  prelude::*,
  render::{
    extract_component::{
      ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
    },
    render_graph::{
      NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner,
    },
    render_resource::{
      BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
      BindingType, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
      MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
      RenderPassDescriptor, RenderPipelineDescriptor, Sampler, SamplerBindingType,
      SamplerDescriptor, ShaderStages, ShaderType, TextureFormat, TextureSampleType,
      TextureViewDimension, TextureDimension, Extent3d, TextureUsages,
    },
    renderer::{RenderContext, RenderDevice},
    texture::{BevyDefault, CompressedImageFormats, ImageType},
    view::{ViewTarget, ViewUniforms, ViewUniform, ViewUniformOffset},
    RenderApp, render_asset::RenderAssets, extract_resource::{ExtractResourcePlugin, ExtractResource},
  }, window::WindowResized,
};

#[derive(Resource, Clone, ExtractResource)]
pub struct PixelArtLUT {
  db32: Handle<Image>
}

pub struct PostPixelPlugin;

impl Plugin for PostPixelPlugin {
  fn build(&self, app: &mut App) {
    // app.register_type::<PostPixelSettings>();
    
    app.add_plugins((
      ExtractComponentPlugin::<PostPixelSettings>::default(),
      UniformComponentPlugin::<PostPixelSettings>::default(),
    ));

    app.add_systems(Update, resize_window);

    if !app.world.is_resource_added::<PixelArtLUT>() {
      let mut images = app.world.resource_mut::<Assets<Image>>();

      let mut lut_texture = Image::from_buffer(
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/luts/db32.png")),
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        false,
        bevy::render::texture::ImageSampler::nearest(),
      ).unwrap();

      lut_texture.texture_descriptor.dimension = TextureDimension::D3;
      lut_texture.texture_descriptor.size = Extent3d {
        width: 64,
        height: 64,
        depth_or_array_layers: 64,
      };
      lut_texture.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
      lut_texture.texture_descriptor.format = TextureFormat::Rgba8Unorm;

      let lut = PixelArtLUT {
        db32: images.add(lut_texture),
      };
      
      app.insert_resource(lut);
    }

    app.add_plugins(ExtractResourcePlugin::<PixelArtLUT>::default());

    let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };

    render_app
      // .add_render_graph_node::<ViewNodeRunner<SobelNode>>(
      //   core_3d::graph::NAME,
      //   SobelNode::NAME,
      // )
      .add_render_graph_node::<ViewNodeRunner<PostPixelNode>>(
        core_3d::graph::NAME,
        PostPixelNode::NAME,
      )
      .add_render_graph_edges(
        core_3d::graph::NAME,
        &[
          core_3d::graph::node::TONEMAPPING,
          PostPixelNode::NAME,
          core_3d::graph::node::END_MAIN_PASS_POST_PROCESSING,
        ],
      );
  }

  fn finish(&self, app: &mut App) {
    let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };

    render_app
      .init_resource::<PostPixelPipeline>();
  }
}

#[derive(Default)]
struct SobelNode;
impl SobelNode {
  pub const NAME: &'static str = "sobel_filter";
}

#[derive(Default)]
struct PostPixelNode;
impl PostPixelNode {
  pub const NAME: &'static str = "post_process";
}

impl ViewNode for PostPixelNode {
  type ViewQuery = (
    &'static ViewTarget,
    &'static ViewPrepassTextures,
    &'static ViewUniformOffset,
    &'static PostPixelSettings,
  );

  fn run(
    &self,
    _graph: &mut RenderGraphContext,
    render_context: &mut RenderContext,
    (view_target, prepass_textures, view_uniform_offset, _): QueryItem<Self::ViewQuery>,
    world: &World,
  ) -> Result<(), NodeRunError> {
    let post_process_pipeline = world.resource::<PostPixelPipeline>();

    let pipeline_cache = world.resource::<PipelineCache>();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(post_process_pipeline.pipeline_id)
    else {
      return Ok(());
    };

    let settings_uniforms = world.resource::<ComponentUniforms<PostPixelSettings>>();
    let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
      return Ok(());
    };

    let post_process = view_target.post_process_write();

    let (
      Some(prepass_depth_texture),
      Some(prepass_normal_texture),
      Some(prepass_deferred_texture),
    ) = (
      &prepass_textures.depth,
      &prepass_textures.normal,
      &prepass_textures.deferred,
    ) else {
      return Ok(());
    };

    let images = world.resource::<RenderAssets<Image>>();
    let lut_res = world.resource::<PixelArtLUT>();

    let lut_image = images.get(&lut_res.db32).unwrap();

    let view_uniforms_resource = world.resource::<ViewUniforms>();

    let bind_group = render_context.render_device().create_bind_group(
      "post_process_bind_group",
      &post_process_pipeline.layout,
      &BindGroupEntries::sequential((
        &view_uniforms_resource.uniforms,
        settings_binding.clone(),
        post_process.source,
        &post_process_pipeline.sampler,
        &prepass_depth_texture.default_view,
        &prepass_normal_texture.default_view,
        &prepass_deferred_texture.default_view,
        &lut_image.texture_view,
      )),
    );

    let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
      label: Some("post_process_pass"),
      color_attachments: &[Some(RenderPassColorAttachment {
        view: post_process.destination,
        resolve_target: None,
        ops: Operations::default(),
      })],
      depth_stencil_attachment: None,
    });

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
    render_pass.draw(0..3, 0..1);

    Ok(())
  }
}

#[derive(Resource)]
struct PostPixelPipeline {
  layout: BindGroupLayout,
  sampler: Sampler,
  pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for PostPixelPipeline {
  fn from_world(world: &mut World) -> Self {
    let render_device = world.resource::<RenderDevice>();

    // let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
      label: Some("post_process_bind_group_layout"),
      entries: &[
        // View uniforms
        BindGroupLayoutEntry {
          binding: 0,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Buffer {
            ty: bevy::render::render_resource::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(ViewUniform::min_size()),
          },
          count: None,
        },
        // Shader uniforms
        BindGroupLayoutEntry {
          binding: 1,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Buffer {
            ty: bevy::render::render_resource::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(PostPixelSettings::min_size()),
          },
          count: None,
        },
        // The screen texture
        BindGroupLayoutEntry {
          binding: 2,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        },
        // The sampler that will be used to sample the screen texture
        BindGroupLayoutEntry {
          binding: 3,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Sampler(SamplerBindingType::Filtering),
          count: None,
        },
        // Depth prepass texture
        BindGroupLayoutEntry {
          binding: 4,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        },
        // Normal prepass texture
        BindGroupLayoutEntry {
          binding: 5,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        },
        // Deferred prepass texture
        BindGroupLayoutEntry {
          binding: 6,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Texture {
            sample_type: TextureSampleType::Uint,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
          },
          count: None,
        },
        // LUT texture
        BindGroupLayoutEntry {
          binding: 7,
          visibility: ShaderStages::FRAGMENT,
          ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D3,
            multisampled: false,
          },
          count: None,
        },
      ],
    });

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let shader = world
      .resource::<AssetServer>()
      .load("shaders/pixel.wgsl");

    let pipeline_id = world
      .resource_mut::<PipelineCache>()
      .queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("post_process_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader_vertex_state(),
        fragment: Some(FragmentState {
          shader,
          shader_defs: vec![],
          entry_point: "fragment".into(),
          targets: vec![Some(ColorTargetState {
            format: TextureFormat::bevy_default(),
            blend: None,
            write_mask: ColorWrites::ALL,
          })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        push_constant_ranges: vec![],
      });

    Self {
      layout,
      sampler,
      pipeline_id,
    }
  }
}

#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
// #[derive(Component, Reflect, Default, Clone, Copy, ExtractComponent, ShaderType)]
// #[reflect(Component)]
pub struct PostPixelSettings {
  pub(crate) resolution: Vec2,
  pub(crate) pixel_scale: f32,
  pub(crate) dither_offset: Vec2,
  #[cfg(feature = "webgl2")] // WebGL2 structs must be 16 byte aligned.
  _webgl2_padding: Vec3,
}

fn resize_window(
  mut resize_event: EventReader<WindowResized>,
  mut post_process_settings: Query<&mut PostPixelSettings>,
) {
  for event in resize_event.read() {
    for mut settings in post_process_settings.iter_mut() {
      settings.resolution = Vec2::new(event.width as f32, event.height as f32);
    }
  }
}
