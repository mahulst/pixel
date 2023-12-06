use bevy::{
  core_pipeline::{
      core_3d,
      fullscreen_vertex_shader::fullscreen_shader_vertex_state, prepass::{ViewPrepassTextures, DepthPrepass},
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
          TextureViewDimension,
      },
      renderer::{RenderContext, RenderDevice},
      texture::BevyDefault,
      view::{ViewTarget, ViewUniforms, ViewUniform},
      RenderApp,
  }, window::WindowResized,
};


pub struct PostProcessPlugin;

impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<PostProcessSettings>::default(),
            UniformComponentPlugin::<PostProcessSettings>::default(),
        ));

        app.add_systems(Update, resize_window);

        app.insert_resource(ViewUniformCache {
            uniforms: None,
        });

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<ViewNodeRunner<PostProcessNode>>(
                core_3d::graph::NAME,
                PostProcessNode::NAME,
            )
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    core_3d::graph::node::TONEMAPPING,
                    PostProcessNode::NAME,
                    core_3d::graph::node::END_MAIN_PASS_POST_PROCESSING,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<PostProcessPipeline>();
    }
}

#[derive(Default)]
struct PostProcessNode;
impl PostProcessNode {
    pub const NAME: &'static str = "post_process";
}

impl ViewNode for PostProcessNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewPrepassTextures,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_target, prepass_textures): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let post_process_pipeline = world.resource::<PostProcessPipeline>();

        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(post_process_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let settings_uniforms = world.resource::<ComponentUniforms<PostProcessSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let (Some(prepass_depth_texture), Some(prepass_normal_texture)) = (&prepass_textures.depth, &prepass_textures.normal) else {
            return Ok(());
        };

        let Some(view_uniforms_resource) = world.get_resource::<ViewUniforms>() else {
            return Ok(());
        };
        let view_uniforms = &view_uniforms_resource.uniforms;

        let bind_group = render_context.render_device().create_bind_group(
            "post_process_bind_group",
            &post_process_pipeline.layout,
            &BindGroupEntries::sequential((
                view_uniforms,
                settings_binding.clone(),
                post_process.source,
                &post_process_pipeline.sampler,
                &prepass_depth_texture.default_view,
                &prepass_normal_texture.default_view,
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
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
struct PostProcessPipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for PostProcessPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("post_process_bind_group_layout"),
            entries: &[
                // View uniforms
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: bevy::render::render_resource::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
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
                        min_binding_size: Some(PostProcessSettings::min_size()),
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
            ],
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/post_processing.wgsl");

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

#[allow(bare_trait_objects)]
#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
pub struct PostProcessSettings {
    pub(crate) resolution: Vec2,
    pub(crate) pixel_scale: f32,
    pub(crate) forward: Vec3,
    // WebGL2 structs must be 16 byte aligned.
    #[cfg(feature = "webgl2")]
    _webgl2_padding: Vec3,
}

#[derive(Resource, Default)]
pub struct ViewUniformCache {
    pub(crate) uniforms: Option<Vec<ViewUniform>>,
}

fn resize_window(
    mut resize_event: EventReader<WindowResized>,
    mut post_process_settings: Query<&mut PostProcessSettings>,
) {
    for event in resize_event.read() {
        for mut settings in post_process_settings.iter_mut() {
            settings.resolution = Vec2::new(event.width as f32, event.height as f32);
        }
    }
}