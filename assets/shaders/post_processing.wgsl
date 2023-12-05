// This shader computes the chromatic aberration effect

// Since post processing is a fullscreen effect, we use the fullscreen vertex shader provided by bevy.
// This will import a vertex shader that renders a single fullscreen triangle.
//
// A fullscreen triangle is a single triangle that covers the entire screen.
// The box in the top left in that diagram is the screen. The 4 x are the corner of the screen
//
// Y axis
//  1 |  x-----x......
//  0 |  |  s  |  . ´
// -1 |  x_____x´
// -2 |  :  .´
// -3 |  :´
//    +---------------  X axis
//      -1  0  1  2  3
//
// As you can see, the triangle ends up bigger than the screen.
//
// You don't need to worry about this too much since bevy will compute the correct UVs for you.
// #import bevy_render::{
//     view::View,
//     globals::Globals,
// }
struct ColorGrading {
    exposure: f32,
    gamma: f32,
    pre_saturation: f32,
    post_saturation: f32,
}

struct View {
    view_proj: mat4x4<f32>,
    unjittered_view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    // viewport(x_origin, y_origin, width, height)
    viewport: vec4<f32>,
    frustum: array<vec4<f32>, 6>,
    color_grading: ColorGrading,
    mip_bias: f32,
};

// #import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
struct FullscreenVertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> settings: PostProcessSettings;
@group(0) @binding(2) var screen_texture: texture_2d<f32>;
@group(0) @binding(3) var texture_sampler: sampler;
struct PostProcessSettings {
    resolution: vec2<f32>,
    pixel_scale: f32,
    forward: vec3<f32>,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: vec3<f32>
#endif
}
@group(0) @binding(4) var depth: texture_depth_2d;
@group(0) @binding(5) var normal: texture_2d<f32>;

const DITHER = array(0.0, 0.5, 0.25, 0.75);

fn linear_depth(depth: f32) -> f32 {
    let z_near: f32 = 0.1;
    let z_far: f32 = 100.0;
    return (1.0 / depth) * (z_far - z_near) - z_far / (z_far - z_near);
}

fn decode_normal(n: vec3<f32>) -> vec3<f32> {
    return n * 2.0 - 1.0;
}

fn sample_depth(uv: vec2<f32>) -> f32 {
    return linear_depth(textureSample(depth, texture_sampler, uv));
}

fn sample_normal(uv: vec2<f32>) -> vec3<f32> {
    return decode_normal(textureSample(normal, texture_sampler, uv).rgb);
}



@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Pixel size
    let px: f32 = settings.pixel_scale;
    // Resolution
    let res: vec2<f32> = (settings.resolution + vec2<f32>(1.0, 1.0)) / px;
    
    let forward: vec3<f32> = settings.forward;

    // Texel size
    let texel: vec3<f32> = vec3<f32>(1.0, 1.0, 0.0) / vec3<f32>(res.x, res.y, 1.0);
    // UV coordinates
    let uv: vec2<f32> = in.uv;

    // Pixel coordinates
    let pixel: vec2<i32> = vec2<i32>(uv * res);

    // Dither factor
    let dither: f32 = 1.0;
    
    // Pixellated UV coordinates
    let puv = vec2<f32>(pixel) / res;

    let c: vec3<f32> = textureSample(screen_texture, texture_sampler, puv).rgb;

    let d: f32 = linear_depth(textureSample(depth, texture_sampler, puv));
    let d_l: f32 = linear_depth(textureSample(depth, texture_sampler, puv - texel.xz));
    let d_u: f32 = linear_depth(textureSample(depth, texture_sampler, puv - texel.zy));
    let d_r: f32 = linear_depth(textureSample(depth, texture_sampler, puv + texel.xz));
    let d_d: f32 = linear_depth(textureSample(depth, texture_sampler, puv + texel.zy));

    let d_diff: f32 = max(max(d - d_l, d - d_u), max(d - d_r, d - d_d));

    let depth_threshold: f32 = 0.01;

    let d_px: f32 = step(depth_threshold, d_diff);

    // let d2: f32 = pow(d, 256.0);
    let d2 = linear_depth(d);

    let r = pow(round(pow(c.r, 1.0/2.2) * 16.0) / 16.0, 2.2);
    let g = pow(round(pow(c.g, 1.0/2.2) * 16.0) / 16.0, 2.2);
    let b = pow(round(pow(c.b, 1.0/2.2) * 16.0) / 16.0, 2.2);

    let color = vec3<f32>(d_px);

    let n = decode_normal(textureSample(normal, texture_sampler, puv).rgb);

    let n_l = decode_normal(textureSample(normal, texture_sampler, puv - texel.xz).rgb);
    let n_u = decode_normal(textureSample(normal, texture_sampler, puv - texel.zy).rgb);
    let n_r = decode_normal(textureSample(normal, texture_sampler, puv + texel.xz).rgb);
    let n_d = decode_normal(textureSample(normal, texture_sampler, puv + texel.zy).rgb);

    let dd1 = d - d_r;
    let dd2 = d - d_d;

    let depth_diff = sqrt(pow(dd1, 2.0) + pow(dd2, 2.0)) * 100.0;

    let nd1 = n_l - n_r;
    let nd2 = n_u - n_d;

    let edge_norm = sqrt(dot(nd1, nd1) + dot(nd2, nd2));

    let norm_diff = step(edge_norm, 0.5);

    let n_diff = max(max(1.0 - dot(n_l, n), 1.0 - dot(n_u, n)), max(1.0 - dot(n_r, n), 1.0 - dot(n_d, n)));

    let pos = view.inverse_projection * vec4<f32>(puv * 2.0 - 1.0, d2, 1.0);

    let vnorm = view.view * vec4<f32>(0.0, 0.0, 1.0, 1.0);

    return vec4<f32>(view.world_position, 1.0);
    // return vec4<f32>(d_px, forward.x, 0.0, 1.0);

    // return vec4<f32>(color, 1.0);
}
