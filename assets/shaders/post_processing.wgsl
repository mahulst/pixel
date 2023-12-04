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
#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
// struct FullscreenVertexOutput {
//     @builtin(position)
//     position: vec4<f32>,
//     @location(0)
//     uv: vec2<f32>,
// };

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
struct PostProcessSettings {
    intensity: f32,
    resolution: vec2<f32>,
    pixel_scale: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: vec3<f32>
#endif
}
@group(0) @binding(2) var<uniform> settings: PostProcessSettings;
@group(0) @binding(3) var depth: texture_depth_2d;
@group(0) @binding(4) var normal: texture_2d<f32>;

const DITHER = array(0.0, 0.5, 0.25, 0.75);

fn linear_depth(depth: f32) -> f32 {
    let z_near: f32 = 0.1;
    let z_far: f32 = 100.0;
    return (2.0 * z_near) / (z_far + z_near - depth * (z_far - z_near));
}

fn decode_normal(n: vec3<f32>) -> vec3<f32> {
    return n * 2.0 - 1.0;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // Chromatic aberration strength
    let offset_strength: f32 = settings.intensity;
    let px: f32 = settings.pixel_scale;
    let res: vec2<f32> = (settings.resolution + vec2<f32>(1.0, 1.0)) / px;
    
    let uv: vec2<f32> = in.uv;

    // let puv: vec2<f32> = floor(uv * res) / res;
    let texel_size: vec3<f32> = vec3<f32>(1.0, 1.0, 0.0) / vec3<f32>(res.x, res.y, 1.0);

    let pixel: vec2<i32> = vec2<i32>(uv * res);
    let dither: f32 = 1.0;
    
    let puv = vec2<f32>(pixel) / res;

    let c: vec3<f32> = textureSample(screen_texture, texture_sampler, puv).rgb;

    let d: f32 = linear_depth(textureSample(depth, texture_sampler, puv));
    let d_l: f32 = linear_depth(textureSample(depth, texture_sampler, puv - texel_size.xz));
    let d_u: f32 = linear_depth(textureSample(depth, texture_sampler, puv - texel_size.zy));
    let d_r: f32 = linear_depth(textureSample(depth, texture_sampler, puv + texel_size.xz));
    let d_d: f32 = linear_depth(textureSample(depth, texture_sampler, puv + texel_size.zy));

    let d_diff: f32 = max(max(d_l - d, d_u - d), max(d_r - d, d_d - d));

    let d_px: f32 = step(0.01, d_diff);

    let d2: f32 = pow(d, 256.0);

    let r = pow(round(pow(c.r, 1.0/2.2) * 16.0) / 16.0, 2.2);
    let g = pow(round(pow(c.g, 1.0/2.2) * 16.0) / 16.0, 2.2);
    let b = pow(round(pow(c.b, 1.0/2.2) * 16.0) / 16.0, 2.2);

    let color = vec3<f32>(d_px);

    let n = decode_normal(textureSample(normal, texture_sampler, puv).rgb);

    let n_l = decode_normal(textureSample(normal, texture_sampler, puv - texel_size.xz).rgb);
    let n_u = decode_normal(textureSample(normal, texture_sampler, puv - texel_size.zy).rgb);
    let n_r = decode_normal(textureSample(normal, texture_sampler, puv + texel_size.xz).rgb);
    let n_d = decode_normal(textureSample(normal, texture_sampler, puv + texel_size.zy).rgb);

    let n_diff = min(min(dot(n_l, n), dot(n_u, n)), min(dot(n_r, n), dot(n_d, n)));

    return vec4<f32>(n_diff, d_diff, 0.0, 1.0);

    // return vec4<f32>(color, 1.0);
}
