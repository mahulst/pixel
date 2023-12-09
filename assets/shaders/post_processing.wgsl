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
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: vec3<f32>
#endif
}
@group(0) @binding(4) var depth: texture_depth_2d;
@group(0) @binding(5) var normal: texture_2d<f32>;

const DITHER = array(0.0, 0.5, 0.25, 0.75);

fn lerpv(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn rgb_to_hsl(color: vec3<f32>) -> vec3<f32> {
    let c_max: f32 = max(max(color.r, color.g), color.b);
    let c_min: f32 = min(min(color.r, color.g), color.b);
    let delta: f32 = c_max - c_min;

    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let l: f32 = (c_max + c_min) / 2.0;

    if (delta > 0.0) {
        if (c_max == color.r) {
            h = (color.g - color.b) / delta;
        } else if (c_max == color.g) {
            h = 2.0 + (color.b - color.r) / delta;
        } else {
            h = 4.0 + (color.r - color.g) / delta;
        }

        if (l < 0.5) {
            s = delta / (c_max + c_min);
        } else {
            s = delta / (2.0 - c_max - c_min);
        }
    }

    return vec3<f32>(h, s, l);
}

fn hsl_to_rgb(color: vec3<f32>) -> vec3<f32> {
    let c: f32 = (1.0 - abs(2.0 * color.z - 1.0)) * color.y;
    let x: f32 = c * (1.0 - abs((color.x % 2.0) - 1.0));
    let m: f32 = color.z - c / 2.0;

    var rgb: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    if (color.x < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (color.x < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (color.x < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (color.x < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (color.x < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m, m, m);
}

fn get_depth(uv: vec2<f32>) -> f32 {
    return textureSample(depth, texture_sampler, uv);
}

fn get_linear_depth(uv: vec2<f32>) -> f32 {
    let z_near: f32 = 0.1;
    let z_far: f32 = 16.0;
    var d: f32 = get_depth(uv);
    return d * (z_far - z_near) + z_near;
}

fn get_normal(uv: vec2<f32>) -> vec3<f32> {
    return textureSample(normal, texture_sampler, uv).rgb * 2.0 - 1.0;
}

fn get_ss_normal(uv: vec2<f32>) -> vec3<f32> {
    var normal = get_normal(uv);
    if (get_depth(uv) < 0.001) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let inverse_view = mat3x3<f32>(
        view.inverse_view[0].xyz,
        view.inverse_view[1].xyz,
        view.inverse_view[2].xyz,
    );
    return inverse_view * normal;
}



fn f_depth_edge(uv: vec2<f32>, texel: vec2<f32>) -> f32 {
    let depth = get_linear_depth(uv);
    let normal = get_ss_normal(uv);

    var diff = 0.0;
    diff += clamp(get_linear_depth(uv + vec2<f32>(1.0, 0.0) * texel) - depth, 0.0, 1.0);
    diff += clamp(get_linear_depth(uv + vec2<f32>(-1.0, 0.0) * texel) - depth, 0.0, 1.0);
    diff += clamp(get_linear_depth(uv + vec2<f32>(0.0, 1.0) * texel) - depth, 0.0, 1.0);
    diff += clamp(get_linear_depth(uv + vec2<f32>(0.0, -1.0) * texel) - depth, 0.0, 1.0);

    // var factor = pow(1.0 - dot(normal, vec3(0.0, 0.0, 1.0)), 3.0);
    // factor = lerpf(0.01, 0.66, factor);

    let ndotv = dot(normal, vec3(0.0, 0.0, 1.0));

    let depth_threshold = 0.3;

    var threshold = saturate((ndotv - depth_threshold) / (1.0 - depth_threshold));

    // return dot(normal, vec3(0.0, 0.0, 1.0));
    return floor(smoothstep(threshold, threshold * 2.0, diff) * 2.0) / 2.0;
}





fn linear_depth(depth: f32) -> f32 {
    let z_near: f32 = 0.1;
    let z_far: f32 = 16.0;
    var d = depth;
    return d * (z_far - z_near) + z_near;
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

fn screen_normal(uv: vec2<f32>) -> vec3<f32> {
    var normal = textureSample(normal, texture_sampler, uv).rgb;
    normal = (normal * 2.0) - 1.0;
    let inverse_view = mat3x3<f32>(
        view.inverse_view[0].xyz,
        view.inverse_view[1].xyz,
        view.inverse_view[2].xyz,
    );
    return inverse_view * normal;
}


fn getDepth(uv: vec2<f32>, texel: vec2<f32>, x: f32, y: f32) -> f32 {
    // return textureSample(depth, texture_sampler, uv + vec2<f32>(x, y) * texel);
    return sample_depth(uv + vec2<f32>(x, y) * texel);
}

fn getNormal(uv: vec2<f32>, texel: vec2<f32>, x: f32, y: f32) -> vec3<f32> {
    // return textureSample(normal, texture_sampler, uv + vec2<f32>(x, y) * texel ).rgb * 2.0 - 1.0;
    let normal = screen_normal(uv + vec2<f32>(x, y) * texel);
    if (length(normal) < 0.9) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normal;
}

fn neighborNormalEdgeIndicator(uv: vec2<f32>, texel: vec2<f32>, x: f32, y: f32, depth: f32, normal: vec3<f32>) -> f32 {
    let depthDiff: f32 = depth - getDepth(uv, texel, x, y);

    // Edge pixels should yield to faces closer to the bias direction.
    let normalEdgeBias: vec3<f32> = vec3(0.25); // This should probably be a parameter.
    let normalDiff: f32 = dot(normal - getNormal(uv, texel, x, y), normalEdgeBias);
    let normalIndicator: f32 = clamp(smoothstep(-.01, .01, normalDiff), 0.0, 1.0);

    // Only the shallower pixel should detect the normal edge.
    let depthIndicator: f32 = clamp(sign(depthDiff * .25 + .025), 0.0, 1.0);

    return (1.0 - clamp(dot(normal, getNormal(uv, texel, x, y)), 0.0, 1.0)) * depthIndicator * normalIndicator;
    // return dot(normal.yzx, getNormal(uv, texel, x, y)) * depthIndicator * normalIndicator;
}

fn depthEdgeIndicator(uv: vec2<f32>, texel: vec2<f32>) -> f32 {
    let depth: f32 = getDepth(uv, texel, 0.0, 0.0);
    let normal: vec3<f32> = getNormal(uv, texel, 0.0, 0.0);
    var diff: f32 = 0.0;
    diff += clamp(getDepth(uv, texel, 1.0, 0.0) - depth, 0.0, 1.0);
    diff += clamp(getDepth(uv, texel, -1.0, 0.0) - depth, 0.0, 1.0);
    diff += clamp(getDepth(uv, texel, 0.0, 1.0) - depth, 0.0, 1.0);
    diff += clamp(getDepth(uv, texel, 0.0, -1.0) - depth, 0.0, 1.0);
    return floor(smoothstep(0.1, 0.2, diff) * 2.) / 2.;
}

fn normalEdgeIndicator(uv: vec2<f32>, texel: vec2<f32>) -> f32 {
    let depth: f32 = getDepth(uv, texel, 0.0, 0.0);
    let normal: vec3<f32> = getNormal(uv, texel, 0.0, 0.0);
    
    var indicator: f32 = 0.0;

    indicator += neighborNormalEdgeIndicator(uv, texel, 0.0, -1.0, depth, normal);
    indicator += neighborNormalEdgeIndicator(uv, texel, 0.0, 1.0, depth, normal);
    indicator += neighborNormalEdgeIndicator(uv, texel, -1.0, 0.0, depth, normal);
    indicator += neighborNormalEdgeIndicator(uv, texel, 1.0, 0.0, depth, normal);

    return step(0.125, indicator);
}

fn depthEdge(uv: vec2<f32>, texel: vec2<f32>) -> f32 {
    let d: f32 = getDepth(uv, texel, 0.0, 0.0);
    let d_l: f32 = getDepth(uv, texel, -1.0, 0.0);
    let d_u: f32 = getDepth(uv, texel, 0.0, -1.0);
    let d_r: f32 = getDepth(uv, texel, 1.0, 0.0);
    let d_d: f32 = getDepth(uv, texel, 0.0, 1.0);

    // return step(d, closest);
    let normal = get_ss_normal(uv);
    let ndotv = dot(normal, vec3(0.0, 0.0, 1.0));

    let curve = curvature(uv, texel);

    let closest: f32 = min(min(d_l, d_r), min(d_u, d_d));
    let closer: f32 = 1.0 - step(d, closest);

    // Bias depth edge test from curvature.
    var bias = lerpf(0.1, 1.0, clamp(ndotv, 0.0, 1.0));

    // Make bias resolution independent
    bias *= texel.y * 256.0;

    let diff: f32 = d - closest;
    return step(bias, diff * closer);
    // return curve * 10.0;
    // return closer;
}

fn normalEdge(uv: vec2<f32>, texel: vec2<f32>) -> f32 {
    // let bias_normal: vec3<f32> = vec3<f32>(0.1, 0.97, 0.11);
    let bias_normal: vec3<f32> = vec3<f32>(0.25);

    let n: vec3<f32> = getNormal(uv, texel, 0.0, 0.0);
    let n_l: vec3<f32> = getNormal(uv, texel, -1.0, 0.0);
    let n_u: vec3<f32> = getNormal(uv, texel, 0.0, -1.0);
    let n_r: vec3<f32> = getNormal(uv, texel, 1.0, 0.0);
    let n_d: vec3<f32> = getNormal(uv, texel, 0.0, 1.0);

    let dot_n: f32 = dot(n, bias_normal);
    let dot_n_l: f32 = dot(n_l, bias_normal);
    let dot_n_u: f32 = dot(n_u, bias_normal);
    let dot_n_r: f32 = dot(n_r, bias_normal);
    let dot_n_d: f32 = dot(n_d, bias_normal);

    let closest: f32 = max(max(dot_n_l, dot_n_r), max(dot_n_u, dot_n_d));
    let closer: f32 = 1.0 - step(0.0, dot_n - closest);

    let diff: f32 = max(max(1.0 - dot(n_l, n), 1.0 - dot(n_u, n)), max(1.0 - dot(n_r, n), 1.0 - dot(n_d, n)));
    return step(0.1, diff * closer);
    // return closer;
}

fn curvature(uv: vec2<f32>, texel: vec2<f32>) -> f32 {
    let n: vec3<f32> = getNormal(uv, texel, 0.0, 0.0);
    let n_l: vec3<f32> = getNormal(uv, texel, -1.0, 0.0);
    let n_u: vec3<f32> = getNormal(uv, texel, 0.0, -1.0);
    let n_r: vec3<f32> = getNormal(uv, texel, 1.0, 0.0);
    let n_d: vec3<f32> = getNormal(uv, texel, 0.0, 1.0);

    let ddx: vec3<f32> = (n_r - n_l) * 0.5;
    let ddy: vec3<f32> = (n_d - n_u) * 0.5;

    let bias_normal: vec3<f32> = vec3(0.25);

    let dot_n: f32 = dot(n, bias_normal);
    let dot_n_l: f32 = dot(n_l, bias_normal);
    let dot_n_u: f32 = dot(n_u, bias_normal);
    let dot_n_r: f32 = dot(n_r, bias_normal);
    let dot_n_d: f32 = dot(n_d, bias_normal);

    let d: f32 = getDepth(uv, texel, 0.0, 0.0);
    let d_l: f32 = getDepth(uv, texel, -1.0, 0.0);
    let d_u: f32 = getDepth(uv, texel, 0.0, -1.0);
    let d_r: f32 = getDepth(uv, texel, 1.0, 0.0);
    let d_d: f32 = getDepth(uv, texel, 0.0, 1.0);

    let closest: f32 = 1.0 - step(max(max(dot_n_l, dot_n_r), max(dot_n_u, dot_n_d)), dot_n);

    let xneg = n - ddx;
    let xpos = n + ddx;
    let yneg = n - ddy;
    let ypos = n + ddy;

    let depth = getDepth(uv, texel, 0.0, 0.0);

    let curvature = (cross(xneg, xpos).y + cross(yneg, ypos).x) * closest;

    return curvature;
}


fn get_texel_size() -> vec2<f32> {
    let px: f32 = settings.pixel_scale;
    let res: vec2<f32> = (settings.resolution + vec2<f32>(1.0, 1.0)) / px;
    return vec2<f32>(1.0, 1.0) / res;
}

fn pixelate_uv(uv: vec2<f32>, texel: vec2<f32>) -> vec2<f32> {
    return floor(uv / texel) * texel + texel * 0.5;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel: vec2<f32> = get_texel_size();
    let uv: vec2<f32> = pixelate_uv(in.uv, texel);
    // let uv: vec2<f32> = in.uv;

    let raw_depth: f32 = textureSample(depth, texture_sampler, uv);
    let depth: f32 = linear_depth(raw_depth);
    let raw_normal: vec3<f32> = textureSample(normal, texture_sampler, uv).rgb;
    let normal: vec3<f32> = decode_normal(raw_normal);

    let raw_color: vec3<f32> = textureSample(screen_texture, texture_sampler, uv).rgb;
    var color: vec3<f32> = raw_color;

    let depth_edge: f32 = depthEdge(uv, texel);
    let normal_edge: f32 = normalEdgeIndicator(uv, texel) * step(-0.5, -depth_edge);

    let mask = step(1.0, depth);

    let luminance: f32 = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

    var curvature = curvature(uv, texel);
    curvature = step(0.66, abs(curvature)) * sign(curvature) * step(-0.5, -depth_edge);
    // curvature = curvature - depth_edge;


    var hsl = rgb_to_hsl(color);
    hsl.x = lerpf(hsl.x, 0.66, depth_edge * 0.25);
    hsl.z = lerpf(hsl.z, 0.0, depth_edge * 0.5);
    // hsl.z = lerpf(hsl.z, 1.0, clamp(curvature, 0.0, 1.0) * 0.15);
    color = hsl_to_rgb(hsl);

    // color = vec3(luminance * mask);

    // color.r = depth_edge;
    // color.g = normal_edge;
    // color.b = step(0.01, color.b);
    // color *= mask;

    // color.r *= 0.0;
    // color.g *= 0.0;
    // color.b *= 0.0;

    // color.r = clamp(curvature(uv, texel), 0.0, 1.0);
    // color.b = mask;
    // color.g = mask;

    return vec4<f32>(color, 1.0);
}
