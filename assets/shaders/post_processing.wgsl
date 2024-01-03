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
#import bevy_render::view::View;
#import bevy_render::globals::Globals;
// #import bevy_render::{
//     view::View,
//     globals::Globals,
// }
// struct ColorGrading {
//     exposure: f32,
//     gamma: f32,
//     pre_saturation: f32,
//     post_saturation: f32,
// }

// struct View {
//     view_proj: mat4x4<f32>,
//     unjittered_view_proj: mat4x4<f32>,
//     inverse_view_proj: mat4x4<f32>,
//     view: mat4x4<f32>,
//     inverse_view: mat4x4<f32>,
//     projection: mat4x4<f32>,
//     inverse_projection: mat4x4<f32>,
//     world_position: vec3<f32>,
//     // viewport(x_origin, y_origin, width, height)
//     viewport: vec4<f32>,
//     frustum: array<vec4<f32>, 6>,
//     color_grading: ColorGrading,
//     mip_bias: f32,
// };

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
// struct FullscreenVertexOutput {
//     @builtin(position)
//     position: vec4<f32>,
//     @location(0)
//     uv: vec2<f32>,
// };

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
@group(0) @binding(6) var deferred: texture_2d<u32>;
@group(0) @binding(7) var lut: texture_3d<f32>;

fn lerpv(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * clamp(t, 0., 1.);
}

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * clamp(t, 0., 1.);
}

fn gamma_correct(color: vec3<f32>) -> vec3<f32> {
    return pow(color, vec3<f32>(2.2, 2.2, 2.2));
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

fn get_texel_size() -> vec2<f32> {
    let px: f32 = settings.pixel_scale;
    let res: vec2<f32> = (settings.resolution) / px;
    return vec2<f32>(1.0, 1.0) / res;
}

fn pixelate_uv(uv: vec2<f32>, texel: vec2<f32>) -> vec2<f32> {
    return floor(uv / texel) * texel; // + texel * 0.5
}

fn ref_lut(uv: vec2<f32>) -> vec3<f32> {
    let r = fract(uv.x * 8.0);
    let g = fract(uv.y * 8.0);
    let b = floor(uv.x * 8.0) / 8.0 / 8.0 + floor(uv.y * 8.0) / 9.0;
    return vec3(r, g, b);
}

const LUT_HTX: f32 = 0.015625;
fn color_to_lut(color: vec3<f32>) -> vec3<f32> {
    // let lut_uv = get_lut_uv(color);
    let r = floor(color.r * 64.0) / 64.0 + LUT_HTX;
    let g = floor(color.g * 64.0) / 64.0 + LUT_HTX;
    let b = floor(color.b * 64.0) / 64.0 + LUT_HTX;
    let luv = vec3(r, g, b);
    return textureSample(lut, texture_sampler, color.rbg).rgb;
}

fn cbrt(x: f32) -> f32 {
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

fn rgb_to_oklab(color: vec3<f32>) -> vec3<f32> {
    var l: f32 = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    var m: f32 = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    var s: f32 = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;

    let l_ = cbrt(l);
    let m_ = cbrt(m);
    let s_ = cbrt(s);

    l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
    m = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
    s = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

    return vec3(l, m, s);
}

fn oklab_to_rgb(color: vec3<f32>) -> vec3<f32> {
    let l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
    let m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
    let s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;

    let l: f32 = l_ * l_ * l_;
    let m: f32 = m_ * m_ * m_;
    let s: f32 = s_ * s_ * s_;

    let r: f32 = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
    let g: f32 = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
    let b: f32 = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

    return vec3(r, g, b);
}

fn dither(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    // let pixel: vec2<i32> = vec2(i32(uv.x * settings.resolution.x / settings.pixel_scale), i32(uv.y * settings.resolution.y / settings.pixel_scale));
    let puv = uv * settings.resolution / settings.pixel_scale;
    let pixel: vec2<i32> = vec2(i32(puv.x), i32(puv.y));
    var DITHER: array<i32, 16> = array(
         0,  8,  2, 10,
        12,  4, 14,  6,
         3, 11,  1,  9,
        15,  7, 13,  5
    );
    let x: i32 = pixel.x % 4;
    let y: i32 = pixel.y % 4;
    let i: i32 = x + y * 4;
    let d: f32 = (f32(DITHER[i]) / 16. - 0.5) * (0.5 / 64.0);
    var cout = saturate(color + d);
    // cout.r = f32(x % 4) / 4.;
    // cout.g = f32(y % 4) / 4.;
    return cout;
}

fn quant(color: vec3<f32>) -> vec3<f32> {
    let c = rgb_to_oklab(color);
    let r = floor(c.r * 16.0) / 16.0;
    let g = floor(c.g * 64.0) / 64.0;
    let b = floor(c.b * 64.0) / 64.0;
    return oklab_to_rgb(vec3(r, g, b));
}

fn pixel_lut(color: vec3<f32>) -> vec3<f32> {
    let lut_uv = vec3(pow(color.r, 1. / 2.2), pow(color.g, 1. / 2.2), pow(color.b, 1. / 2.2));
    // let lut_uv = vec3(pow(color.r, 2.2), pow(color.g, 2.2), pow(color.b, 2.2));
    let lut_color = textureSample(lut, texture_sampler, lut_uv).rgb;
    return gamma_correct(lut_color.rbg);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel: vec2<f32> = get_texel_size();
    let unit: vec3<f32> = vec3<f32>(1.0, -1.0, 0.0);
    let uv: vec2<f32> = pixelate_uv(in.uv, texel);

    let depth: f32 = get_linear_depth(uv);
    let normal: vec3<f32> = get_ss_normal(uv);

    let raw_color: vec3<f32> = textureSample(screen_texture, texture_sampler, uv).rgb;
    var color: vec3<f32> = raw_color;
    var color_light: vec3<f32> = rgb_to_oklab(color);
    var color_dark: vec3<f32> = rgb_to_oklab(color);

    color_light.x = clamp(color_light.x + 0.1, 0.0, 1.0);
    color_dark.x = clamp(color_dark.x - 0.15, 0.0, 1.0);

    color_light = oklab_to_rgb(color_light);
    color_dark = oklab_to_rgb(color_dark);


    let du = get_linear_depth(uv + texel * unit.zy);
    let dd = get_linear_depth(uv + texel * unit.zx);
    let dl = get_linear_depth(uv + texel * unit.yz);
    let dr = get_linear_depth(uv + texel * unit.xz);

    var ddiff = 0.0;
    ddiff += depth - du;
    ddiff += depth - dd;
    ddiff += depth - dl;
    ddiff += depth - dr;

    let dedge = step(0.1, ddiff);



    let nu = get_ss_normal(uv + texel * unit.zy);
    let nd = get_ss_normal(uv + texel * unit.zx);
    let nl = get_ss_normal(uv + texel * unit.yz);
    let nr = get_ss_normal(uv + texel * unit.xz);

    let normal_bias = vec3(1.0, 1.0, 1.0);


    var dotsum: f32 = 0.0;

    let ndu = nu - normal;
    let ndd = nd - normal;
    let ndl = nl - normal;
    let ndr = nr - normal;

    let nbdu = dot(ndu, normal_bias);
    let nbdd = dot(ndd, normal_bias);
    let nbdl = dot(ndl, normal_bias);
    let nbdr = dot(ndr, normal_bias);

    let ni_u = smoothstep(-.01, .01, nbdu);
    let ni_d = smoothstep(-.01, .01, nbdd);
    let ni_l = smoothstep(-.01, .01, nbdl);
    let ni_r = smoothstep(-.01, .01, nbdr);

    dotsum += dot(ndu, ndu) * ni_u;
    dotsum += dot(ndd, ndd) * ni_d;
    dotsum += dot(ndl, ndl) * ni_l;
    dotsum += dot(ndr, ndr) * ni_r;

    let nei = sqrt(dotsum);
    let nedge = step(0.25, nei);

    if (ddiff > 0.) {
        if (dedge > 0.0) {
            color = lerpv(color, color_dark, dedge);
        } else if (dedge < 0.15) {
            color = lerpv(color, color_light, nedge);
        }
    }

    let outline = dedge + nedge * step(0., ddiff);

    // color = vec3(dedge, nedge, 0.0);
    // color.r = dedge;

    // color = lerpv(color, color_dark, dei);
    // color = lerpv(color, color_light, nedge);

    // let r_x = settings.resolution.x << 4;

    let pixel: vec2<i32> = vec2<i32>(i32(uv.x * settings.resolution.x / settings.pixel_scale), i32(uv.y * settings.resolution.y / settings.pixel_scale));
    if (outline < 1.0) {
    color = dither(color, uv);
    }
    // color = quant(color);
    color = pixel_lut(color);

    // color = vec3(outline);

    // let puv = uv * settings.resolution / settings.pixel_scale;
    // color.r = f32(pixel.x % 4) / 4.;
    // color.g = f32(pixel.y % 4) / 4.;

    return vec4<f32>(color, 1.0);
}
