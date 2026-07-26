// Flat ground grid.
//   Perspective: a world-space plane at y = gridY, fading into the bg with fog.
//   Orthographic (screen mode): camera-independent, mapped straight to the full
//   viewport at the far depth (z = 1) so it sits behind everything.

struct GridU {
  mvp : mat4x4f,
  cam : vec4f,      // xyz: camera world pos, w: grid Y height
  fogColor : vec4f, // rgb: background color
  fog : vec4f,      // x: fade start, y: fade end, z: screenMode flag
  screen : vec4f,   // xy: NDC scale per local unit (aspect-corrected)
};

@group(0) @binding(0) var<uniform> u : GridU;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) worldPos : vec3f,
};

@vertex
fn vs(@location(0) p : vec3f) -> VSOut {
  var o : VSOut;
  if (u.fog.z > 0.5) {
    // Screen space: fixed cell size (aspect-corrected), pinned to the far plane.
    // Extent overflows [-1,1] and clips, so it always fills the viewport.
    o.pos = vec4f(p.x * u.screen.x, p.z * u.screen.y, 1.0, 1.0);
    o.worldPos = vec3f(0.0, 0.0, 0.0);
  } else {
    o.pos = u.mvp * vec4f(p, 1.0);
    // The grid lies on y = gridY in world space (local y is 0).
    o.worldPos = vec3f(p.x, u.cam.w, p.z);
  }
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let base = vec4f(0.478, 0.635, 1.0, 0.35);
  if (u.fog.z > 0.5) {
    return base; // screen-space grid: camera-independent, no fog
  }
  let dist = distance(in.worldPos, u.cam.xyz);
  let f = clamp((dist - u.fog.x) / (u.fog.y - u.fog.x), 0.0, 1.0);
  // Tint toward the background and fade alpha out so far lines vanish into it.
  let rgb = mix(base.rgb, u.fogColor.rgb, f);
  let a = base.a * (1.0 - f);
  return vec4f(rgb, a);
}
