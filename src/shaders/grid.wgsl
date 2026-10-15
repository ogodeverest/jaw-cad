struct GridU {
  mvp : mat4x4f,
  cam : vec4f,
  fogColor : vec4f,
  fogMode : vec4f,
  screen : vec4f,
};

@group(0) @binding(0) var<uniform> u : GridU;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) worldPos : vec3f,
};

@vertex
fn vs(@location(0) p : vec3f) -> VSOut {
  var o : VSOut;
  if (u.fogMode.z > 0.5) {
    o.pos = vec4f(p.x * u.screen.x, p.z * u.screen.y, 1.0, 1.0);
    o.worldPos = vec3f(0.0, 0.0, 0.0);
  } else {
    o.pos = u.mvp * vec4f(p, 1.0);
    o.worldPos = vec3f(p.x, u.cam.w, p.z);
  }
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let base = vec4f(0.478, 0.635, 1.0, 0.35);
  if (u.fogMode.z > 0.5) {
    return base;
  }
  let dist = distance(in.worldPos, u.cam.xyz);
  let f = clamp((dist - u.fogMode.x) / (u.fogMode.y - u.fogMode.x), 0.0, 1.0);
  let rgb = mix(base.rgb, u.fogColor.rgb, f);
  let a = base.a * (1.0 - f);
  return vec4f(rgb, a);
}
