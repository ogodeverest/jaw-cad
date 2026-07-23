struct Uniforms {
  mvp : mat4x4f,    // model-view-projection
  model : mat4x4f,  // world transform (positions + normals)
  base : vec4f,     // albedo rgb
  camPos : vec4f,   // camera world position (xyz)
  material : vec4f, // x: metallic, y: roughness
};

@group(0) @binding(0) var<uniform> u : Uniforms;

const PI = 3.14159265359;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) normal : vec3f,
  @location(1) worldPos : vec3f,
};

@vertex
fn vs(@location(0) p : vec3f, @location(1) n : vec3f) -> VSOut {
  var out : VSOut;
  let world = u.model * vec4f(p, 1.0);
  out.pos = u.mvp * vec4f(p, 1.0);
  out.worldPos = world.xyz;
  out.normal = (u.model * vec4f(n, 0.0)).xyz;
  return out;
}

// --- Cook-Torrance terms ---

// GGX / Trowbridge-Reitz normal distribution.
fn distributionGGX(nDotH : f32, roughness : f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Smith geometry with Schlick-GGX, direct-lighting k.
fn geometrySchlickGGX(nDotX : f32, roughness : f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return nDotX / (nDotX * (1.0 - k) + k);
}

fn geometrySmith(nDotV : f32, nDotL : f32, roughness : f32) -> f32 {
  return geometrySchlickGGX(nDotV, roughness) *
         geometrySchlickGGX(nDotL, roughness);
}

fn fresnelSchlick(cosTheta : f32, f0 : vec3f) -> vec3f {
  return f0 + (vec3f(1.0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Three analytic directional lights: key, fill, rim.
const LIGHT_DIRS = array<vec3f, 3>(
  vec3f(0.5, 0.8, 0.6),
  vec3f(-0.6, 0.3, -0.4),
  vec3f(0.1, -0.4, -0.9),
);
const LIGHT_COLORS = array<vec3f, 3>(
  vec3f(3.2, 3.1, 3.0),
  vec3f(0.9, 0.95, 1.1),
  vec3f(0.5, 0.45, 0.4),
);

@fragment
fn fs(in : VSOut) -> @location(0) vec4f {
  let n = normalize(in.normal);
  let v = normalize(u.camPos.xyz - in.worldPos);
  let nDotV = max(dot(n, v), 0.0001);

  let albedo = u.base.rgb;
  let metallic = u.material.x;
  let roughness = clamp(u.material.y, 0.04, 1.0);

  // Base reflectivity: 0.04 for dielectrics, albedo for metals.
  let f0 = mix(vec3f(0.04), albedo, metallic);

  var lo = vec3f(0.0);
  for (var i = 0; i < 3; i = i + 1) {
    let l = normalize(LIGHT_DIRS[i]);
    let h = normalize(v + l);
    let radiance = LIGHT_COLORS[i];

    let nDotL = max(dot(n, l), 0.0);
    let nDotH = max(dot(n, h), 0.0);
    let hDotV = max(dot(h, v), 0.0);

    let ndf = distributionGGX(nDotH, roughness);
    let g = geometrySmith(nDotV, nDotL, roughness);
    let f = fresnelSchlick(hDotV, f0);

    let specular = (ndf * g * f) / (4.0 * nDotV * nDotL + 0.0001);

    // Energy conservation: diffuse only from refracted (non-reflected),
    // and metals have no diffuse.
    let kd = (vec3f(1.0) - f) * (1.0 - metallic);

    lo += (kd * albedo / PI + specular) * radiance * nDotL;
  }

  let ambient = vec3f(0.03) * albedo;
  var color = ambient + lo;

  // Reinhard tone map + gamma.
  color = color / (color + vec3f(1.0));
  color = pow(color, vec3f(1.0 / 2.2));
  return vec4f(color, 1.0);
}
