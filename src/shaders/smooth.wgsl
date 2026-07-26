// Screen-space smoothing brush: one thread per vertex.
// Projects the vertex through the viewport's MVP, and if it lands inside the
// brush circle (framebuffer px, y-down), moves it toward the average of its
// neighbours (uniform Laplacian). Reads from posIn, writes posOut, so a single
// step never sees a half-updated neighbourhood; the caller copies posOut back.

struct Params {
  mvp : mat4x4f,
  center : vec2f,   // circle center, framebuffer px relative to the viewport
  viewport : vec2f, // viewport size in px (w, h)
  radius : f32,     // circle radius in px
  lambda : f32,     // smoothing strength, 0..1 per step
};

@group(0) @binding(0) var<storage, read> posIn : array<f32>;
@group(0) @binding(1) var<storage, read_write> posOut : array<f32>;
@group(0) @binding(2) var<storage, read> offsets : array<u32>;
@group(0) @binding(3) var<storage, read> faces : array<u32>;
@group(0) @binding(4) var<storage, read> indices : array<u32>;
@group(0) @binding(5) var<uniform> params : Params;

fn getPos(i : u32) -> vec3f {
  return vec3f(posIn[i * 3u], posIn[i * 3u + 1u], posIn[i * 3u + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x;
  let vertexCount = arrayLength(&posIn) / 3u;
  if (v >= vertexCount) {
    return;
  }

  let p = getPos(v);
  var outP = p;

  // Project to the viewport's screen space and test against the brush circle.
  let clip = params.mvp * vec4f(p, 1.0);
  if (clip.w > 0.0) {
    let ndc = clip.xy / clip.w;
    let px = (ndc.x * 0.5 + 0.5) * params.viewport.x;
    let py = (1.0 - (ndc.y * 0.5 + 0.5)) * params.viewport.y; // NDC +y is up
    if (distance(vec2f(px, py), params.center) <= params.radius) {
      // Uniform Laplacian: average the incident faces' other two vertices.
      var sum = vec3f(0.0);
      var count = 0u;
      let start = offsets[v];
      let end = offsets[v + 1u];
      for (var j = start; j < end; j = j + 1u) {
        let f = faces[j];
        for (var k = 0u; k < 3u; k = k + 1u) {
          let nb = indices[f * 3u + k];
          if (nb != v) {
            sum += getPos(nb);
            count += 1u;
          }
        }
      }
      if (count > 0u) {
        outP = mix(p, sum / f32(count), params.lambda);
      }
    }
  }

  posOut[v * 3u] = outP.x;
  posOut[v * 3u + 1u] = outP.y;
  posOut[v * 3u + 2u] = outP.z;
}
