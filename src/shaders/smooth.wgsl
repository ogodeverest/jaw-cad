struct Params {
  mvp : mat4x4f,
  center : vec2f,
  viewport : vec2f,
  radius : f32,
  lambda : f32,
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

  let clip = params.mvp * vec4f(p, 1.0);
  if (clip.w > 0.0) {
    let ndc = clip.xy / clip.w;
    let px = (ndc.x * 0.5 + 0.5) * params.viewport.x;
    let py = (1.0 - (ndc.y * 0.5 + 0.5)) * params.viewport.y;
    if (distance(vec2f(px, py), params.center) <= params.radius) {
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
