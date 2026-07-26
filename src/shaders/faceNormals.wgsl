// Pass 1 of GPU normal generation: one thread per triangle.
// Writes an un-normalized face normal (length ∝ area) so pass 2's sum is
// area-weighted, matching the CPU reference.

@group(0) @binding(0) var<storage, read> positions : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<u32>;
@group(0) @binding(2) var<storage, read_write> faceNormals : array<f32>;

fn getPos(i : u32) -> vec3f {
  return vec3f(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let f = gid.x;
  let faceCount = arrayLength(&indices) / 3u;
  if (f >= faceCount) {
    return;
  }

  let p0 = getPos(indices[f * 3u]);
  let p1 = getPos(indices[f * 3u + 1u]);
  let p2 = getPos(indices[f * 3u + 2u]);

  let n = cross(p1 - p0, p2 - p0);

  faceNormals[f * 3u] = n.x;
  faceNormals[f * 3u + 1u] = n.y;
  faceNormals[f * 3u + 2u] = n.z;
}
