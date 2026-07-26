// Pass 2 of GPU normal generation: one thread per vertex.
// Gathers the vertex's incident face normals via CSR adjacency
// (offsets + faces), sums them, and normalizes — no atomics needed.

@group(0) @binding(0) var<storage, read> faceNormals : array<f32>;
@group(0) @binding(1) var<storage, read> offsets : array<u32>;
@group(0) @binding(2) var<storage, read> faces : array<u32>;
@group(0) @binding(3) var<storage, read_write> normals : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let v = gid.x;
  let vertexCount = arrayLength(&offsets) - 1u;
  if (v >= vertexCount) {
    return;
  }

  var sum = vec3f(0.0, 0.0, 0.0);
  let start = offsets[v];
  let end = offsets[v + 1u];
  for (var j = start; j < end; j = j + 1u) {
    let f = faces[j];
    sum += vec3f(faceNormals[f * 3u], faceNormals[f * 3u + 1u], faceNormals[f * 3u + 2u]);
  }

  let len = length(sum);
  var n = vec3f(0.0, 0.0, 1.0);
  if (len > 0.0) {
    n = sum / len;
  }

  normals[v * 3u] = n.x;
  normals[v * 3u + 1u] = n.y;
  normals[v * 3u + 2u] = n.z;
}
