import { load } from '@loaders.gl/core'
import { PLYLoader } from '@loaders.gl/ply'

export interface MeshData {
  /** Rescaled positions (original location kept), 3 floats per vertex. */
  positions: Float32Array
  indices: Uint32Array
  vertexCount: number
  faceCount: number
  /** Lowest Y of the (rescaled) mesh — where the grid sits. */
  minY: number
  /** CSR vertex→faces adjacency: offsets has vertexCount+1 entries. */
  csrOffsets: Uint32Array
  /** Flattened incident face indices, length faceCount*3. */
  csrFaces: Uint32Array
}

/**
 * Fetches/reads + parses a PLY source (URL or uploaded File). Positions are
 * scaled to a ~1.6 unit box (keeping their original location relative to the
 * origin), and a CSR vertex→faces adjacency is built for the GPU normal passes.
 */
export async function loadPLY(source: string | Blob): Promise<MeshData> {
  const data = await load(source, PLYLoader)

  const position = data.attributes?.POSITION?.value
  if (!position) throw new Error('PLY has no POSITION attribute')
  const src =
    position instanceof Float32Array ? position : new Float32Array(position)

  const rawIndices = data.indices?.value
  if (!rawIndices) throw new Error('PLY has no face indices')
  const indices =
    rawIndices instanceof Uint32Array ? rawIndices : new Uint32Array(rawIndices)

  const vertexCount = src.length / 3

  // --- Rescale positions to fit a ~1.6 unit box (no recenter: keep original
  // position relative to the origin) ---
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < src.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = src[i + a]
      if (v < min[a]) min[a] = v
      if (v > max[a]) max[a] = v
    }
  }
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  const scale = extent > 0 ? 1.6 / extent : 1
  const minY = min[1] * scale // lowest Y after rescale

  const positions = new Float32Array(src.length)
  for (let i = 0; i < src.length; i += 3) {
    positions[i] = src[i] * scale
    positions[i + 1] = src[i + 1] * scale
    positions[i + 2] = src[i + 2] * scale
  }

  // --- Build CSR vertex→faces adjacency (used by the GPU normal passes) ---
  const faceCount = indices.length / 3
  const csrOffsets = new Uint32Array(vertexCount + 1)
  // Count incident faces per vertex into the next slot, then prefix-sum so
  // csrOffsets[v]..csrOffsets[v+1] is vertex v's slice.
  for (let i = 0; i < indices.length; i++) csrOffsets[indices[i] + 1]++
  for (let v = 0; v < vertexCount; v++) csrOffsets[v + 1] += csrOffsets[v]

  const csrFaces = new Uint32Array(indices.length) // faceCount * 3 incidences
  const cursor = csrOffsets.slice(0, vertexCount) // mutable per-vertex write heads
  for (let f = 0; f < faceCount; f++) {
    for (let k = 0; k < 3; k++) {
      const vi = indices[f * 3 + k]
      csrFaces[cursor[vi]++] = f
    }
  }

  return {
    positions,
    indices,
    vertexCount,
    faceCount,
    minY,
    csrOffsets,
    csrFaces,
  }
}

/**
 * CPU reference for area-weighted vertex normals. Superseded by the GPU compute
 * path (see `normals.ts`); kept for comparison / as a fallback.
 */
export function computeVertexNormalsCPU(
  positions: Float32Array,
  indices: Uint32Array,
): Float32Array {
  const normals = new Float32Array(positions.length)
  for (let f = 0; f < indices.length; f += 3) {
    const i0 = indices[f] * 3
    const i1 = indices[f + 1] * 3
    const i2 = indices[f + 2] * 3

    const e1x = positions[i1] - positions[i0]
    const e1y = positions[i1 + 1] - positions[i0 + 1]
    const e1z = positions[i1 + 2] - positions[i0 + 2]
    const e2x = positions[i2] - positions[i0]
    const e2y = positions[i2 + 1] - positions[i0 + 1]
    const e2z = positions[i2 + 2] - positions[i0 + 2]

    // Cross product; magnitude ∝ triangle area, so this weights by area.
    const nx = e1y * e2z - e1z * e2y
    const ny = e1z * e2x - e1x * e2z
    const nz = e1x * e2y - e1y * e2x

    for (const idx of [i0, i1, i2]) {
      normals[idx] += nx
      normals[idx + 1] += ny
      normals[idx + 2] += nz
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const len = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1
    normals[i] /= len
    normals[i + 1] /= len
    normals[i + 2] /= len
  }
  return normals
}
