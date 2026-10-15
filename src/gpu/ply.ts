import { load } from "@loaders.gl/core";
import { PLYLoader } from "@loaders.gl/ply";

export interface MeshData {
  positions: Float32Array;
  indices: Uint32Array;
  vertexCount: number;
  faceCount: number;
  minY: number;
  csrOffsets: Uint32Array;
  csrFaces: Uint32Array;
}

//  Fetches and reads ply source
export async function loadPLY(source: string | Blob): Promise<MeshData> {
  const data = await load(source, PLYLoader);

  const position = data.attributes?.POSITION?.value;
  if (!position) throw new Error("PLY has no POSITION attribute");
  const src =
    position instanceof Float32Array ? position : new Float32Array(position);

  const rawIndices = data.indices?.value;
  if (!rawIndices) throw new Error("PLY has no face indices");
  const indices =
    rawIndices instanceof Uint32Array
      ? rawIndices
      : new Uint32Array(rawIndices);

  const vertexCount = src.length / 3;

  // rescale mesh to fit 1.6unit box + find lowest Y
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < src.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = src[i + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  const scale = extent > 0 ? 1.6 / extent : 1;
  const minY = min[1] * scale; // lowest Y after rescale

  const positions = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    positions[i] = src[i] * scale;
    positions[i + 1] = src[i + 1] * scale;
    positions[i + 2] = src[i + 2] * scale;
  }

  // Build CSR vertex / faces adjacency
  const faceCount = indices.length / 3;
  const csrOffsets = new Uint32Array(vertexCount + 1);

  // this determines the size of each bucket
  for (let i = 0; i < indices.length; i++) {
    csrOffsets[indices[i] + 1]++;
  }

  // preparing the slots for the triangles that this vertex is touching basically the slot boundaries
  for (let v = 0; v < vertexCount; v++) csrOffsets[v + 1] += csrOffsets[v];

  // this step fills the reserved slots with the actual face ids
  const csrFaces = new Uint32Array(indices.length);
  const cursor = csrOffsets.slice(0, vertexCount);
  for (let f = 0; f < faceCount; f++) {
    for (let k = 0; k < 3; k++) {
      const vi = indices[f * 3 + k];
      csrFaces[cursor[vi]++] = f;
    }
  }

  // return mesh information
  return {
    positions,
    indices,
    vertexCount,
    faceCount,
    minY,
    csrOffsets,
    csrFaces,
  };
}
