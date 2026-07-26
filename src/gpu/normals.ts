import faceNormalsShader from '../shaders/faceNormals.wgsl?raw'
import vertexNormalsShader from '../shaders/vertexNormals.wgsl?raw'

const WG_SIZE = 64 // must match @workgroup_size in the normal shaders

/** Buffers the normal passes read from and the target they write to. */
export interface NormalTargets {
  positionBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  /** Destination — filled with per-vertex normals (3 floats per vertex). */
  normalBuffer: GPUBuffer
  /** Persistent CSR vertex→faces adjacency (owned by the caller). */
  csrOffsetBuffer: GPUBuffer
  csrFaceBuffer: GPUBuffer
  faceCount: number
  vertexCount: number
}

export interface NormalGenerator {
  /** Runs the two-pass CSR gather to (re)compute `normalBuffer` on the GPU. */
  generate(targets: NormalTargets): void
}

/**
 * Builds the face- and vertex-normal compute pipelines once, then vends a
 * `generate` that computes area-weighted vertex normals for a mesh:
 *   pass 1 (per triangle) → un-normalized face normals,
 *   pass 2 (per vertex)   → gather incident faces via CSR, sum, normalize.
 */
export function createNormalGenerator(device: GPUDevice): NormalGenerator {
  const facePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: faceNormalsShader }),
      entryPoint: 'main',
    },
  })
  const vertexPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: vertexNormalsShader }),
      entryPoint: 'main',
    },
  })

  const generate = (t: NormalTargets) => {
    // Face normals are scratch (freed after the two passes); the CSR adjacency
    // buffers are persistent and owned by the caller.
    const faceNormalBuffer = device.createBuffer({
      size: t.faceCount * 3 * 4,
      usage: GPUBufferUsage.STORAGE,
    })

    const faceBind = device.createBindGroup({
      layout: facePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: t.positionBuffer } },
        { binding: 1, resource: { buffer: t.indexBuffer } },
        { binding: 2, resource: { buffer: faceNormalBuffer } },
      ],
    })
    const vertBind = device.createBindGroup({
      layout: vertexPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: faceNormalBuffer } },
        { binding: 1, resource: { buffer: t.csrOffsetBuffer } },
        { binding: 2, resource: { buffer: t.csrFaceBuffer } },
        { binding: 3, resource: { buffer: t.normalBuffer } },
      ],
    })

    // Two separate passes so pass 1's writes are visible to pass 2.
    const encoder = device.createCommandEncoder()
    const p1 = encoder.beginComputePass()
    p1.setPipeline(facePipeline)
    p1.setBindGroup(0, faceBind)
    p1.dispatchWorkgroups(Math.ceil(t.faceCount / WG_SIZE))
    p1.end()
    const p2 = encoder.beginComputePass()
    p2.setPipeline(vertexPipeline)
    p2.setBindGroup(0, vertBind)
    p2.dispatchWorkgroups(Math.ceil(t.vertexCount / WG_SIZE))
    p2.end()
    device.queue.submit([encoder.finish()])

    // Destroy is deferred until the GPU is done with the submitted work.
    faceNormalBuffer.destroy()
  }

  return { generate }
}
