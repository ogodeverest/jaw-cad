import smoothShader from '../shaders/smooth.wgsl?raw'

const WG_SIZE = 64 // must match @workgroup_size in smooth.wgsl
const PARAMS_FLOATS = 24 // mat4(16) + center(2) + viewport(2) + radius + lambda + pad(2)

/** One screen-space brush stroke against a single mesh in one viewport. */
export interface BrushStroke {
  /** viewProj * model for the mesh in the target viewport. */
  mvp: Float32Array
  /** Target viewport size in framebuffer px. */
  viewportW: number
  viewportH: number
  /** Circle center in framebuffer px, relative to the viewport origin (y-down). */
  centerX: number
  centerY: number
  /** Circle radius in framebuffer px. */
  radius: number
  /** Smoothing strength per step, 0..1. */
  lambda: number
}

/** Buffers the smoothing pass reads from / writes to for one mesh. */
export interface SmoothTargets {
  /** Live vertex positions (read as input, overwritten with the result). */
  positionBuffer: GPUBuffer
  /** Scratch position buffer (STORAGE | COPY_SRC), same size as positions. */
  posScratch: GPUBuffer
  /** CSR vertex→faces adjacency (persistent GPU buffers). */
  csrOffsetBuffer: GPUBuffer
  csrFaceBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  vertexCount: number
}

export interface Smoother {
  apply(targets: SmoothTargets, stroke: BrushStroke): void
}

/**
 * Builds the smoothing compute pipeline once, then vends `apply`, which runs one
 * Laplacian step over the vertices under the brush circle and copies the result
 * back into `positionBuffer`. Callers should re-run normal generation after.
 */
export function createSmoother(device: GPUDevice): Smoother {
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: smoothShader }),
      entryPoint: 'main',
    },
  })

  // One reusable uniform buffer + CPU staging array; strokes submit in order.
  const paramsBuffer = device.createBuffer({
    size: PARAMS_FLOATS * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const params = new Float32Array(PARAMS_FLOATS)

  const apply = (t: SmoothTargets, s: BrushStroke) => {
    params.set(s.mvp, 0)
    params[16] = s.centerX
    params[17] = s.centerY
    params[18] = s.viewportW
    params[19] = s.viewportH
    params[20] = s.radius
    params[21] = s.lambda
    device.queue.writeBuffer(paramsBuffer, 0, params)

    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: t.positionBuffer } },
        { binding: 1, resource: { buffer: t.posScratch } },
        { binding: 2, resource: { buffer: t.csrOffsetBuffer } },
        { binding: 3, resource: { buffer: t.csrFaceBuffer } },
        { binding: 4, resource: { buffer: t.indexBuffer } },
        { binding: 5, resource: { buffer: paramsBuffer } },
      ],
    })

    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bind)
    pass.dispatchWorkgroups(Math.ceil(t.vertexCount / WG_SIZE))
    pass.end()
    // Copy the smoothed positions back over the live buffer.
    encoder.copyBufferToBuffer(
      t.posScratch,
      0,
      t.positionBuffer,
      0,
      t.vertexCount * 3 * 4,
    )
    device.queue.submit([encoder.finish()])
  }

  return { apply }
}
