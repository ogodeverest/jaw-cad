import smoothShader from '../shaders/smooth.wgsl?raw'

const WG_SIZE = 64 // must match @workgroup_size in smooth.wgsl
const PARAMS_FLOATS = 24 // mat4(16) + center(2) + viewport(2) + radius + lambda + pad(2)

// one screen-space brush stroke against a single mesh in one viewport
export interface BrushStroke {
  // viewProj * model for the mesh in the target viewport
  mvp: Float32Array
  // target viewport size in framebuffer px
  viewportW: number
  viewportH: number
  // circle center in framebuffer px, relative to the viewport origin (y-down)
  centerX: number
  centerY: number
  // circle radius in framebuffer px
  radius: number
  // smoothing strength per step, 0..1
  lambda: number
}

// buffers the smoothing pass reads from / writes to for one mesh
export interface SmoothTargets {
  // live vertex positions (read as input, overwritten with the result)
  positionBuffer: GPUBuffer
  // scratch position buffer (STORAGE | COPY_SRC), same size as positions
  posScratch: GPUBuffer
  // CSR vertex-faces adjacency (persistent gpu buffers)
  csrOffsetBuffer: GPUBuffer
  csrFaceBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  vertexCount: number
}

export interface Smoother {
  apply(targets: SmoothTargets, stroke: BrushStroke): void
}

// builds the smoothing compute pipeline once, then hands back apply()
// apply() projects every vertex through the given mvp, tests it against the brush
// circle in screen space, and if inside, moves it toward the average of its
// neighbours (one laplacian step) - all done in smooth.wgsl in a single dispatch
// results go to a scratch buffer first (so threads never read each other's
// half-updated positions), then get copied back over positionBuffer
// normals are stale after this, caller has to regenerate them
export function createSmoother(device: GPUDevice): Smoother {
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: smoothShader }),
      entryPoint: 'main',
    },
  })

  // one reusable uniform buffer + CPU staging array, strokes submit in order
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
    // copy the smoothed positions back over the live buffer
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
