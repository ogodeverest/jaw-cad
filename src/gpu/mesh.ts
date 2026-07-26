import { mat4 } from 'wgpu-matrix'
import meshShader from '../shaders/mesh.wgsl?raw'
import type { FrameContext } from './useWebGPU'
import { DEPTH_FORMAT, SAMPLE_COUNT } from './useWebGPU'
import {
  setOrbitCamera,
  type OrbitState,
  type Projection,
  type Viewport,
} from './camera'
import { buildModelMatrix } from './transform'
import { createNormalGenerator } from './normals'
import { createSmoother, type BrushStroke } from './smooth'
import { createGridRenderer, BG, type GridItem } from './grid'
import type { MeshData } from './ply'

// Public API re-exports so consumers keep importing from './gpu/mesh'.
export type { Projection, OrbitState, Viewport } from './camera'
export { DEFAULT_ORBIT } from './camera'
export type { GridItem } from './grid'
export type { MeshData } from './ply'
export type { BrushStroke } from './smooth'
export { loadPLY, computeVertexNormalsCPU } from './ply'

/** One mesh drawn into one viewport with a chosen projection. */
export interface SceneItem {
  mesh: GpuMesh
  viewport: Viewport
  projection: Projection
  orbit: OrbitState
  /** World-space translation applied to the model (e.g. to stack jaws). */
  offset: [number, number, number]
  /** Euler rotation in radians, applied X→Y→Z (Three.js "XYZ" order). */
  rotation: [number, number, number]
  /** Per-axis scale applied to the model. */
  scale: [number, number, number]
  /** Material base color (albedo), linear rgb in 0..1. */
  color: [number, number, number]
  metallic: number
  roughness: number
}

/** GPU-resident buffers for a single mesh. Positions + normals live in
 *  separate buffers so a compute pass can (re)write normals in place. */
export interface GpuMesh {
  positionBuffer: GPUBuffer
  normalBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
  destroy(): void
}

export interface Renderer {
  createMesh(data: MeshData): GpuMesh
  /** Clears the whole canvas, then draws each item into its viewport. */
  render(frame: FrameContext, items: SceneItem[], grids: GridItem[]): void
  /** viewProj * model for `item`, written into `out` (for screen-space picking). */
  mvpForItem(item: SceneItem, out: Float32Array): void
  /** Applies one screen-space smoothing stroke, then regenerates normals. */
  smooth(mesh: GpuMesh, stroke: BrushStroke): void
}

/** Persistent per-mesh buffers used by the smoothing + normal passes. */
interface MeshExtra {
  csrOffsetBuffer: GPUBuffer
  csrFaceBuffer: GPUBuffer
  posScratch: GPUBuffer
  faceCount: number
  vertexCount: number
}

const POS_STRIDE = 3 * 4 // vec3 positions, tightly packed
const NORMAL_STRIDE = 3 * 4 // vec3 normals, tightly packed
// Per-draw transform: two mat4x4f + camPos vec4f. Rewritten every frame.
const TRANSFORM_SIZE = (2 * 16 + 4) * 4
// Per-material: base vec4f + pbr vec4f. Rewritten only when the material changes.
const MATERIAL_SIZE = 2 * 4 * 4

interface MaterialSlot {
  buffer: GPUBuffer
  bindGroup: GPUBindGroup
  last: { color: readonly number[]; metallic: number; roughness: number } | null
}

/**
 * Builds the shared mesh pipeline once, then vends per-mesh GPU buffers and
 * draws the scene. Normal generation is delegated to `normals.ts` and the
 * ground grid to `grid.ts`; camera + model math live in `camera.ts` /
 * `transform.ts`.
 */
export function createRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
): Renderer {
  const module = device.createShaderModule({ code: meshShader })

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module,
      entryPoint: 'vs',
      // Positions and normals come from separate buffers so the normal buffer
      // can be (re)written by the compute passes independently.
      buffers: [
        {
          arrayStride: POS_STRIDE,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        },
        {
          arrayStride: NORMAL_STRIDE,
          attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
        },
      ],
    },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: {
      format: DEPTH_FORMAT,
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
    multisample: { count: SAMPLE_COUNT },
  })

  const normals = createNormalGenerator(device)
  const smoother = createSmoother(device)
  const grid = createGridRenderer(device, format)

  // Persistent buffers the smoothing/normal passes reuse, keyed by mesh.
  const extras = new WeakMap<GpuMesh, MeshExtra>()

  // Scratch matrices reused across items/frames.
  const proj = mat4.create()
  const view = mat4.create()
  const viewProj = mat4.create()
  const model = mat4.create()
  const mvp = mat4.create()
  const invView = mat4.create()
  const transformData = new Float32Array(36) // mvp(16) + model(16) + camPos(4)
  const materialData = new Float32Array(8) // base(4) + pbr(4)

  // Per-mesh material buffer + bind group (group 1), rewritten only on change.
  const materials = new WeakMap<GpuMesh, MaterialSlot>()

  const createMesh = (data: MeshData): GpuMesh => {
    const { positions, indices, vertexCount, faceCount, csrOffsets, csrFaces } =
      data

    const positionBuffer = device.createBuffer({
      size: positions.byteLength,
      usage:
        GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(positionBuffer, 0, positions)

    // Normals are produced by the compute passes; no initial upload.
    const normalBuffer = device.createBuffer({
      size: vertexCount * 3 * 4,
      usage:
        GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    const indexBuffer = device.createBuffer({
      size: indices.byteLength,
      usage:
        GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(indexBuffer, 0, indices)

    // Persistent CSR vertex→faces adjacency, shared by the normal + smoothing
    // passes (both need to re-run after positions change).
    const csrOffsetBuffer = device.createBuffer({
      size: csrOffsets.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(csrOffsetBuffer, 0, csrOffsets)
    const csrFaceBuffer = device.createBuffer({
      size: csrFaces.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(csrFaceBuffer, 0, csrFaces)

    // Scratch positions the smoothing pass writes before copying back.
    const posScratch = device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    })

    normals.generate({
      positionBuffer,
      indexBuffer,
      normalBuffer,
      csrOffsetBuffer,
      csrFaceBuffer,
      faceCount,
      vertexCount,
    })

    // Material buffer + bind group (group 1) live with the mesh; content is
    // written lazily on the first draw and only again when it changes.
    const materialBuffer = device.createBuffer({
      size: MATERIAL_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const materialBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: materialBuffer } }],
    })

    const mesh: GpuMesh = {
      positionBuffer,
      normalBuffer,
      indexBuffer,
      indexCount: indices.length,
      destroy() {
        positionBuffer.destroy()
        normalBuffer.destroy()
        indexBuffer.destroy()
        materialBuffer.destroy()
        csrOffsetBuffer.destroy()
        csrFaceBuffer.destroy()
        posScratch.destroy()
      },
    }
    materials.set(mesh, {
      buffer: materialBuffer,
      bindGroup: materialBindGroup,
      last: null,
    })
    extras.set(mesh, {
      csrOffsetBuffer,
      csrFaceBuffer,
      posScratch,
      faceCount,
      vertexCount,
    })
    return mesh
  }

  // One transform buffer + bind group per draw. A mesh can appear in several
  // viewports with different cameras in the same pass, so each draw needs its
  // own transform storage — a shared buffer would be clobbered by the last write.
  const transformSlots: { buffer: GPUBuffer; bindGroup: GPUBindGroup }[] = []
  const getTransformSlot = (i: number) => {
    if (!transformSlots[i]) {
      const buffer = device.createBuffer({
        size: TRANSFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer } }],
      })
      transformSlots[i] = { buffer, bindGroup }
    }
    return transformSlots[i]
  }

  const render = (
    frame: FrameContext,
    items: SceneItem[],
    grids: GridItem[],
  ) => {
    const { device, encoder, view: colorView, msaaView, depthView } = frame

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: msaaView,
          resolveTarget: colorView,
          clearValue: { r: BG.r, g: BG.g, b: BG.b, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    })

    items.forEach((item, i) => {
      const {
        mesh,
        viewport: vp,
        projection,
        color,
        orbit,
        offset,
        rotation,
        scale,
        metallic,
        roughness,
      } = item
      setOrbitCamera(proj, view, viewProj, projection, orbit, vp.width / vp.height)

      // Camera world position = translation of the inverse view matrix.
      mat4.inverse(view, invView)

      buildModelMatrix(model, offset, rotation, scale)
      mat4.multiply(viewProj, model, mvp)

      // Per-draw transform (group 0): rewritten every frame.
      transformData.set(mvp, 0)
      transformData.set(model, 16)
      transformData.set(invView.subarray(12, 15), 32) // camPos.xyz
      transformData[35] = 1
      const slot = getTransformSlot(i)
      device.queue.writeBuffer(slot.buffer, 0, transformData)

      // Per-material (group 1): rewrite only when it differs from last frame.
      const material = materials.get(mesh)!
      const last = material.last
      if (
        !last ||
        last.metallic !== metallic ||
        last.roughness !== roughness ||
        last.color[0] !== color[0] ||
        last.color[1] !== color[1] ||
        last.color[2] !== color[2]
      ) {
        materialData.set(color, 0) // base.rgb
        materialData[3] = 1
        materialData[4] = metallic // pbr.x
        materialData[5] = roughness // pbr.y
        device.queue.writeBuffer(material.buffer, 0, materialData)
        material.last = { color: [...color], metallic, roughness }
      }

      pass.setViewport(vp.x, vp.y, vp.width, vp.height, 0, 1)
      pass.setScissorRect(vp.x, vp.y, vp.width, vp.height)
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, slot.bindGroup)
      pass.setBindGroup(1, material.bindGroup)
      pass.setVertexBuffer(0, mesh.positionBuffer)
      pass.setVertexBuffer(1, mesh.normalBuffer)
      pass.setIndexBuffer(mesh.indexBuffer, 'uint32')
      pass.drawIndexed(mesh.indexCount)
    })

    // Grids drawn after the opaque meshes so alpha blends over them / the bg,
    // and depth-tested so the mesh occludes the grid where it's in front.
    grid.draw(pass, grids)

    pass.end()
  }

  // Recompute the same MVP `render` uses for an item — for screen-space picking
  // outside the render pass (e.g. the smoothing brush). Uses the shared scratch
  // matrices, safe because it runs between frames, not during `render`.
  const mvpForItem = (item: SceneItem, out: Float32Array) => {
    const vp = item.viewport
    setOrbitCamera(proj, view, viewProj, item.projection, item.orbit, vp.width / vp.height)
    buildModelMatrix(model, item.offset, item.rotation, item.scale)
    mat4.multiply(viewProj, model, out)
  }

  const smooth = (mesh: GpuMesh, stroke: BrushStroke) => {
    const ex = extras.get(mesh)
    if (!ex) return
    smoother.apply(
      {
        positionBuffer: mesh.positionBuffer,
        posScratch: ex.posScratch,
        csrOffsetBuffer: ex.csrOffsetBuffer,
        csrFaceBuffer: ex.csrFaceBuffer,
        indexBuffer: mesh.indexBuffer,
        vertexCount: ex.vertexCount,
      },
      stroke,
    )
    // Positions moved, so the shading normals are stale — regenerate them.
    normals.generate({
      positionBuffer: mesh.positionBuffer,
      indexBuffer: mesh.indexBuffer,
      normalBuffer: mesh.normalBuffer,
      csrOffsetBuffer: ex.csrOffsetBuffer,
      csrFaceBuffer: ex.csrFaceBuffer,
      faceCount: ex.faceCount,
      vertexCount: ex.vertexCount,
    })
  }

  return { createMesh, render, mvpForItem, smooth }
}
