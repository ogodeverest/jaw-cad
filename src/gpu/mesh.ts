import { load } from '@loaders.gl/core'
import { PLYLoader } from '@loaders.gl/ply'
import { mat4 } from 'wgpu-matrix'
import meshShader from '../shaders/mesh.wgsl?raw'
import type { FrameContext } from './useWebGPU'
import { DEPTH_FORMAT, SAMPLE_COUNT } from './useWebGPU'

export type Projection = 'perspective' | 'orthographic'

/** Orbit camera around the origin, in spherical-ish terms. */
export interface OrbitState {
  azimuth: number // radians, around +Y
  polar: number // radians, tilt up/down (clamp to avoid gimbal flip)
  radius: number // distance from target
}

export const DEFAULT_ORBIT: OrbitState = {
  azimuth: 0.6,
  polar: 0.5,
  radius: 4.5,
}

export interface MeshData {
  /** Interleaved position(xyz) + normal(xyz), 6 floats per vertex. */
  vertices: Float32Array
  indices: Uint32Array
  vertexCount: number
}

/** A framebuffer-pixel rectangle, origin at the top-left of the canvas. */
export interface Viewport {
  x: number
  y: number
  width: number
  height: number
}

/** One mesh drawn into one viewport with a chosen projection. */
export interface SceneItem {
  mesh: GpuMesh
  viewport: Viewport
  projection: Projection
  orbit: OrbitState
  /** World-space translation applied to the model (e.g. to stack jaws). */
  offset: [number, number, number]
  /** Material base color (albedo), linear rgb in 0..1. */
  color: [number, number, number]
  metallic: number
  roughness: number
}

/** GPU-resident buffers for a single mesh. */
export interface GpuMesh {
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  indexCount: number
  destroy(): void
}

export interface Renderer {
  createMesh(data: MeshData): GpuMesh
  /** Clears the whole canvas, then draws each item into its viewport. */
  render(frame: FrameContext, items: SceneItem[]): void
}

const STRIDE = 6 * 4 // 6 floats per vertex
const UNIFORM_SIZE = (2 * 16 + 3 * 4) * 4 // two mat4x4f + three vec4f

/**
 * Fetches/reads + parses a PLY source (URL or uploaded File) into an
 * interleaved, render-ready mesh. Positions are recentered on the origin and
 * scaled to a unit-ish size, and per-vertex normals are computed
 * (area-weighted) since the source files have none.
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

  // --- Recenter + rescale positions to fit a ~1.6 unit box ---
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (let i = 0; i < src.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = src[i + a]
      if (v < min[a]) min[a] = v
      if (v > max[a]) max[a] = v
    }
  }
  const center = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ]
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  const scale = extent > 0 ? 1.6 / extent : 1

  const positions = new Float32Array(src.length)
  for (let i = 0; i < src.length; i += 3) {
    positions[i] = (src[i] - center[0]) * scale
    positions[i + 1] = (src[i + 1] - center[1]) * scale
    positions[i + 2] = (src[i + 2] - center[2]) * scale
  }

  // --- Area-weighted vertex normals from face normals ---
  const normals = new Float32Array(src.length)
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

  // --- Interleave ---
  const vertices = new Float32Array(vertexCount * 6)
  for (let v = 0; v < vertexCount; v++) {
    vertices[v * 6] = positions[v * 3]
    vertices[v * 6 + 1] = positions[v * 3 + 1]
    vertices[v * 6 + 2] = positions[v * 3 + 2]
    vertices[v * 6 + 3] = normals[v * 3]
    vertices[v * 6 + 4] = normals[v * 3 + 1]
    vertices[v * 6 + 5] = normals[v * 3 + 2]
  }

  return { vertices, indices, vertexCount }
}

/**
 * Builds the shared pipeline once, then vends per-mesh GPU buffers. All meshes
 * share the vertex layout + shader; each gets its own uniform buffer so it can
 * carry its own camera for its viewport.
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
      buffers: [
        {
          arrayStride: STRIDE,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
            { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' }, // normal
          ],
        },
      ],
    },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list', cullMode: 'back' },
    depthStencil: {
      format: DEPTH_FORMAT,
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
    multisample: { count: SAMPLE_COUNT },
  })

  // Scratch matrices reused across items/frames.
  const proj = mat4.create()
  const view = mat4.create()
  const viewProj = mat4.create()
  const model = mat4.create()
  const mvp = mat4.create()
  const invView = mat4.create()
  // 2 mat4 (32) + base(4) + camPos(4) + material(4)
  const uniformData = new Float32Array(44)

  const createMesh = (data: MeshData): GpuMesh => {
    const vertexBuffer = device.createBuffer({
      size: data.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(vertexBuffer, 0, data.vertices)

    const indexBuffer = device.createBuffer({
      size: data.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(indexBuffer, 0, data.indices)

    return {
      vertexBuffer,
      indexBuffer,
      indexCount: data.indices.length,
      destroy() {
        vertexBuffer.destroy()
        indexBuffer.destroy()
      },
    }
  }

  // One uniform buffer + bind group per draw. A mesh can appear in several
  // viewports with different cameras in the same pass, so each draw needs its
  // own uniform storage — a shared buffer would be clobbered by the last write.
  const uniformSlots: { buffer: GPUBuffer; bindGroup: GPUBindGroup }[] = []
  const getUniformSlot = (i: number) => {
    if (!uniformSlots[i]) {
      const buffer = device.createBuffer({
        size: UNIFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer } }],
      })
      uniformSlots[i] = { buffer, bindGroup }
    }
    return uniformSlots[i]
  }

  const render = (frame: FrameContext, items: SceneItem[]) => {
    const { device, encoder, view: colorView, msaaView, depthView } = frame

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: msaaView,
          resolveTarget: colorView,
          clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 },
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
        metallic,
        roughness,
      } = item
      const aspect = vp.width / vp.height
      if (projection === 'perspective') {
        mat4.perspective((50 * Math.PI) / 180, aspect, 0.1, 100, proj)
      } else {
        // Tie the ortho box to radius so the wheel zooms in both modes.
        const halfH = orbit.radius * 0.31
        const halfW = halfH * aspect
        mat4.ortho(-halfW, halfW, -halfH, halfH, 0.1, 100, proj)
      }

      // Orbit view built from rotation matrices: pull back by radius, then
      // tilt (polar, about X) and spin (azimuth, about Y) around the origin.
      mat4.identity(view)
      mat4.translate(view, [0, 0, -orbit.radius], view)
      mat4.rotateX(view, orbit.polar, view)
      mat4.rotateY(view, orbit.azimuth, view)
      mat4.multiply(proj, view, viewProj)

      // Camera world position = translation of the inverse view matrix.
      mat4.inverse(view, invView)

      mat4.identity(model)
      mat4.translate(model, offset, model)
      mat4.multiply(viewProj, model, mvp)
      uniformData.set(mvp, 0)
      uniformData.set(model, 16)
      uniformData[32] = color[0] // base.rgb
      uniformData[33] = color[1]
      uniformData[34] = color[2]
      uniformData[35] = 1
      uniformData[36] = invView[12] // camPos.xyz
      uniformData[37] = invView[13]
      uniformData[38] = invView[14]
      uniformData[39] = 1
      uniformData[40] = metallic // material.x
      uniformData[41] = roughness // material.y
      uniformData[42] = 0
      uniformData[43] = 0

      // Per-draw uniform slot so multi-viewport draws don't clobber each other.
      const slot = getUniformSlot(i)
      device.queue.writeBuffer(slot.buffer, 0, uniformData)

      pass.setViewport(vp.x, vp.y, vp.width, vp.height, 0, 1)
      pass.setScissorRect(vp.x, vp.y, vp.width, vp.height)
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, slot.bindGroup)
      pass.setVertexBuffer(0, mesh.vertexBuffer)
      pass.setIndexBuffer(mesh.indexBuffer, 'uint32')
      pass.drawIndexed(mesh.indexCount)
    })

    pass.end()
  }

  return { createMesh, render }
}
