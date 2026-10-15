import { mat4 } from "wgpu-matrix";
import meshShader from "../shaders/mesh.wgsl?raw";
import type { FrameContext } from "./useWebGPU";
import { DEPTH_FORMAT, SAMPLE_COUNT } from "./useWebGPU";
import {
  setOrbitCamera,
  type OrbitState,
  type Projection,
  type Viewport,
} from "./camera";
import { buildModelMatrix } from "./transform";
import { createNormalGenerator } from "./normals";
import { createSmoother, type BrushStroke } from "./smooth";
import { createGridRenderer, BG, type GridItem } from "./grid";
import type { MeshData } from "./ply";

export type { Projection, OrbitState, Viewport } from "./camera";
export { DEFAULT_ORBIT } from "./camera";
export type { GridItem } from "./grid";
export type { MeshData } from "./ply";
export type { BrushStroke } from "./smooth";
export { loadPLY } from "./ply";

export interface SceneItem {
  // mesh buffer data
  mesh: GpuMesh;
  // viewport for the item x,y offset + width / height
  viewport: Viewport;
  projection: Projection;
  orbit: OrbitState;
  // translation applied
  offset: [number, number, number];
  // Euler rotation in radians
  rotation: [number, number, number];
  // per axis scale
  scale: [number, number, number];
  // Material base color
  color: [number, number, number];
  // physically based rendering properties
  metallic: number;
  roughness: number;
}

// GPU buffers, normals and positions are separate buffers because normal compute generation rewrites normalBuffer independently of positions
export interface GpuMesh {
  positionBuffer: GPUBuffer;
  normalBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  destroy(): void;
}

// the renderers public api: build meshes once, draw the scene each frame, and smooth a mesh
export interface Renderer {
  createMesh(data: MeshData): GpuMesh;
  render(frame: FrameContext, items: SceneItem[], grids: GridItem[]): void;
  mvpForItem(item: SceneItem, out: Float32Array): void;
  smooth(mesh: GpuMesh, stroke: BrushStroke): void;
}

// mesh buffers used for compute normals + brush
interface ComputeSlot {
  csrOffsetBuffer: GPUBuffer;
  csrFaceBuffer: GPUBuffer;
  posScratch: GPUBuffer;
  faceCount: number;
  vertexCount: number;
}

const POS_STRIDE = 3 * 4; // vec3 positions * 4bytes per component
const NORMAL_STRIDE = 3 * 4; // vec3 normals * 4bytes per component
// per-draw transform: two mat4x4f + camPos vec4f
const TRANSFORM_SIZE = (2 * 16 + 4) * 4;
// per-material: base vec4f + pbr vec4f. rewritten only when the material changes.
const MATERIAL_SIZE = 2 * 4 * 4;

// per mesh material slot (buffer + bindgroup)
interface MaterialSlot {
  buffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  last: {
    color: readonly number[];
    metallic: number;
    roughness: number;
  } | null;
}

// builds the shared mesh pipeline once, then hands out per-mesh buffers and draws the scene
// normal generation lives in normals.ts, grid drawing in grid.ts, camera + model math in camera.ts / transform.ts
export function createRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
): Renderer {
  // shader module with physically based rendering
  const module = device.createShaderModule({ code: meshShader });

  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs",
      // Positions and normals come from separate buffers so the normal buffer can be (re)written by the compute passes separately.
      buffers: [
        {
          arrayStride: POS_STRIDE,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
        {
          arrayStride: NORMAL_STRIDE,
          attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
        },
      ],
    },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list", cullMode: "none" },
    // specify the depth operations (write + compare) + depth texture format
    depthStencil: {
      format: DEPTH_FORMAT,
      depthWriteEnabled: true,
      depthCompare: "less",
    },
    // specify multisample count (4)
    multisample: { count: SAMPLE_COUNT },
  });

  // callback for normal calculations (compute)
  const normals = createNormalGenerator(device);
  // callback for smoothing (compute)
  const smoother = createSmoother(device);
  // callback for drawing the grid (render)
  const grid = createGridRenderer(device, format);

  // Persistent buffers the smoothing/normal passes reuse, keyed by mesh.
  const compute = new WeakMap<GpuMesh, ComputeSlot>();

  // 4x4 matrices reused across views/frames.
  const proj = mat4.create();
  const view = mat4.create();
  const viewProj = mat4.create();
  const model = mat4.create();
  const mvp = mat4.create();
  const invView = mat4.create();
  const transformData = new Float32Array(36); // mvp(16) + model(16) + camPos(4)
  const materialData = new Float32Array(8); // base(4) + pbr(4)

  // Per-mesh material buffer + bind group (group 1) written only on change.
  const materials = new WeakMap<GpuMesh, MaterialSlot>();

  // creates gpu mesh buffers
  const createMesh = (data: MeshData): GpuMesh => {
    const { positions, indices, vertexCount, faceCount, csrOffsets, csrFaces } =
      data;

    // positions buffer: VERTEX -> used as vertex buffer,
    // STORAGE -> used as storage buffer for read/write (smoothing step)
    // COPY_DST -> allows to copy data into
    // read-only during the smoothing pass itself, posScratch is what gets written
    const positionBuffer = device.createBuffer({
      size: positions.byteLength,
      usage:
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST,
    });
    // guaranteed to have data copied to the buffer before the pass begins
    // 2-stage copy: CPU -> Host Visible Buffer -> VRAM Buffer (positionBuffer)
    device.queue.writeBuffer(positionBuffer, 0, positions);

    // same for normal buffer but no initial write
    const normalBuffer = device.createBuffer({
      size: vertexCount * 3 * 4,
      usage:
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST,
    });

    // same for index buffer but we write initially the indices data via writeBuffer (2-stage copy)
    const indexBuffer = device.createBuffer({
      size: indices.byteLength,
      usage:
        GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, indices);

    // persistent CSR vertex - faces adjacency, shared by the normal + smoothing
    const csrOffsetBuffer = device.createBuffer({
      size: csrOffsets.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(csrOffsetBuffer, 0, csrOffsets);

    const csrFaceBuffer = device.createBuffer({
      size: csrFaces.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(csrFaceBuffer, 0, csrFaces);

    // smoothing pass writes before copying back (modified each pass)
    const posScratch = device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // run compute shader to generate normals
    normals.generate({
      positionBuffer,
      indexBuffer,
      normalBuffer,
      csrOffsetBuffer,
      csrFaceBuffer,
      faceCount,
      vertexCount,
    });

    // Material buffer + bind group (group 1), content is written lazily on the first draw and only again when it changes
    const materialBuffer = device.createBuffer({
      size: MATERIAL_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const materialBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: materialBuffer } }],
    });

    const mesh: GpuMesh = {
      positionBuffer,
      normalBuffer,
      indexBuffer,
      indexCount: indices.length,
      destroy() {
        positionBuffer.destroy();
        normalBuffer.destroy();
        indexBuffer.destroy();
        materialBuffer.destroy();
        csrOffsetBuffer.destroy();
        csrFaceBuffer.destroy();
        posScratch.destroy();
      },
    };
    materials.set(mesh, {
      buffer: materialBuffer,
      bindGroup: materialBindGroup,
      last: null,
    });
    compute.set(mesh, {
      csrOffsetBuffer,
      csrFaceBuffer,
      posScratch,
      faceCount,
      vertexCount,
    });

    return mesh;
  };

  // One transform buffer + bind group per draw
  const transformSlots: { buffer: GPUBuffer; bindGroup: GPUBindGroup }[] = [];

  const getTransformSlot = (i: number) => {
    if (!transformSlots[i]) {
      const buffer = device.createBuffer({
        size: TRANSFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer } }],
      });
      transformSlots[i] = { buffer, bindGroup };
    }
    return transformSlots[i];
  };

  const render = (
    frame: FrameContext,
    items: SceneItem[],
    grids: GridItem[],
  ) => {
    // get per frame context objects
    const { device, encoder, view: colorView, msaaView, depthView } = frame;

    // start the shared render pass (meshes, grids)
    const pass = encoder.beginRenderPass({
      // color attachments options
      colorAttachments: [
        {
          view: msaaView,
          resolveTarget: colorView,
          clearValue: { r: BG.r, g: BG.g, b: BG.b, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      // depth texture attachments options
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    });

    // item = (jaw, view)
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
      } = item;

      // create camera transform matrix
      setOrbitCamera(
        proj,
        view,
        viewProj,
        projection,
        orbit,
        vp.width / vp.height,
      );

      // Inverse of camera matrix
      mat4.inverse(view, invView);

      // create model world space transform
      buildModelMatrix(model, offset, rotation, scale);
      // create MVP matrix
      mat4.multiply(viewProj, model, mvp);

      // set transformData view (typed array) data
      transformData.set(mvp, 0);
      transformData.set(model, 16);
      // extract only camera position from this
      transformData.set(invView.subarray(12, 15), 32); // camPos.xyz
      transformData[35] = 1;
      // get the slot and write to its buffer
      const slot = getTransformSlot(i);
      device.queue.writeBuffer(slot.buffer, 0, transformData);

      // Per-material (group 1): rewrite only when it differs from last frame
      const material = materials.get(mesh)!;
      const last = material.last;
      if (
        !last ||
        last.metallic !== metallic ||
        last.roughness !== roughness ||
        last.color[0] !== color[0] ||
        last.color[1] !== color[1] ||
        last.color[2] !== color[2]
      ) {
        materialData.set(color, 0); // base.rgb
        materialData[3] = 1;
        materialData[4] = metallic; // pbr.x
        materialData[5] = roughness; // pbr.y
        device.queue.writeBuffer(material.buffer, 0, materialData);
        material.last = { color: [...color], metallic, roughness };
      }

      // set viewport (map from NDC to this viewport (window) coordinates)
      pass.setViewport(vp.x, vp.y, vp.width, vp.height, 0, 1);
      // set scissor rectangle (discard fragments outside this rectangle)
      pass.setScissorRect(vp.x, vp.y, vp.width, vp.height);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, slot.bindGroup);
      pass.setBindGroup(1, material.bindGroup);
      // set normal and position buffers
      pass.setVertexBuffer(0, mesh.positionBuffer);
      pass.setVertexBuffer(1, mesh.normalBuffer);
      //set index buffer
      pass.setIndexBuffer(mesh.indexBuffer, "uint32");
      // draw indexed
      pass.drawIndexed(mesh.indexCount);
    });

    // draw grids, transparent always draw after opaques for correct blending (simple case here)
    grid.draw(pass, grids);

    pass.end();
  };

  // closure to construct MVP again here for the smoothing screen-space
  const mvpForItem = (item: SceneItem, out: Float32Array) => {
    const vp = item.viewport;
    setOrbitCamera(
      proj,
      view,
      viewProj,
      item.projection,
      item.orbit,
      vp.width / vp.height,
    );
    buildModelMatrix(model, item.offset, item.rotation, item.scale);
    mat4.multiply(viewProj, model, out);
  };

  // closure for applying the smoothing
  const smooth = (mesh: GpuMesh, stroke: BrushStroke) => {
    const ex = compute.get(mesh);
    if (!ex) return;
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
    );
    // normals need to regenerate
    normals.generate({
      positionBuffer: mesh.positionBuffer,
      indexBuffer: mesh.indexBuffer,
      normalBuffer: mesh.normalBuffer,
      csrOffsetBuffer: ex.csrOffsetBuffer,
      csrFaceBuffer: ex.csrFaceBuffer,
      faceCount: ex.faceCount,
      vertexCount: ex.vertexCount,
    });
  };

  return { createMesh, render, mvpForItem, smooth };
}
