import { mat4 } from "wgpu-matrix";
import gridShader from "../shaders/grid.wgsl?raw";
import { DEPTH_FORMAT, SAMPLE_COUNT } from "./useWebGPU";
import {
  setOrbitCamera,
  type OrbitState,
  type Projection,
  type Viewport,
} from "./camera";

// background / clear color, shared with the grid fog so far lines fade into it
export const BG = { r: 0.05, g: 0.06, b: 0.08 };

const GRID_HALF = 12; // grid spans [-GRID_HALF, GRID_HALF] on X and Z
const GRID_DIVISIONS = 96; // keeps ~0.25-unit cells at this larger extent
const GRID_STEP = (2 * GRID_HALF) / GRID_DIVISIONS; // world size of one cell
const GRID_SCREEN_CELL = 0.13; // ortho screen-space cell height, in NDC-Y
const POS_STRIDE = 3 * 4;
const GRID_UNIFORM_FLOATS = 32; // mvp(16)+cam(4)+fogColor(4)+fogMode(4)+screen(4)

// one ground grid drawn in one viewport, at height y
export interface GridItem {
  viewport: Viewport;
  projection: Projection;
  orbit: OrbitState;
  y: number;
}

export interface GridRenderer {
  // draws each grid into its viewport, call inside an already-open render pass
  draw(pass: GPURenderPassEncoder, grids: GridItem[]): void;
}

// builds the line-list vertices for a square grid on the XZ plane (y = 0)
// built once and shared by every viewport, not per-mesh
function buildGridLines(half: number, divisions: number): Float32Array {
  const step = (half * 2) / divisions;
  const verts: number[] = [];
  for (let i = 0; i <= divisions; i++) {
    const t = -half + i * step;
    verts.push(-half, 0, t, half, 0, t); // line parallel to X
    verts.push(t, 0, -half, t, 0, half); // line parallel to Z
  }
  return new Float32Array(verts);
}

// builds the grid pipeline + the shared line geometry once, reused by every viewport
// translucent, alpha-blended line-list, depth-tested but not depth-writing: the mesh
// occludes the grid, the grid never occludes anything
//   perspective: a real world-space plane at grid height, distance-fogged into the background
//   orthographic: skips the camera entirely, drawn straight in screen space with a fixed
//   cell size at the far depth, so no fog is needed
export function createGridRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
): GridRenderer {
  const module = device.createShaderModule({ code: gridShader });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs",
      buffers: [
        {
          arrayStride: POS_STRIDE,
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
        },
      ],
    },
    fragment: {
      module,
      entryPoint: "fs",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "src-alpha",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add",
            },
          },
        },
      ],
    },
    primitive: { topology: "line-list" },
    depthStencil: {
      format: DEPTH_FORMAT,
      depthWriteEnabled: false,
      depthCompare: "less-equal",
    },
    multisample: { count: SAMPLE_COUNT },
  });

  const gridVertices = buildGridLines(GRID_HALF, GRID_DIVISIONS);
  const vertexBuffer = device.createBuffer({
    size: gridVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, gridVertices);
  const vertexCount = gridVertices.length / 3;

  // scratch matrices + uniform staging
  const proj = mat4.create();
  const view = mat4.create();
  const viewProj = mat4.create();
  const model = mat4.create();
  const mvp = mat4.create();
  const invView = mat4.create();
  const data = new Float32Array(GRID_UNIFORM_FLOATS);

  // one uniform buffer + bind group per draw
  const slots: { buffer: GPUBuffer; bindGroup: GPUBindGroup }[] = [];
  const getSlot = (i: number) => {
    if (!slots[i]) {
      const buffer = device.createBuffer({
        size: GRID_UNIFORM_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer } }],
      });
      slots[i] = { buffer, bindGroup };
    }
    return slots[i];
  };

  const draw = (pass: GPURenderPassEncoder, grids: GridItem[]) => {
    grids.forEach((g, i) => {
      const { viewport: vp, projection, orbit, y } = g;
      const aspect = vp.width / vp.height;
      setOrbitCamera(proj, view, viewProj, projection, orbit, aspect);
      mat4.inverse(view, invView); // for the camera world position

      mat4.identity(model);
      mat4.translate(model, [0, y, 0], model);
      mat4.multiply(viewProj, model, mvp);
      data.set(mvp, 0);
      // cam.xyz = camera world pos, cam.w = grid height
      data[16] = invView[12];
      data[17] = invView[13];
      data[18] = invView[14];
      data[19] = y;
      // fog color = background
      data[20] = BG.r;
      data[21] = BG.g;
      data[22] = BG.b;
      data[23] = 1;
      // fog range in world units, relative to the camera distance (radius).
      data[24] = orbit.radius + 1.5; // start fading
      data[25] = orbit.radius + 12; // fully faded
      // ortho: draw the grid in screen space, camera-independent, at z = 1
      data[26] = projection === "orthographic" ? 1 : 0;
      data[27] = 0;
      // screen-space cell scale: fixed vertical size, X divided by aspect so
      // cells stay square in pixels regardless of the viewport shape
      const scaleY = GRID_SCREEN_CELL / GRID_STEP;
      data[28] = scaleY / aspect; // NDC-X per local unit
      data[29] = scaleY; // NDC-Y per local unit
      data[30] = 0;
      data[31] = 0;

      const slot = getSlot(i);
      device.queue.writeBuffer(slot.buffer, 0, data);

      pass.setViewport(vp.x, vp.y, vp.width, vp.height, 0, 1);
      pass.setScissorRect(vp.x, vp.y, vp.width, vp.height);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, slot.bindGroup);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(vertexCount);
    });
  };

  return { draw };
}
