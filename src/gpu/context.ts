// Extend base Error class to WebGPUUnsupportedError for branching
export class WebGPUUnsupportedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "WebGPUUnsupportedError";
  }
}

// What we return from the initWebGPU function
// device: logical device, allows us to create buffers, textures, pipeline, shader modules etc
// adapter: represents a specific physical GPU (limits, capabilities, features) (not included)
// context: canvas context
// format: texture format our device supports for rendering (most of the time bgra-8bit on desktops, pixel-format of swap-chain/canvas where we render into)
// canvas: canvas element
export interface GPUContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  canvas: HTMLCanvasElement;
}

// this function handles the requesting of adapter, device as well as returns canvas, webgpu context and the expected texture format for rendering
// also configures the webgpu context with device, format and alphaMode
export async function initWebGPU(
  canvas: HTMLCanvasElement,
): Promise<GPUContext> {
  if (!("gpu" in navigator)) {
    throw new WebGPUUnsupportedError("WebGPU isn't available in this browser");
  }

  // request adapter: high-performance asks for discrete GPU not the integrated one
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new WebGPUUnsupportedError("No GPU adapter available.");
  }

  // requests device the logical device
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new WebGPUUnsupportedError("Couldn't get a webgpu canvas context.");
  }

  // preferred swapchain texture format, getting this the same as what the device expects
  // will avoid another pass to swap color channels (better performance)
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    // canvas alpha channel is ignored, renders as opaque on top of the other HTML
    alphaMode: "opaque",
  });

  // return GPUContext
  return { device, context, format, canvas };
}
