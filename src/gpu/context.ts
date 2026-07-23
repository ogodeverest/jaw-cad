export class WebGPUUnsupportedError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'WebGPUUnsupportedError'
  }
}

export interface GPUContext {
  device: GPUDevice
  context: GPUCanvasContext
  /** Preferred swap-chain format for the current platform, e.g. "bgra8unorm". */
  format: GPUTextureFormat
  canvas: HTMLCanvasElement
}

/**
 * Requests an adapter + device and configures the canvas' WebGPU context.
 * Throws {@link WebGPUUnsupportedError} when WebGPU is unavailable so callers
 * can show a friendly fallback.
 */
export async function initWebGPU(
  canvas: HTMLCanvasElement,
): Promise<GPUContext> {
  if (!('gpu' in navigator)) {
    throw new WebGPUUnsupportedError(
      'WebGPU is not supported in this browser. Try a recent version of Chrome, Edge, or Safari.',
    )
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  })
  if (!adapter) {
    throw new WebGPUUnsupportedError(
      'No suitable GPU adapter found. WebGPU may be disabled or unavailable on this hardware.',
    )
  }

  const device = await adapter.requestDevice()

  const context = canvas.getContext('webgpu')
  if (!context) {
    throw new WebGPUUnsupportedError('Failed to acquire a WebGPU canvas context.')
  }

  const format = navigator.gpu.getPreferredCanvasFormat()
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  })

  return { device, context, format, canvas }
}
