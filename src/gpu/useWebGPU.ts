import { useEffect, useRef, useState } from "react";
import { initWebGPU, WebGPUUnsupportedError, type GPUContext } from "./context";

const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";
const SAMPLE_COUNT = 4; // 4x MSAA

/** Everything a per-frame render callback needs. */
export interface FrameContext extends GPUContext {
  encoder: GPUCommandEncoder;
  /** Swap-chain view — use as the MSAA resolve target. */
  view: GPUTextureView;
  /** Multisampled color view to render into. */
  msaaView: GPUTextureView;
  depthView: GPUTextureView;
  width: number;
  height: number;
}

export interface UseWebGPUState {
  status: "pending" | "ready" | "error";
  error: Error | null;
}

/**
 * Boots WebGPU against the given canvas, keeps the canvas + depth texture sized
 * to the element (device-pixel-ratio aware), and runs an rAF render loop that
 * hands each frame to `onFrame`. A default clear pass runs when `onFrame` is
 * omitted, so you get visible pixels before wiring a real pipeline.
 */
export function useWebGPU(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  onFrame?: (frame: FrameContext) => void,
): UseWebGPUState {
  const [state, setState] = useState<UseWebGPUState>({
    status: "pending",
    error: null,
  });

  // Keep the latest callback without restarting the whole GPU lifecycle.
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;
    let raf = 0;
    let depthTexture: GPUTexture | null = null;
    let msaaTexture: GPUTexture | null = null;

    const run = async () => {
      let gpu: GPUContext;
      try {
        gpu = await initWebGPU(canvas);
      } catch (err) {
        if (disposed) return;
        setState({
          status: "error",
          error:
            err instanceof Error
              ? err
              : new WebGPUUnsupportedError("Failed to initialize WebGPU."),
        });
        return;
      }

      if (disposed) {
        gpu.device.destroy();
        return;
      }

      const { device, context, format } = gpu;

      const resize = () => {
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
        const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
        if (
          canvas.width === width &&
          canvas.height === height &&
          depthTexture &&
          msaaTexture
        ) {
          return;
        }
        canvas.width = width;
        canvas.height = height;

        depthTexture?.destroy();
        depthTexture = device.createTexture({
          size: [width, height],
          format: DEPTH_FORMAT,
          sampleCount: SAMPLE_COUNT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Multisampled color target that resolves into the swap chain.
        msaaTexture?.destroy();
        msaaTexture = device.createTexture({
          size: [width, height],
          format,
          sampleCount: SAMPLE_COUNT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
      };

      resize();
      const observer = new ResizeObserver(resize);
      observer.observe(canvas);

      setState({ status: "ready", error: null });

      const frame = () => {
        if (disposed) return;
        resize();

        const encoder = device.createCommandEncoder();
        const view = context.getCurrentTexture().createView();
        const msaaView = msaaTexture!.createView();
        const depthView = depthTexture!.createView();

        const cb = onFrameRef.current;
        if (cb) {
          cb({
            ...gpu,
            encoder,
            view,
            msaaView,
            depthView,
            width: canvas.width,
            height: canvas.height,
          });
        } else {
          // Default: clear to a dark slate so the canvas is obviously alive.
          const pass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: msaaView,
                resolveTarget: view,
                clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
            depthStencilAttachment: {
              view: depthView,
              depthClearValue: 1,
              depthLoadOp: "clear",
              depthStoreOp: "store",
            },
          });
          pass.end();
        }

        device.queue.submit([encoder.finish()]);
        raf = requestAnimationFrame(frame);
      };
      raf = requestAnimationFrame(frame);

      cleanup = () => {
        observer.disconnect();
        cancelAnimationFrame(raf);
        depthTexture?.destroy();
        msaaTexture?.destroy();
        device.destroy();
      };
    };

    let cleanup = () => {};
    void run();

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      cleanup();
    };
  }, [canvasRef]);

  return state;
}

export { DEPTH_FORMAT, SAMPLE_COUNT };
