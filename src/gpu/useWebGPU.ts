import { useEffect, useRef, useState } from "react";
import { initWebGPU, WebGPUUnsupportedError, type GPUContext } from "./context";

// depth texture format: "depth24plus" at least 24bit depth (unorm or float, impl's choice)
const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";
// 4 sub-pixel sample count for MSAA
const SAMPLE_COUNT = 4;

// per frame data extends GPUContext with:
// encoder: command encoder created from the device, render passes and other encoder-level commands are recorded into it
// the command buffer is the output which is then submitted
// view: swap-chain view (where MSAA resolves)
// msaaView: the view of the MSAA texture
// depthView: the view of the depthTexture
// width / height of the canvas
export interface FrameContext extends GPUContext {
  encoder: GPUCommandEncoder;
  view: GPUTextureView;
  msaaView: GPUTextureView;
  depthView: GPUTextureView;
  width: number;
  height: number;
}

export interface UseWebGPUState {
  status: "pending" | "ready" | "error";
  error: Error | null;
}

// canvasRef: canvas html element where we will render into
// onFrame: callback injected for rendering outside the hook
// hook handles booting, the loop and resizing only
export function useWebGPU(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  onFrame?: (frame: FrameContext) => void,
): UseWebGPUState {
  // keep track of WebGpUState
  const [state, setState] = useState<UseWebGPUState>({
    status: "pending",
    error: null,
  });

  // this puts the onFrame cb inside a ref so changes to it do not restart the useEffect
  // and re-set + cleanup the GPUContext state
  const onFrameRef = useRef(onFrame);
  onFrameRef.current = onFrame;

  // on component initial mount
  useEffect(() => {
    // get current html canvas reference
    const canvas = canvasRef.current;

    //early return
    if (!canvas) return;

    let disposed = false;
    // raf ID
    let raf = 0;
    // depth texture and MSAA texture
    let depthTexture: GPUTexture | null = null;
    let msaaTexture: GPUTexture | null = null;

    const run = async () => {
      let gpu: GPUContext;
      // try to boot webgpu
      try {
        gpu = await initWebGPU(canvas);
      } catch (err) {
        // catch error but if the component is unmounted do not set state
        if (disposed) return;

        // set error and status state
        setState({
          status: "error",
          error:
            err instanceof Error
              ? err
              : new WebGPUUnsupportedError("Failed to initialize WebGPU."),
        });

        return;
      }

      // if the component is unmounted  destroy the device and return early
      if (disposed) {
        gpu.device.destroy();
        return;
      }

      // get requested device, context and format from gpu
      const { device, context, format } = gpu;

      // resize callback
      const resize = () => {
        // get devicePixelRatio CSS -> device pixels
        const dpr = window.devicePixelRatio || 1;
        // convert to device pixels
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
        // set new width / height
        canvas.width = width;
        canvas.height = height;

        // destroy old depth texture
        depthTexture?.destroy();
        // create new one
        depthTexture = device.createTexture({
          // same size as current canvas
          size: [width, height],
          format: DEPTH_FORMAT,
          sampleCount: SAMPLE_COUNT,
          // RENDER_ATTACHMENT: usable as a render-pass attachment (depth-stencil here)
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // destroy old MSAA texture
        msaaTexture?.destroy();
        //create new one
        msaaTexture = device.createTexture({
          size: [width, height],
          format,
          // 4 sub-pixel samples: coverage is tested per sample at triangle edges each sample gets either pixel color or destination color then all samples are averaged together to form final pixel color,
          sampleCount: SAMPLE_COUNT,
          // RENDER_ATTACHMENT: accepted as color attachment for rendering
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
      };

      resize();

      //set up resize observer with resize callback and observe canvas element
      const observer = new ResizeObserver(resize);
      observer.observe(canvas);

      setState({ status: "ready", error: null });

      // raf callback
      const frame = () => {
        // if component is unmounted return early
        if (disposed) return;

        // create command encoder for this frame
        // handles per pass command or encoder level commands
        // creates a final command buffer which is executed in GPU
        const encoder = device.createCommandEncoder();
        // current swap-chain texture, this just gets whichever texture is free to render into this frame
        const view = context.getCurrentTexture().createView();
        // create views for msaa and depth view from our textures
        const msaaView = msaaTexture!.createView();
        const depthView = depthTexture!.createView();

        // if a onFrame cb exists call it with per frame context
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
          // otherwise just clear the view
          const pass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: msaaView,
                // where MSAA  resolves to
                resolveTarget: view,
                clearValue: { r: 0.05, g: 0.06, b: 0.08, a: 1 },

                // we want to clear previous color here
                loadOp: "clear",

                // texture needs to be preserved so it can be rendered we cannot discard it
                storeOp: "store",
              },
            ],
            // instructions for depth buffer
            depthStencilAttachment: {
              view: depthView,
              depthClearValue: 1,
              depthLoadOp: "clear",
              // we can discard this because it is not useful to keep around in GPU VRAM (we will not read this back)
              depthStoreOp: "discard",
            },
          });
          // end the pass
          pass.end();
        }

        // submit all the recorded commands for execution in GPU
        device.queue.submit([encoder.finish()]);
        // schedule next frame
        raf = requestAnimationFrame(frame);
      };
      // start raf cycle here
      raf = requestAnimationFrame(frame);

      // assign clean up closure here
      // disconnect observer, cancel raf and destroy textures + device on unmount
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

    // mark as disposed and cleanup on unmount
    return () => {
      disposed = true;
      cleanup();
    };
  }, [canvasRef]);

  return state;
}

export { DEPTH_FORMAT, SAMPLE_COUNT };
