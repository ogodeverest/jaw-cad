import { useCallback, useEffect, useRef } from "react";
import { useWebGPU, type FrameContext } from "../gpu/useWebGPU";
import {
  createRenderer,
  DEFAULT_ORBIT,
  loadPLY,
  type GpuMesh,
  type GridItem,
  type MeshData,
  type OrbitState,
  type Projection,
  type Renderer,
  type SceneItem,
  type Viewport,
} from "../gpu/mesh";
import { MATERIAL_PRESETS, type MaterialId } from "./materials";
import {
  clamp,
  DEG2RAD,
  LEFT_COL,
  type JawKey,
  type SlotState,
  type Vec3,
  type ViewKey,
} from "./types";

const POLAR_LIMIT = 1.45; // 83°, keeps the camera from flipping over the pole

interface SceneSlot {
  data: MeshData | null;
  gpu: GpuMesh | null;
}

interface UseSceneArgs {
  slots: Record<JawKey, SlotState>;
  patchSlot: (key: JawKey, patch: Partial<SlotState>) => void;
  // when the smooth brush is active, pointer drags paint instead of orbit
  brushActive: boolean;
  // brush circle radius in CSS px (matches the on-screen cursor)
  brushRadius: number;
}

// one paintable mesh in one viewport, snapshotted each frame for pointer handlers
interface PaintTarget {
  view: ViewKey;
  jaw: JawKey;
  item: SceneItem;
}

const SMOOTH_LAMBDA = 0.4;

// owns the gpu-side scene + render loop, slots gets mirrored into refs each
// render so onFrame (runs outside react) can read the latest values
export function useScene({
  slots,
  patchSlot,
  brushActive,
  brushRadius,
}: UseSceneArgs) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // mutable gpu-side scene, driven from the render loop (not react state)
  const sceneRef = useRef<{
    renderer: Renderer | null;
    upper: SceneSlot;
    lower: SceneSlot;
  }>({
    renderer: null,
    upper: { data: null, gpu: null },
    lower: { data: null, gpu: null },
  });

  const visibleRef = useRef<Record<JawKey, boolean>>({
    upper: false,
    lower: false,
  });
  const projectionRef = useRef<Record<JawKey, Projection>>({
    upper: "orthographic",
    lower: "orthographic",
  });
  const materialRef = useRef<Record<JawKey, MaterialId>>({
    upper: "sand",
    lower: "resin",
  });
  const translationRef = useRef<Record<JawKey, Vec3>>({
    upper: [0, 0, 0],
    lower: [0, 0, 0],
  });
  const rotationRef = useRef<Record<JawKey, Vec3>>({
    upper: [0, 0, 0],
    lower: [0, 0, 0],
  });
  const scaleRef = useRef<Record<JawKey, Vec3>>({
    upper: [1, 1, 1],
    lower: [1, 1, 1],
  });
  visibleRef.current = {
    upper: slots.upper.visible,
    lower: slots.lower.visible,
  };
  projectionRef.current = {
    upper: slots.upper.projection,
    lower: slots.lower.projection,
  };
  materialRef.current = {
    upper: slots.upper.material,
    lower: slots.lower.material,
  };
  translationRef.current = {
    upper: slots.upper.translation,
    lower: slots.lower.translation,
  };
  rotationRef.current = {
    upper: slots.upper.rotation,
    lower: slots.lower.rotation,
  };
  scaleRef.current = {
    upper: slots.upper.scale,
    lower: slots.lower.scale,
  };

  // orbit cameras live entirely in refs, dragging mutates them without
  // triggering react re-renders - the render loop reads them each frame
  const orbitRef = useRef<Record<ViewKey, OrbitState>>({
    // upper jaw: camera below, looking up (-Y up), shows its occlusal surface
    upper: { ...DEFAULT_ORBIT, azimuth: 0, polar: -Math.PI / 2 },
    // lower jaw: camera above, looking down (+Y down)
    lower: { ...DEFAULT_ORBIT, azimuth: 0, polar: Math.PI / 2 },
    combined: { ...DEFAULT_ORBIT, radius: 4 },
  });
  const dragRef = useRef<{ key: ViewKey; x: number; y: number } | null>(null);

  // per-frame snapshot of what's drawn where, written at the end of onFrame
  const paintTargetsRef = useRef<PaintTarget[]>([]);
  // which view is being painted (null = not painting)
  const paintRef = useRef<ViewKey | null>(null);
  const mvpScratch = useRef(new Float32Array(16));

  // brush inputs mirrored into refs for the window listeners (stable deps)
  const brushActiveRef = useRef(brushActive);
  brushActiveRef.current = brushActive;
  const brushRadiusRef = useRef(brushRadius);
  brushRadiusRef.current = brushRadius;

  const applyStroke = (view: ViewKey, clientX: number, clientY: number) => {
    const scene = sceneRef.current;
    const renderer = scene.renderer;
    const canvas = canvasRef.current;
    if (!renderer || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / rect.width; // framebuffer px per CSS px
    for (const t of paintTargetsRef.current) {
      if (t.view !== view) continue;
      const mesh = scene[t.jaw].gpu;
      if (!mesh) continue;
      const vp = t.item.viewport;
      renderer.mvpForItem(t.item, mvpScratch.current);
      renderer.smooth(mesh, {
        mvp: mvpScratch.current,
        viewportW: vp.width,
        viewportH: vp.height,
        // circle center relative to the viewport origin, in framebuffer px
        centerX: (clientX - rect.left) * dpr - vp.x,
        centerY: (clientY - rect.top) * dpr - vp.y,
        radius: brushRadiusRef.current * dpr,
        lambda: SMOOTH_LAMBDA,
      });
    }
  };
  // held in a ref so the (stable) window listeners always call the latest one
  const applyStrokeRef = useRef(applyStroke);
  applyStrokeRef.current = applyStroke;

  // only pointerdown lives on the overlay, move/up are on window (below) so a
  // drag keeps orbiting/painting even when the pointer leaves the small panel
  const onPointerDown = (key: ViewKey) => (e: React.PointerEvent) => {
    if (brushActiveRef.current) {
      paintRef.current = key;
      applyStrokeRef.current(key, e.clientX, e.clientY);
      return;
    }
    dragRef.current = { key, x: e.clientX, y: e.clientY };
  };
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (paintRef.current) {
        applyStrokeRef.current(paintRef.current, e.clientX, e.clientY);
        return;
      }
      const drag = dragRef.current;
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      drag.x = e.clientX;
      drag.y = e.clientY;
      const orbit = orbitRef.current[drag.key];
      orbit.azimuth += dx * 0.01;
      orbit.polar = clamp(orbit.polar + dy * 0.01, -POLAR_LIMIT, POLAR_LIMIT);
    };
    const onUp = () => {
      dragRef.current = null;
      paintRef.current = null;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);
  const onWheel = (key: ViewKey) => (e: React.WheelEvent) => {
    const orbit = orbitRef.current[key];
    orbit.radius = clamp(orbit.radius * (1 + e.deltaY * 0.001), 2, 20);
  };

  const handleFile = async (key: JawKey, file: File | undefined) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".ply")) {
      patchSlot(key, { status: "error" });
      return;
    }
    patchSlot(key, { status: "loading", name: file.name });
    try {
      const data = await loadPLY(file);
      const slot = sceneRef.current[key];
      slot.gpu?.destroy();
      slot.gpu = null; // rebuilt from new data on the next frame
      slot.data = data;
      patchSlot(key, { status: "ready", visible: true, name: file.name });
    } catch (err) {
      console.error(err);
      patchSlot(key, { status: "error" });
    }
  };

  const onFrame = useCallback((frame: FrameContext) => {
    const scene = sceneRef.current;
    if (!scene.renderer) {
      scene.renderer = createRenderer(frame.device, frame.format);
    }
    const renderer = scene.renderer;

    for (const key of ["upper", "lower"] as const) {
      const slot = scene[key];
      if (slot.data && !slot.gpu) slot.gpu = renderer.createMesh(slot.data);
    }

    const vis = visibleRef.current;
    // layout is keyed on loaded jaws, a hidden jaw keeps its panel (grid only)
    // but its mesh isn't drawn - not in its own panel nor the combined view
    const loaded = (["upper", "lower"] as const).filter((k) => scene[k].gpu);

    const W = frame.width;
    const H = frame.height;
    const items: SceneItem[] = [];
    const targets: PaintTarget[] = [];

    // draw one jaw's mesh, using its own material but a chosen view's camera
    const push = (
      jaw: JawKey,
      view: ViewKey,
      viewport: Viewport,
      projection: Projection,
      offset: [number, number, number],
    ) => {
      const mat = MATERIAL_PRESETS[materialRef.current[jaw]];
      const rot = rotationRef.current[jaw];
      const item: SceneItem = {
        mesh: scene[jaw].gpu!,
        viewport,
        projection,
        orbit: orbitRef.current[view],
        offset,
        rotation: [rot[0] * DEG2RAD, rot[1] * DEG2RAD, rot[2] * DEG2RAD],
        scale: scaleRef.current[jaw],
        color: mat.color,
        metallic: mat.metallic,
        roughness: mat.roughness,
      };
      items.push(item);
      targets.push({ view, jaw, item });
    };

    const grids: GridItem[] = [];
    const minY = (k: JawKey) => scene[k].data?.minY ?? 0;
    const trans = translationRef.current;
    const scl = scaleRef.current;
    // a jaw's own translation, plus the combined-view stacking offset on Y
    const jawOffset = (k: JawKey, stackY: number): Vec3 => [
      trans[k][0],
      trans[k][1] + stackY,
      trans[k][2],
    ];
    // grid sticks to the jaw's (scaled) bottom, following Y scale + translation
    const gridY = (k: JawKey, stackY: number) =>
      minY(k) * scl[k][1] + trans[k][1] + stackY;

    const shown = loaded.filter((k) => vis[k]);

    // the combined view always renders (even with nothing uploaded, just the
    // grid), left column stacks one panel per visible jaw, with no jaw shown
    // it collapses to zero width and the combined view fills the whole stage
    const leftW = shown.length === 0 ? 0 : Math.round(W * LEFT_COL);
    const right: Viewport = { x: leftW, y: 0, width: W - leftW, height: H };

    const n = shown.length;
    shown.forEach((k, i) => {
      const y0 = Math.floor((H * i) / n);
      const y1 = Math.floor((H * (i + 1)) / n);
      const vp: Viewport = { x: 0, y: y0, width: leftW, height: y1 - y0 };
      push(k, k, vp, projectionRef.current[k], jawOffset(k, 0));
      grids.push({
        viewport: vp,
        projection: projectionRef.current[k],
        orbit: orbitRef.current[k],
        y: gridY(k, 0),
      });
    });

    shown.forEach((k) =>
      push(k, "combined", right, "perspective", jawOffset(k, 0)),
    );
    grids.push({
      viewport: right,
      projection: "perspective",
      orbit: orbitRef.current.combined,
      y: shown.length ? Math.min(...shown.map((k) => gridY(k, 0))) : 0,
    });

    paintTargetsRef.current = targets;

    renderer.render(frame, items, grids);
  }, []);

  const { status, error } = useWebGPU(canvasRef, onFrame);

  return { canvasRef, status, error, onPointerDown, onWheel, handleFile };
}
