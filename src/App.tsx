import { useCallback, useEffect, useRef, useState } from 'react'
import { useWebGPU, type FrameContext } from './gpu/useWebGPU'
import {
  createRenderer,
  DEFAULT_ORBIT,
  loadPLY,
  type GpuMesh,
  type MeshData,
  type OrbitState,
  type Projection,
  type Renderer,
  type SceneItem,
  type Viewport,
} from './gpu/mesh'
import './App.css'

const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v))
const POLAR_LIMIT = 1.45 // ~83°, keeps the camera from flipping over the pole
const CURSOR_MIN_RADIUS = 6
const CURSOR_MAX_RADIUS = 44 // current (largest) circle size

type JawKey = 'upper' | 'lower'
// Cameras: one per jaw, plus a shared "combined" view showing both meshes.
type ViewKey = JawKey | 'combined'
type SlotStatus = 'empty' | 'loading' | 'ready' | 'error'

// Vertical separation of the two jaws in the combined view.
const COMBINED_OFFSET = 0.9
// Fraction of the stage width used by the stacked upper/lower column.
const LEFT_COL = 0.4
const LEFT_PCT = `${LEFT_COL * 100}%`
const RIGHT_PCT = `${(1 - LEFT_COL) * 100}%`

interface SlotState {
  name: string | null
  visible: boolean
  status: SlotStatus
  projection: Projection
  material: MaterialId
}

const EMPTY_SLOT: SlotState = {
  name: null,
  visible: false,
  status: 'empty',
  projection: 'perspective',
  material: 'sand',
}

// Selectable PBR material presets: albedo + metallic/roughness.
type MaterialId = 'sand' | 'resin' | 'gum' | 'gold' | 'chrome' | 'jade'

interface Material {
  label: string
  color: [number, number, number]
  metallic: number
  roughness: number
}

const MATERIAL_PRESETS: Record<MaterialId, Material> = {
  sand: { label: 'Sand', color: [0.8, 0.66, 0.46], metallic: 0, roughness: 0.55 },
  resin: { label: 'Blue resin', color: [0.45, 0.6, 0.95], metallic: 0, roughness: 0.35 },
  gum: { label: 'Gum pink', color: [0.85, 0.45, 0.5], metallic: 0, roughness: 0.5 },
  gold: { label: 'Gold', color: [1.0, 0.78, 0.34], metallic: 1, roughness: 0.25 },
  chrome: { label: 'Chrome', color: [0.95, 0.95, 0.97], metallic: 1, roughness: 0.08 },
  jade: { label: 'Jade', color: [0.35, 0.7, 0.5], metallic: 0, roughness: 0.3 },
}

interface SceneSlot {
  data: MeshData | null
  gpu: GpuMesh | null
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Mutable GPU-side scene, driven from the render loop (not React state).
  const sceneRef = useRef<{
    renderer: Renderer | null
    upper: SceneSlot
    lower: SceneSlot
  }>({
    renderer: null,
    upper: { data: null, gpu: null },
    lower: { data: null, gpu: null },
  })

  const [slots, setSlots] = useState<Record<JawKey, SlotState>>({
    upper: { ...EMPTY_SLOT, material: 'sand' },
    lower: { ...EMPTY_SLOT, material: 'resin' },
  })
  // Mirror the per-jaw view state into refs the render loop reads each frame.
  const visibleRef = useRef<Record<JawKey, boolean>>({
    upper: false,
    lower: false,
  })
  const projectionRef = useRef<Record<JawKey, Projection>>({
    upper: 'perspective',
    lower: 'perspective',
  })
  const materialRef = useRef<Record<JawKey, MaterialId>>({
    upper: 'sand',
    lower: 'resin',
  })
  visibleRef.current = {
    upper: slots.upper.visible,
    lower: slots.lower.visible,
  }
  projectionRef.current = {
    upper: slots.upper.projection,
    lower: slots.lower.projection,
  }
  materialRef.current = {
    upper: slots.upper.material,
    lower: slots.lower.material,
  }

  // Orbit cameras live entirely in refs — dragging mutates them without
  // triggering React re-renders; the render loop reads them each frame.
  const orbitRef = useRef<Record<ViewKey, OrbitState>>({
    upper: { ...DEFAULT_ORBIT },
    lower: { ...DEFAULT_ORBIT },
    combined: { ...DEFAULT_ORBIT, radius: 6 },
  })
  const dragRef = useRef<{ key: ViewKey; x: number; y: number } | null>(null)

  // Screen-space cursor circle: moved directly via ref to avoid re-rendering.
  const [cursorCircle, setCursorCircle] = useState(false)
  const [cursorRadius, setCursorRadius] = useState(CURSOR_MAX_RADIUS)
  const cursorRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!cursorCircle) return
    const onMove = (e: PointerEvent) => {
      const el = cursorRef.current
      if (el) el.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`
    }
    window.addEventListener('pointermove', onMove)
    return () => window.removeEventListener('pointermove', onMove)
  }, [cursorCircle])

  // Only pointerdown lives on the overlay; move/up are on window (below) so a
  // drag keeps orbiting even when the pointer leaves the small panel.
  const onPointerDown = (key: ViewKey) => (e: React.PointerEvent) => {
    if (cursorCircle) return // tool active: don't orbit
    dragRef.current = { key, x: e.clientX, y: e.clientY }
  }
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return
      const dx = e.clientX - drag.x
      const dy = e.clientY - drag.y
      drag.x = e.clientX
      drag.y = e.clientY
      const orbit = orbitRef.current[drag.key]
      orbit.azimuth += dx * 0.01
      orbit.polar = clamp(orbit.polar + dy * 0.01, -POLAR_LIMIT, POLAR_LIMIT)
    }
    const onUp = () => {
      dragRef.current = null
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
  }, [])
  const onWheel = (key: ViewKey) => (e: React.WheelEvent) => {
    const orbit = orbitRef.current[key]
    orbit.radius = clamp(orbit.radius * (1 + e.deltaY * 0.001), 2, 20)
  }

  const patchSlot = (key: JawKey, patch: Partial<SlotState>) =>
    setSlots((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }))

  const handleFile = async (key: JawKey, file: File | undefined) => {
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.ply')) {
      patchSlot(key, { status: 'error' })
      return
    }
    patchSlot(key, { status: 'loading', name: file.name })
    try {
      const data = await loadPLY(file)
      const slot = sceneRef.current[key]
      slot.gpu?.destroy()
      slot.gpu = null // rebuilt from new data on the next frame
      slot.data = data
      patchSlot(key, { status: 'ready', visible: true, name: file.name })
    } catch (err) {
      console.error(err)
      patchSlot(key, { status: 'error' })
    }
  }

  const onFrame = useCallback((frame: FrameContext) => {
    const scene = sceneRef.current
    if (!scene.renderer) {
      scene.renderer = createRenderer(frame.device, frame.format)
    }
    const renderer = scene.renderer

    // Build GPU buffers for any slot that has fresh mesh data.
    for (const key of ['upper', 'lower'] as const) {
      const slot = scene[key]
      if (slot.data && !slot.gpu) slot.gpu = renderer.createMesh(slot.data)
    }

    const vis = visibleRef.current
    const active = (['upper', 'lower'] as const).filter(
      (k) => vis[k] && scene[k].gpu,
    )

    const W = frame.width
    const H = frame.height
    const items: SceneItem[] = []

    // Draw one jaw's mesh, using its own material but a chosen view's camera.
    const push = (
      jaw: JawKey,
      view: ViewKey,
      viewport: Viewport,
      projection: Projection,
      offset: [number, number, number],
    ) => {
      const mat = MATERIAL_PRESETS[materialRef.current[jaw]]
      items.push({
        mesh: scene[jaw].gpu!,
        viewport,
        projection,
        orbit: orbitRef.current[view],
        offset,
        color: mat.color,
        metallic: mat.metallic,
        roughness: mat.roughness,
      })
    }

    const NO_OFFSET: [number, number, number] = [0, 0, 0]

    if (active.length === 1) {
      const k = active[0]
      push(k, k, { x: 0, y: 0, width: W, height: H }, projectionRef.current[k], NO_OFFSET)
    } else if (active.length === 2) {
      // Left column: upper stacked over lower. Right: both jaws combined.
      const leftW = Math.round(W * LEFT_COL)
      const halfH = Math.floor(H / 2)
      push('upper', 'upper', { x: 0, y: 0, width: leftW, height: halfH }, projectionRef.current.upper, NO_OFFSET)
      push('lower', 'lower', { x: 0, y: halfH, width: leftW, height: H - halfH }, projectionRef.current.lower, NO_OFFSET)

      const right: Viewport = { x: leftW, y: 0, width: W - leftW, height: H }
      push('upper', 'combined', right, 'perspective', [0, COMBINED_OFFSET, 0])
      push('lower', 'combined', right, 'perspective', [0, -COMBINED_OFFSET, 0])
    }

    renderer.render(frame, items)
  }, [])

  const { status, error } = useWebGPU(canvasRef, onFrame)

  // Overlays that track the render viewports (kept in sync with onFrame's layout).
  const activeKeys = (['upper', 'lower'] as const).filter(
    (k) => slots[k].visible && slots[k].status === 'ready',
  )
  const overlays: {
    key: ViewKey
    label: string
    style: React.CSSProperties
  }[] =
    activeKeys.length === 1
      ? [
          {
            key: activeKeys[0],
            label: activeKeys[0] === 'upper' ? 'Upper jaw' : 'Lower jaw',
            style: { left: 0, top: 0, width: '100%', height: '100%' },
          },
        ]
      : activeKeys.length === 2
        ? [
            {
              key: 'upper',
              label: 'Upper jaw',
              style: { left: 0, top: 0, width: LEFT_PCT, height: '50%' },
            },
            {
              key: 'lower',
              label: 'Lower jaw',
              style: { left: 0, top: '50%', width: LEFT_PCT, height: '50%' },
            },
            {
              key: 'combined',
              label: 'Both jaws',
              style: { left: LEFT_PCT, top: 0, width: RIGHT_PCT, height: '100%' },
            },
          ]
        : []

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar__title">jaw-scan-cad</div>

        <JawControl
          label="Upper jaw"
          slot={slots.upper}
          onUpload={(file) => handleFile('upper', file)}
          onToggle={() => patchSlot('upper', { visible: !slots.upper.visible })}
          onProjection={(p) => patchSlot('upper', { projection: p })}
          onMaterial={(m) => patchSlot('upper', { material: m })}
        />
        <JawControl
          label="Lower jaw"
          slot={slots.lower}
          onUpload={(file) => handleFile('lower', file)}
          onToggle={() => patchSlot('lower', { visible: !slots.lower.visible })}
          onProjection={(p) => patchSlot('lower', { projection: p })}
          onMaterial={(m) => patchSlot('lower', { material: m })}
        />

        <section className="control">
          <div className="control__label">Tools</div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={cursorCircle}
              onChange={(e) => setCursorCircle(e.target.checked)}
            />
            <span>Cursor circle</span>
          </label>
          <label className="slider">
            <span>Radius</span>
            <input
              type="range"
              min={CURSOR_MIN_RADIUS}
              max={CURSOR_MAX_RADIUS}
              value={cursorRadius}
              disabled={!cursorCircle}
              onChange={(e) => setCursorRadius(Number(e.target.value))}
            />
            <span className="slider__value">{cursorRadius}</span>
          </label>
        </section>
      </aside>

      <div className="stage">
        <canvas ref={canvasRef} className="gpu-canvas" />

        <div className="viewports">
          {overlays.map(({ key, label, style }) => (
            <div
              key={key}
              className="viewport-overlay"
              style={style}
              tabIndex={0}
              role="group"
              aria-label={`${label} viewport`}
              onPointerDown={onPointerDown(key)}
              onWheel={onWheel(key)}
            >
              <span className="viewport-overlay__tag">{label}</span>
            </div>
          ))}
        </div>

        {status === 'error' && (
          <div className="overlay overlay--error">
            <h1>WebGPU unavailable</h1>
            <p>{error?.message}</p>
          </div>
        )}
      </div>

      {cursorCircle && (
        <div ref={cursorRef} className="cursor-circle" aria-hidden="true">
          {(() => {
            const size = cursorRadius * 2 + 4 // +stroke padding
            return (
              <svg
                width={size}
                height={size}
                viewBox={`0 0 ${size} ${size}`}
              >
                <circle
                  cx={size / 2}
                  cy={size / 2}
                  r={cursorRadius}
                  fill="none"
                  stroke="#7aa2ff"
                  strokeWidth="2"
                />
              </svg>
            )
          })()}
        </div>
      )}
    </div>
  )
}

interface JawControlProps {
  label: string
  slot: SlotState
  onUpload: (file: File | undefined) => void
  onToggle: () => void
  onProjection: (projection: Projection) => void
  onMaterial: (material: MaterialId) => void
}

function JawControl({
  label,
  slot,
  onUpload,
  onToggle,
  onProjection,
  onMaterial,
}: JawControlProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const ready = slot.status === 'ready'

  return (
    <section className="control">
      <div className="control__label">{label}</div>

      <input
        ref={inputRef}
        type="file"
        accept=".ply"
        hidden
        onChange={(e) => {
          onUpload(e.target.files?.[0])
          e.target.value = '' // allow re-selecting the same file
        }}
      />

      <button
        type="button"
        className="upload-btn"
        onClick={() => inputRef.current?.click()}
      >
        {slot.status === 'loading' ? 'Loading…' : 'Upload .ply'}
      </button>

      <div className="control__row">
        <label className="toggle">
          <input
            type="checkbox"
            checked={slot.visible}
            disabled={!ready}
            onChange={onToggle}
          />
          <span>Show</span>
        </label>
        <span
          className={
            'filename' + (slot.status === 'error' ? ' filename--error' : '')
          }
          title={slot.name ?? undefined}
        >
          {slot.status === 'error'
            ? 'Not a .ply file'
            : (slot.name ?? 'No file')}
        </span>
      </div>

      <div
        className="segmented"
        role="group"
        aria-label={`${label} projection`}
      >
        {(['perspective', 'orthographic'] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            className={
              'segmented__btn' +
              (slot.projection === mode ? ' segmented__btn--active' : '')
            }
            aria-pressed={slot.projection === mode}
            onClick={() => onProjection(mode)}
          >
            {mode === 'perspective' ? 'Perspective' : 'Orthographic'}
          </button>
        ))}
      </div>

      <select
        className="select"
        aria-label={`${label} material`}
        value={slot.material}
        onChange={(e) => onMaterial(e.target.value as MaterialId)}
      >
        {(Object.keys(MATERIAL_PRESETS) as MaterialId[]).map((id) => (
          <option key={id} value={id}>
            {MATERIAL_PRESETS[id].label}
          </option>
        ))}
      </select>
    </section>
  )
}

export default App
