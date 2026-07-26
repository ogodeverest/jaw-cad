import type { CSSProperties, RefObject } from 'react'
import {
  LEFT_PCT,
  RIGHT_PCT,
  type JawKey,
  type SlotState,
  type ViewKey,
} from '../scene/types'

interface Overlay {
  key: ViewKey
  label: string
  style: CSSProperties
}

const jawLabel = (k: JawKey) => (k === 'upper' ? 'Upper jaw' : 'Lower jaw')

// Overlays mirror the render viewports: left panels for the *visible* jaws
// (sized to fill), plus the combined view — which always renders and fills the
// whole stage when no jaw is shown.
function computeOverlays(slots: Record<JawKey, SlotState>): Overlay[] {
  const shownKeys = (['upper', 'lower'] as const).filter(
    (k) => slots[k].status === 'ready' && slots[k].visible,
  )
  const overlays: Overlay[] = []
  const n = shownKeys.length
  shownKeys.forEach((k, i) => {
    overlays.push({
      key: k,
      label: jawLabel(k),
      style: {
        left: 0,
        top: `${(100 * i) / n}%`,
        width: LEFT_PCT,
        height: `${100 / n}%`,
      },
    })
  })
  const noneShown = shownKeys.length === 0
  overlays.push({
    key: 'combined',
    label: 'All jaws',
    style: {
      left: noneShown ? 0 : LEFT_PCT,
      top: 0,
      width: noneShown ? '100%' : RIGHT_PCT,
      height: '100%',
    },
  })
  return overlays
}

interface StageProps {
  canvasRef: RefObject<HTMLCanvasElement | null>
  slots: Record<JawKey, SlotState>
  status: 'pending' | 'ready' | 'error'
  error: Error | null
  onPointerDown: (key: ViewKey) => (e: React.PointerEvent) => void
  onWheel: (key: ViewKey) => (e: React.WheelEvent) => void
}

export function Stage({
  canvasRef,
  slots,
  status,
  error,
  onPointerDown,
  onWheel,
}: StageProps) {
  const overlays = computeOverlays(slots)

  return (
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
  )
}
