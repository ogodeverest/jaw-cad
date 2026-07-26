import type { JawKey, SlotState } from '../scene/types'
import { VecInput } from './VecInput'
import {
  CURSOR_MAX_RADIUS,
  CURSOR_MIN_RADIUS,
  type CursorBrush,
} from './cursorBrush'

type TransformField = 'translation' | 'rotation' | 'scale'

interface ToolsPanelProps {
  selectedJaw: JawKey
  onSelectJaw: (jaw: JawKey) => void
  slot: SlotState
  brush: CursorBrush
  onVecAxis: (field: TransformField, axis: number, value: number) => void
  onResetTransform: () => void
}

export function ToolsPanel({
  selectedJaw,
  onSelectJaw,
  slot,
  brush,
  onVecAxis,
  onResetTransform,
}: ToolsPanelProps) {
  return (
    <section className="control">
      <div className="control__label">Tools</div>
      <select
        className="select"
        aria-label="Selected jaw"
        value={selectedJaw}
        onChange={(e) => onSelectJaw(e.target.value as JawKey)}
      >
        <option value="upper">Upper jaw</option>
        <option value="lower">Lower jaw</option>
      </select>

      <div className="tool-group">
        <div className="tool-group__title">Smooth</div>
        <label className="toggle">
          <input
            type="checkbox"
            checked={brush.active}
            onChange={(e) => brush.setActive(e.target.checked)}
          />
          <span>Brush</span>
        </label>
        <label className="slider">
          <span>Radius</span>
          <input
            type="range"
            min={CURSOR_MIN_RADIUS}
            max={CURSOR_MAX_RADIUS}
            value={brush.radius}
            disabled={!brush.active}
            onChange={(e) => brush.setRadius(Number(e.target.value))}
          />
          <span className="slider__value">{brush.radius}</span>
        </label>
      </div>

      <div className="tool-group">
        <div className="tool-group__title">Transform (TRS)</div>

        <VecInput
          label="Translation"
          step={0.05}
          value={slot.translation}
          onChange={(axis, value) => onVecAxis('translation', axis, value)}
        />
        <VecInput
          label="Rotation (°)"
          step={5}
          value={slot.rotation}
          onChange={(axis, value) => onVecAxis('rotation', axis, value)}
          style={{ marginTop: 10 }}
        />
        <VecInput
          label="Scale"
          step={0.05}
          value={slot.scale}
          onChange={(axis, value) => onVecAxis('scale', axis, value)}
          style={{ marginTop: 10 }}
        />

        <button
          type="button"
          className="upload-btn"
          style={{ marginTop: 12 }}
          onClick={onResetTransform}
        >
          Reset transform
        </button>
      </div>
    </section>
  )
}
