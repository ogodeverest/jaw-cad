import { useRef } from 'react'
import type { Projection } from '../gpu/mesh'
import type { SlotState } from '../scene/types'
import { MATERIAL_PRESETS, type MaterialId } from '../scene/materials'

interface JawControlProps {
  label: string
  slot: SlotState
  onUpload: (file: File | undefined) => void
  onToggle: () => void
  onProjection: (projection: Projection) => void
  onMaterial: (material: MaterialId) => void
}

export function JawControl({
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
