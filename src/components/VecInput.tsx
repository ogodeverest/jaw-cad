import type { CSSProperties } from 'react'
import type { Vec3 } from '../scene/types'

interface VecInputProps {
  label: string
  step: number
  value: Vec3
  onChange: (axis: number, value: number) => void
  style?: CSSProperties
}

/** A labeled X/Y/Z row of number inputs for one transform component. */
export function VecInput({ label, step, value, onChange, style }: VecInputProps) {
  return (
    <>
      <div className="tool-note" style={style}>
        {label}
      </div>
      <div className="vec-input">
        {(['X', 'Y', 'Z'] as const).map((axis, i) => (
          <label key={axis} className="vec-input__field">
            <span>{axis}</span>
            <input
              type="number"
              step={step}
              value={value[i]}
              onChange={(e) => onChange(i, Number(e.target.value))}
            />
          </label>
        ))}
      </div>
    </>
  )
}
