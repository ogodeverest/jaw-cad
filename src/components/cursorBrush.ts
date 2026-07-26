import { useState } from 'react'

export const CURSOR_MIN_RADIUS = 6
export const CURSOR_MAX_RADIUS = 44 // current (largest) circle size

export interface CursorBrush {
  active: boolean
  setActive: (active: boolean) => void
  radius: number
  setRadius: (radius: number) => void
}

/** Screen-space smooth-brush cursor: an on/off toggle plus its radius. */
export function useCursorBrush(): CursorBrush {
  const [active, setActive] = useState(false)
  const [radius, setRadius] = useState(CURSOR_MAX_RADIUS)
  return { active, setActive, radius, setRadius }
}
