import type { Projection } from '../gpu/mesh'
import type { MaterialId } from './materials'

export type Vec3 = [number, number, number]

export type JawKey = 'upper' | 'lower'
export type ViewKey = JawKey | 'combined'
export type SlotStatus = 'empty' | 'loading' | 'ready' | 'error'

export interface SlotState {
  name: string | null
  visible: boolean
  status: SlotStatus
  projection: Projection
  material: MaterialId
  translation: Vec3
  rotation: Vec3
  scale: Vec3
}

export const EMPTY_SLOT: SlotState = {
  name: null,
  visible: false,
  status: 'empty',
  projection: 'orthographic',
  material: 'sand',
  translation: [0, 0, 0],
  rotation: [0, 0, 0],
  scale: [1, 1, 1],
}

export const DEG2RAD = Math.PI / 180

export const LEFT_COL = 0.4
export const LEFT_PCT = `${LEFT_COL * 100}%`
export const RIGHT_PCT = `${(1 - LEFT_COL) * 100}%`

export const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v))
