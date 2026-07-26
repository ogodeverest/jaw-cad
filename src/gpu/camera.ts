import { mat4 } from 'wgpu-matrix'

export type Projection = 'perspective' | 'orthographic'

/** Orbit camera around the origin, in spherical-ish terms. */
export interface OrbitState {
  azimuth: number // radians, around +Y
  polar: number // radians, tilt up/down (clamp to avoid gimbal flip)
  radius: number // distance from target
}

export const DEFAULT_ORBIT: OrbitState = {
  azimuth: 0.6,
  polar: 0.5,
  radius: 3.2, // dollied in near the model along the view (z)
}

/** A framebuffer-pixel rectangle, origin at the top-left of the canvas. */
export interface Viewport {
  x: number
  y: number
  width: number
  height: number
}

const FOV_Y = (50 * Math.PI) / 180
const NEAR = 0.1
const FAR = 100

/**
 * Writes `proj`, `view`, and `viewProj` for an orbit camera into the provided
 * (reused) matrices. Orthographic ties its box to the orbit radius so the wheel
 * zooms in both modes.
 */
export function setOrbitCamera(
  proj: Float32Array,
  view: Float32Array,
  viewProj: Float32Array,
  projection: Projection,
  orbit: OrbitState,
  aspect: number,
): void {
  if (projection === 'perspective') {
    mat4.perspective(FOV_Y, aspect, NEAR, FAR, proj)
  } else {
    const halfH = orbit.radius * 0.38 // ortho box (radius 3.2 → ~1.2 half-height)
    const halfW = halfH * aspect
    mat4.ortho(-halfW, halfW, -halfH, halfH, NEAR, FAR, proj)
  }
  // Orbit view: pull back by radius, tilt (polar, X), spin (azimuth, Y).
  mat4.identity(view)
  mat4.translate(view, [0, 0, -orbit.radius], view)
  mat4.rotateX(view, orbit.polar, view)
  mat4.rotateY(view, orbit.azimuth, view)
  mat4.multiply(proj, view, viewProj)
}
