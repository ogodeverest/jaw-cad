import { mat4 } from 'wgpu-matrix'

export type Vec3 = [number, number, number]

/**
 * Builds a TRS model matrix `T * Rx * Ry * Rz * S` (Three.js "XYZ" Euler order)
 * into the provided (reused) matrix. Rotation is in radians.
 */
export function buildModelMatrix(
  out: Float32Array,
  offset: Vec3,
  rotation: Vec3,
  scale: Vec3,
): void {
  mat4.identity(out)
  mat4.translate(out, offset, out)
  mat4.rotateX(out, rotation[0], out)
  mat4.rotateY(out, rotation[1], out)
  mat4.rotateZ(out, rotation[2], out)
  mat4.scale(out, scale, out)
}
