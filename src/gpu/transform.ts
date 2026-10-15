import { mat4 } from "wgpu-matrix";

export type Vec3 = [number, number, number];

// This function builds a TRS system of transforms
// Translation * Rotation (Euler X/Y/Z) * Scale
// Matrix multiplication happens in order from right to left
export function buildModelMatrix(
  out: Float32Array,
  offset: Vec3,
  rotation: Vec3,
  scale: Vec3,
): void {
  mat4.identity(out);
  // Last goes the translation matrix set by offset x,y,z
  mat4.translate(out, offset, out);
  // Second we compose the Rotation matrix from Euler angles in order (X,Y,Z)
  // Rotation around X goes on last
  mat4.rotateX(out, rotation[0], out);
  // Rotation around Y happens second with respect to X rotation (gimbal lock happens here if Y rotation is 90deg and Z and X align, we loose a degree of rotation here)
  mat4.rotateY(out, rotation[1], out);
  // Rotation around Z happens first with respect to Y rotation
  mat4.rotateZ(out, rotation[2], out);
  // First Scale is applied from x,y,z scale values
  mat4.scale(out, scale, out);
}
