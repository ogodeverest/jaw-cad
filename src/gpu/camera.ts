import { mat4 } from "wgpu-matrix";

// Type of projection: 'perspective' | 'orthographic'
export type Projection = "perspective" | "orthographic";

// State of camera orbit values
// Position of camera in a sphere with a certain radius
export interface OrbitState {
  azimuth: number; // camera y-rotation around the target (horizontal rotation)
  polar: number; // camera x-rotation around the target (tilt up/down)
  radius: number; // distance of camera from target
}

// Default camera orbit values
export const DEFAULT_ORBIT: OrbitState = {
  azimuth: 0.6,
  polar: 0.5,
  radius: 3.2,
};

// position and size of viewport where we write framebuffer
// x,y offset from top left corner, width and height size in device pixels
export interface Viewport {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Default perspective camera parameters
const FOV_Y = (50 * Math.PI) / 180;
const NEAR = 0.1;
const FAR = 100;

// Function takes:
// proj destination (if perspective): projection matrix composed by multiplication by:
//   1. center translation matrix
//   2. near value in x and y as part of similar triangles calculation (scale matrix)
//   3. matrix that maps z value non-linearly from near to far to 0 - 1 NDC coordinates (also sends z down at w-component for perspective divide next as part of similar triangles equation)
//   4. last matrix is a scale matrix that will fit the near plane from width-height to -1 to 1 NDC coordinates
//
//   scale-to-NDC * near-scale * z-warp+w * center-translation * p:
//   r,l,t,b are frustum bounds, c1,c2 constants for remap depth (from near/far)
//
//   | 2/(r-l)  0        0   0 |   | near  0     0  0 |   | 1  0  0    0   |   | 1  0  0  -(l+r)/2 |   | x |   | x' |
//   | 0        2/(t-b)  0   0 | * | 0     near  0  0 | * | 0  1  0    0   | * | 0  1  0  -(b+t)/2 | * | y | = | y' |
//   | 0        0        1   0 |   | 0     0     1  0 |   | 0  0  -c2  c1  |   | 0  0  1     0     |   | z |   | z' |
//   | 0        0        0   1 |   | 0     0     0  1 |   | 0  0  -1   0   |   | 0  0  0     1     |   | 1 |   | w' |

// proj destination (if orthographic): projection matrix composed by multiplication by:
//   1. center translation matrix
//   2. a scale matrix that will fit the near plane from width-height to -1 to 1 NDC coordinates
//   3. change of basis matrix from right handed coordinate system to left handed NDC (z -> 0-1)
//
//   change of basis flip * scale * center translation * p:
//   sx,sy,sz = scale_x,scale_y,scale_z ; mid = box center
//
//   | 1  0   0  0 |   | sx  0   0   0 |   | 1  0  0  -mid_x |   | x |   | x' |
//   | 0  1   0  0 | * | 0   sy  0   0 | * | 0  1  0  -mid_y | * | y | = | y' |
//   | 0  0  -1  0 |   | 0   0   sz  0 |   | 0  0  1  -mid_z |   | z |   | z' |
//   | 0  0   0  1 |   | 0   0   0   1 |   | 0  0  0    1    |   | 1 |   | w' |

//   camera transform matrix: basis vectors i, j, k in the
//   columns, camera position p in the last column:
//   rotation of camera (i,j,k) and position (p) are set from the camera orbiting + translation
//
//   | ix  jx  kx  px |     i = camera right   (world space)
//   | iy  jy  ky  py |     j = camera up      (world space)
//   | iz  jz  kz  pz |     k = camera forward (world space)
//   |  0   0   0   1 |     p = camera position
//
//   Ry(t) * Rx(f) * Translate(0,0,d),
//   azimuth, polar, radius. The rotations make it so the camera has the right orientation to look at the target
//   and position is correct.
//
//   | cy  0 sy 0 |   | 1 0  0   0 |   | 1  0  0  0 |   | ix jx kx px |
//   | 0   1 0  0 | * | 0 cx -sx 0 | * | 0  1  0  0 | = | iy jy ky py |
//   | -sy 0 cy 0 |   | 0 sx cx  0 |   | 0  0  1  d |   | iz jz kz pz |
//   | 0   0 0  1 |   | 0 0  0   1 |   | 0  0  0  1 |   | 0  0  0  1  |
//
//   view = inverse of the above.

// viewProj destination: projectionMatrix x viewMatrix
//   P = proj matrix (perspective or ortho);
//   V = view = inverse(camera->world)
//
//           | x |   | xc |
//   P * V * | y | = | yc |
//           | z |   | zc |
//           | 1 |   | wc |
// this outputs the vertex is clip-space after w division it will be in NDC (z 0-1, x,y -1,1)

// orbit -> state of orbit (azimuth, polar, radius)
// aspect -> width / height of viewport
export function setOrbitCamera(
  proj: Float32Array,
  view: Float32Array,
  viewProj: Float32Array,
  projection: Projection,
  orbit: OrbitState,
  aspect: number,
): void {
  if (projection === "perspective") {
    mat4.perspective(FOV_Y, aspect, NEAR, FAR, proj);
  } else {
    const halfH = orbit.radius * 0.38;
    const halfW = halfH * aspect;
    mat4.ortho(-halfW, halfW, -halfH, halfH, NEAR, FAR, proj);
  }

  mat4.identity(view);
  mat4.translate(view, [0, 0, -orbit.radius], view);
  mat4.rotateX(view, orbit.polar, view);
  mat4.rotateY(view, orbit.azimuth, view);
  mat4.multiply(proj, view, viewProj);
}
