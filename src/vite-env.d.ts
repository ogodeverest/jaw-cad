/// <reference types="vite/client" />

// WGSL shaders are imported as raw strings via Vite's `?raw` suffix,
// e.g. `import shader from './shader.wgsl?raw'`.
declare module '*.wgsl?raw' {
  const src: string
  export default src
}

// Bare `.wgsl` imports also resolve to their source text.
declare module '*.wgsl' {
  const src: string
  export default src
}

// PLY models imported as a served URL via Vite's `?url` suffix.
declare module '*.ply?url' {
  const src: string
  export default src
}
