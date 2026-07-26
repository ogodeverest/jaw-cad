// Selectable PBR material presets: albedo + metallic/roughness.
export type MaterialId =
  | 'sand'
  | 'resin'
  | 'gum'
  | 'gold'
  | 'silver'
  | 'chrome'
  | 'jade'

export interface Material {
  label: string
  color: [number, number, number]
  metallic: number
  roughness: number
}

export const MATERIAL_PRESETS: Record<MaterialId, Material> = {
  sand: { label: 'Sand', color: [0.8, 0.66, 0.46], metallic: 0, roughness: 0.55 },
  resin: { label: 'Blue resin', color: [0.45, 0.6, 0.95], metallic: 0, roughness: 0.35 },
  gum: { label: 'Gum pink', color: [0.85, 0.45, 0.5], metallic: 0, roughness: 0.5 },
  gold: { label: 'Gold', color: [1.0, 0.78, 0.34], metallic: 1, roughness: 0.25 },
  silver: { label: 'Silver', color: [0.95, 0.93, 0.88], metallic: 1, roughness: 0.2 },
  chrome: { label: 'Chrome', color: [0.95, 0.95, 0.97], metallic: 1, roughness: 0.08 },
  jade: { label: 'Jade', color: [0.35, 0.7, 0.5], metallic: 0, roughness: 0.3 },
}
