# jaw-scan-cad

A small WebGPU viewer for intraoral (dental) scans. Load an upper and lower jaw,
look at them from a few angles, change the material, and smooth rough spots with
a brush.

## Sample models

A couple of `.ply` files live under `src/models/` if you want something to load:

- `intra-oral/upper.ply` and `intra-oral/lower.ply` — a real scan split into the two jaws
- `dragon.ply` — not a jaw, just handy for testing the renderer

Click **Upload .ply** under Upper jaw / Lower jaw and pick one. Only `.ply` is accepted.

## What you can do

- Load an upper and a lower jaw separately, and show/hide each.
- Orbit by dragging inside a viewport, zoom with the scroll wheel.
- Switch projection per jaw (perspective / orthographic).
- Pick a material — sand, blue resin, gum, gold, silver, chrome, jade.
- Transform the selected jaw: translation, rotation (in degrees), scale, plus a reset.
- Smooth with the brush: turn it on, set a radius, then drag over the mesh. It
  smooths whatever lands inside the circle on screen.

The visible jaws stack on the left (lower looking down from above, upper looking
up from below), with a combined perspective view on the right.
