import { useState } from 'react'
import { EMPTY_SLOT, type JawKey, type SlotState, type Vec3 } from './scene/types'
import { useScene } from './scene/useScene'
import { JawControl } from './components/JawControl'
import { ToolsPanel } from './components/ToolsPanel'
import { Stage } from './components/Stage'
import { CursorCircle } from './components/CursorCircle'
import { useCursorBrush } from './components/cursorBrush'
import './App.css'

function App() {
  const [slots, setSlots] = useState<Record<JawKey, SlotState>>({
    upper: { ...EMPTY_SLOT, material: 'sand' },
    lower: { ...EMPTY_SLOT, material: 'resin' },
  })

  // Which jaw the Tools section acts on.
  const [selectedJaw, setSelectedJaw] = useState<JawKey>('upper')

  const brush = useCursorBrush()

  const patchSlot = (key: JawKey, patch: Partial<SlotState>) =>
    setSlots((prev) => ({ ...prev, [key]: { ...prev[key], ...patch } }))

  // Reset the selected jaw's transform back to identity.
  const resetTransform = () =>
    patchSlot(selectedJaw, {
      translation: [0, 0, 0],
      rotation: [0, 0, 0],
      scale: [1, 1, 1],
    })

  // Update one axis of the selected jaw's translation, rotation, or scale.
  const setVecAxis = (
    field: 'translation' | 'rotation' | 'scale',
    axis: number,
    value: number,
  ) => {
    if (Number.isNaN(value)) return
    setSlots((prev) => {
      const next = [...prev[selectedJaw][field]] as Vec3
      next[axis] = value
      return {
        ...prev,
        [selectedJaw]: { ...prev[selectedJaw], [field]: next },
      }
    })
  }

  const { canvasRef, status, error, onPointerDown, onWheel, handleFile } =
    useScene({
      slots,
      patchSlot,
      brushActive: brush.active,
      brushRadius: brush.radius,
    })

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar__title">jaw-scan-cad</div>

        <JawControl
          label="Upper jaw"
          slot={slots.upper}
          onUpload={(file) => handleFile('upper', file)}
          onToggle={() => patchSlot('upper', { visible: !slots.upper.visible })}
          onProjection={(p) => patchSlot('upper', { projection: p })}
          onMaterial={(m) => patchSlot('upper', { material: m })}
        />
        <JawControl
          label="Lower jaw"
          slot={slots.lower}
          onUpload={(file) => handleFile('lower', file)}
          onToggle={() => patchSlot('lower', { visible: !slots.lower.visible })}
          onProjection={(p) => patchSlot('lower', { projection: p })}
          onMaterial={(m) => patchSlot('lower', { material: m })}
        />

        <ToolsPanel
          selectedJaw={selectedJaw}
          onSelectJaw={setSelectedJaw}
          slot={slots[selectedJaw]}
          brush={brush}
          onVecAxis={setVecAxis}
          onResetTransform={resetTransform}
        />
      </aside>

      <Stage
        canvasRef={canvasRef}
        slots={slots}
        status={status}
        error={error}
        onPointerDown={onPointerDown}
        onWheel={onWheel}
      />

      {brush.active && <CursorCircle radius={brush.radius} />}
    </div>
  )
}

export default App
