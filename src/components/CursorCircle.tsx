import { useEffect, useRef } from 'react'

export function CursorCircle({ radius }: { radius: number }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const el = ref.current
      if (el) el.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`
    }
    window.addEventListener('pointermove', onMove)
    return () => window.removeEventListener('pointermove', onMove)
  }, [])

  const size = radius * 2 + 4
  return (
    <div ref={ref} className="cursor-circle" aria-hidden="true">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#7aa2ff"
          strokeWidth="2"
        />
      </svg>
    </div>
  )
}
