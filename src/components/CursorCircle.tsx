import { useEffect, useRef } from 'react'

/**
 * The floating brush circle that follows the pointer. Moved via a ref (not
 * React state) to avoid re-rendering on every pointer move.
 */
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

  const size = radius * 2 + 4 // +stroke padding
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
