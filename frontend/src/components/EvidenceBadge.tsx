import { clsx } from 'clsx'

interface EvidenceBadgeProps {
  level?: string
  className?: string
}

const levelLabels: Record<string, string> = {
  I:   'Meta-analysis / RCT',
  II:  'Prospective Cohort',
  III: 'Retrospective Cohort',
  IV:  'Case Series',
  V:   'Expert Opinion',
}

export function EvidenceBadge({ level, className }: EvidenceBadgeProps) {
  if (!level) return null
  return (
    <span
      title={levelLabels[level] ?? `Level ${level}`}
      className={clsx(
        'inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-xs font-semibold',
        `ev-${level}`,
        className,
      )}
    >
      Lvl {level}
    </span>
  )
}

interface EvidenceSummaryProps {
  summary: Record<string, number>
  className?: string
}

export function EvidenceSummary({ summary, className }: EvidenceSummaryProps) {
  const entries = Object.entries(summary).sort(([a], [b]) => a.localeCompare(b))
  if (!entries.length) return null

  return (
    <div className={clsx('flex flex-wrap gap-1.5 items-center', className)}>
      <span className="text-xs text-slate-500 mr-1">Evidence:</span>
      {entries.map(([level, count]) => (
        <span
          key={level}
          title={levelLabels[level] ?? `Level ${level}`}
          className={clsx(
            'inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium',
            `ev-${level}`,
          )}
        >
          Lvl {level} × {count}
        </span>
      ))}
    </div>
  )
}
