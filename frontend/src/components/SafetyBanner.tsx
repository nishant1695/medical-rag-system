import { clsx } from 'clsx'
import { AlertTriangle, ShieldAlert, BookOpen } from 'lucide-react'
import type { SafetyClass } from '../types'

interface SafetyBannerProps {
  classification: SafetyClass
  className?: string
}

const config: Record<SafetyClass, {
  icon: React.ReactNode
  title: string
  message: string
  cls: string
}> = {
  literature: {
    icon: <BookOpen size={14} />,
    title: 'Literature Query',
    message: 'This is a summary of published research and should not replace clinical judgment.',
    cls: 'safety-literature',
  },
  patient_specific: {
    icon: <AlertTriangle size={14} />,
    title: 'Patient-Specific Query Detected',
    message:
      'This system provides literature summaries only. For individual patient decisions, '
      + 'please consult a qualified clinician.',
    cls: 'safety-patient_specific',
  },
  emergency: {
    icon: <ShieldAlert size={14} />,
    title: '⚠️  Emergency Detected',
    message:
      'If this is a medical emergency, call emergency services immediately (911 / 999 / 112). '
      + 'This AI cannot provide emergency guidance.',
    cls: 'safety-emergency',
  },
}

export function SafetyBanner({ classification, className }: SafetyBannerProps) {
  const c = config[classification]
  return (
    <div className={clsx('flex items-start gap-2 px-3 py-2 rounded border text-xs', c.cls, className)}>
      <span className="mt-0.5 shrink-0">{c.icon}</span>
      <div>
        <span className="font-semibold">{c.title}: </span>
        {c.message}
      </div>
    </div>
  )
}
