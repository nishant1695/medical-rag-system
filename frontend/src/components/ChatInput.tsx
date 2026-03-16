import { useRef, useEffect, KeyboardEvent } from 'react'
import { Send, Square } from 'lucide-react'
import { clsx } from 'clsx'

interface ChatInputProps {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onStop: () => void
  streaming: boolean
  disabled?: boolean
  placeholder?: string
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  onStop,
  streaming,
  disabled,
  placeholder = 'Ask about DIEP flap reconstruction, outcomes, complications…',
}: ChatInputProps) {
  const ref = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    if (!ref.current) return
    ref.current.style.height = 'auto'
    ref.current.style.height = Math.min(ref.current.scrollHeight, 200) + 'px'
  }, [value])

  const handleKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (streaming) return
      onSubmit()
    }
  }

  return (
    <div className="flex items-end gap-2 bg-white border border-slate-300 rounded-2xl px-4 py-3 shadow-sm focus-within:border-medical-400 focus-within:ring-2 focus-within:ring-medical-100 transition-all">
      <textarea
        ref={ref}
        rows={1}
        value={value}
        onChange={e => onChange(e.target.value)}
        onKeyDown={handleKey}
        disabled={disabled || streaming}
        placeholder={placeholder}
        className="flex-1 resize-none bg-transparent outline-none text-sm text-slate-800 placeholder-slate-400 leading-relaxed max-h-48 overflow-y-auto disabled:opacity-50"
      />
      {streaming ? (
        <button
          onClick={onStop}
          className="shrink-0 w-8 h-8 rounded-xl bg-red-100 hover:bg-red-200 flex items-center justify-center transition-colors"
          title="Stop generation"
        >
          <Square size={14} className="text-red-600" />
        </button>
      ) : (
        <button
          onClick={onSubmit}
          disabled={disabled || !value.trim()}
          className={clsx(
            'shrink-0 w-8 h-8 rounded-xl flex items-center justify-center transition-colors',
            value.trim()
              ? 'bg-medical-600 hover:bg-medical-700 text-white'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed',
          )}
          title="Send (Enter)"
        >
          <Send size={14} />
        </button>
      )}
    </div>
  )
}
