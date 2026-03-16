import { clsx } from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertCircle, Bot, User } from 'lucide-react'
import type { Message } from '../types'
import { EvidenceSummary } from './EvidenceBadge'
import { SafetyBanner } from './SafetyBanner'
import { CitationPanel } from './CitationPanel'

interface ChatMessageProps {
  message: Message
}

/** Replace inline [cid] with styled citation badges */
function CitedContent({ content }: { content: string }) {
  // Split on [xxxx] citation markers
  const parts = content.split(/(\[[a-z0-9]{4}\])/g)

  return (
    <div className="prose-chat text-sm text-slate-800 leading-relaxed">
      {parts.map((part, i) => {
        const m = part.match(/^\[([a-z0-9]{4})\]$/)
        if (m) {
          return (
            <span key={i} className="citation-badge mx-0.5" title={`Source ${m[1]}`}>
              {m[1]}
            </span>
          )
        }
        return (
          <ReactMarkdown key={i} remarkPlugins={[remarkGfm]}>
            {part}
          </ReactMarkdown>
        )
      })}
    </div>
  )
}

export function ChatMessageBubble({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="flex gap-3 justify-end">
        <div className="max-w-[80%] bg-medical-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed">
          {message.content}
        </div>
        <div className="shrink-0 w-7 h-7 rounded-full bg-medical-100 flex items-center justify-center mt-1">
          <User size={14} className="text-medical-700" />
        </div>
      </div>
    )
  }

  // Assistant bubble
  return (
    <div className="flex gap-3">
      <div className="shrink-0 w-7 h-7 rounded-full bg-emerald-100 flex items-center justify-center mt-1">
        <Bot size={14} className="text-emerald-700" />
      </div>

      <div className="flex-1 min-w-0 space-y-2">
        {/* Error state */}
        {message.error && (
          <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            {message.error}
          </div>
        )}

        {/* Main content bubble */}
        {message.content && (
          <div
            className={clsx(
              'bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm',
              message.streaming && 'typing-cursor',
            )}
          >
            <CitedContent content={message.content} />
          </div>
        )}

        {/* Streaming placeholder */}
        {message.streaming && !message.content && (
          <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
            <span className="typing-cursor text-slate-400 text-sm" />
          </div>
        )}

        {/* Safety banner */}
        {!message.streaming && message.safety_classification && message.safety_classification !== 'literature' && (
          <SafetyBanner classification={message.safety_classification} />
        )}

        {/* Evidence summary */}
        {!message.streaming && Object.keys(message.evidence_summary).length > 0 && (
          <EvidenceSummary summary={message.evidence_summary} />
        )}

        {/* Citations */}
        {!message.streaming && message.sources.length > 0 && (
          <CitationPanel sources={message.sources} />
        )}
      </div>
    </div>
  )
}
