import { useState, useRef, useEffect, useCallback } from 'react'
import { Trash2 } from 'lucide-react'
import { nanoid } from '../utils/nanoid'
import { streamChat, api } from '../api'
import type { Message, SourceChunk } from '../types'
import { ChatMessageBubble } from '../components/ChatMessage'
import { ChatInput } from '../components/ChatInput'

interface ChatPageProps {
  workspaceId: number
}

const SUGGESTED = [
  'What are the complication rates for bilateral DIEP flap reconstruction?',
  'Compare DIEP flap vs TRAM flap outcomes',
  'What factors predict free flap failure?',
  'What is the evidence for immediate vs delayed breast reconstruction?',
]

export function ChatPage({ workspaceId }: ChatPageProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Load persisted chat history when workspace changes
  useEffect(() => {
    setMessages([])
    setLoadingHistory(true)
    api.chat.history(workspaceId)
      .then(rows => {
        const restored: Message[] = rows.map(r => ({
          id: r.message_id,
          role: r.role as Message['role'],
          content: r.content,
          sources: (r.sources as SourceChunk[]) ?? [],
          evidence_summary: r.evidence_quality ?? {},
          safety_classification: r.safety_classification as Message['safety_classification'],
          thinking: r.thinking ?? undefined,
        }))
        setMessages(restored)
      })
      .catch(() => { /* ignore — history is non-critical */ })
      .finally(() => setLoadingHistory(false))
  }, [workspaceId])

  // Scroll to bottom on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleClearHistory = async () => {
    if (!confirm('Clear conversation history for this workspace?')) return
    await api.chat.clearHistory(workspaceId).catch(() => {})
    setMessages([])
  }

  const updateLastMessage = useCallback((updater: (m: Message) => Message) => {
    setMessages(prev => {
      if (!prev.length) return prev
      const last = prev[prev.length - 1]
      return [...prev.slice(0, -1), updater(last)]
    })
  }, [])

  const handleSubmit = useCallback(() => {
    const text = input.trim()
    if (!text || streaming) return

    // Add user message
    const userMsg: Message = {
      id: nanoid(),
      role: 'user',
      content: text,
      sources: [],
      evidence_summary: {},
    }

    // Placeholder assistant message
    const assistantMsg: Message = {
      id: nanoid(),
      role: 'assistant',
      content: '',
      sources: [],
      evidence_summary: {},
      streaming: true,
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setInput('')
    setStreaming(true)

    // Build history from prior turns only (current question is sent as `message`)
    const history = messages.map(m => ({
      role: m.role,
      content: m.content,
    }))

    // Collect sources for final attachment
    let pendingSources: SourceChunk[] = []

    abortRef.current = streamChat(
      workspaceId,
      text,
      history,
      {
        onText(delta) {
          updateLastMessage(m => ({ ...m, content: m.content + delta }))
        },
        onThinking(t) {
          updateLastMessage(m => ({ ...m, thinking: (m.thinking ?? '') + t }))
        },
        onSources(sources) {
          pendingSources = sources
        },
        onDone({ safety_classification, evidence_summary, specialist_contexts }) {
          updateLastMessage(m => ({
            ...m,
            streaming: false,
            sources: pendingSources,
            safety_classification: safety_classification as Message['safety_classification'],
            evidence_summary,
            specialist_contexts,
          }))
          setStreaming(false)
        },
        onError(msg) {
          updateLastMessage(m => ({
            ...m,
            streaming: false,
            error: msg,
            sources: pendingSources,
          }))
          setStreaming(false)
        },
      },
    )
  }, [input, streaming, messages, workspaceId, updateLastMessage])

  const handleStop = () => {
    abortRef.current?.abort()
    updateLastMessage(m => ({ ...m, streaming: false }))
    setStreaming(false)
  }

  const handleSuggestion = (s: string) => {
    setInput(s)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {loadingHistory ? (
          <div className="flex items-center justify-center h-full text-slate-400 text-sm">
            Loading conversation…
          </div>
        ) : messages.length === 0 ? (
          /* Welcome / empty state */
          <div className="flex flex-col items-center justify-center h-full gap-6 text-center pb-16">
            <div>
              <h2 className="text-xl font-semibold text-slate-800 mb-1">
                Breast Surgery Research Assistant
              </h2>
              <p className="text-sm text-slate-500 max-w-md">
                Ask clinical questions and get evidence-based answers with PMID citations from 50 peer-reviewed papers.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl">
              {SUGGESTED.map(s => (
                <button
                  key={s}
                  onClick={() => handleSuggestion(s)}
                  className="text-left px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-xs text-slate-700 hover:border-medical-300 hover:bg-medical-50 transition-colors shadow-sm"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map(msg => (
            <ChatMessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-slate-200 bg-white px-4 py-3">
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          onStop={handleStop}
          streaming={streaming}
        />
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs text-slate-400">
            For research only — not clinical advice. Always apply professional judgment.
          </p>
          {messages.length > 0 && !streaming && (
            <button
              onClick={handleClearHistory}
              className="flex items-center gap-1 text-xs text-slate-400 hover:text-red-500 transition-colors"
              title="Clear conversation history"
            >
              <Trash2 size={11} />
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
