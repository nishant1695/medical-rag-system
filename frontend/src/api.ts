/**
 * API client for Medical RAG backend.
 * Base URL is proxied through Vite dev server (/api → http://localhost:8000/api).
 */

const BASE = '/api/v1'

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body?.detail ?? JSON.stringify(body)
    } catch { /* ignore */ }
    throw new Error(detail)
  }
  return res.json()
}

// ─── Workspaces ────────────────────────────────────────────────────────────────

import type { Workspace, Document, SearchResponse, WorkspaceStats } from './types'

export const api = {
  workspaces: {
    list: (): Promise<Workspace[]> => req('/workspaces'),
    get:  (id: number): Promise<Workspace> => req(`/workspaces/${id}`),
    create: (payload: { name: string; description?: string; subspecialty?: string }): Promise<Workspace> =>
      req('/workspaces', { method: 'POST', body: JSON.stringify(payload) }),
    delete: (id: number): Promise<void> => req(`/workspaces/${id}`, { method: 'DELETE' }),
  },

  documents: {
    list: (workspaceId: number): Promise<Document[]> =>
      req(`/workspaces/${workspaceId}/documents`),
    delete: (workspaceId: number, docId: number): Promise<void> =>
      req(`/workspaces/${workspaceId}/documents/${docId}`, { method: 'DELETE' }),
    upload: (workspaceId: number, file: File, pmid?: string): Promise<Document> => {
      const form = new FormData()
      form.append('file', file)
      if (pmid) form.append('pmid', pmid)
      return fetch(`${BASE}/workspaces/${workspaceId}/documents/upload`, {
        method: 'POST',
        body: form,
      }).then(async r => {
        if (!r.ok) throw new Error((await r.json())?.detail ?? `HTTP ${r.status}`)
        return r.json()
      })
    },
  },

  chat: {
    stats: (workspaceId: number): Promise<WorkspaceStats> =>
      req(`/workspaces/${workspaceId}/stats`),

    search: (workspaceId: number, query: string, topK = 5): Promise<SearchResponse> =>
      req(`/workspaces/${workspaceId}/search`, {
        method: 'POST',
        body: JSON.stringify({ query, top_k: topK }),
      }),
  },
}

// ─── SSE streaming chat ────────────────────────────────────────────────────────

export interface SSECallbacks {
  onText:     (delta: string) => void
  onThinking: (text: string) => void
  onSources:  (sources: import('./types').SourceChunk[]) => void
  onDone:     (data: { safety_classification: string; evidence_summary: Record<string, number> }) => void
  onError:    (msg: string) => void
}

/**
 * Opens a streaming chat request and fires callbacks for each SSE event.
 * Returns an AbortController so callers can cancel mid-stream.
 */
export function streamChat(
  workspaceId: number,
  message: string,
  history: { role: string; content: string }[],
  callbacks: SSECallbacks,
  opts: { enableThinking?: boolean; forceSearch?: boolean } = {},
): AbortController {
  const controller = new AbortController()

  ;(async () => {
    let res: Response
    try {
      res = await fetch(`${BASE}/workspaces/${workspaceId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          history,
          enable_thinking: opts.enableThinking ?? false,
          force_search: opts.forceSearch ?? true,
        }),
        signal: controller.signal,
      })
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        callbacks.onError((err as Error).message ?? 'Network error')
      }
      return
    }

    if (!res.ok) {
      let detail = `HTTP ${res.status}`
      try { detail = (await res.json())?.detail ?? detail } catch { /* */ }
      callbacks.onError(detail)
      return
    }

    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buf = ''

    while (true) {
      let done = false
      let value: Uint8Array | undefined
      try {
        ;({ done, value } = await reader.read())
      } catch {
        break
      }
      if (done) break

      buf += decoder.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''

      let currentEvent = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          const rawData = line.slice(6)
          try {
            const data = JSON.parse(rawData)
            switch (currentEvent) {
              case 'text':
                callbacks.onText(data.delta ?? '')
                break
              case 'thinking':
                callbacks.onThinking(data.text ?? '')
                break
              case 'sources':
                callbacks.onSources(data.sources ?? [])
                break
              case 'done':
                callbacks.onDone({
                  safety_classification: data.safety_classification ?? 'literature',
                  evidence_summary: data.evidence_summary ?? {},
                })
                break
              case 'error':
                callbacks.onError(data.message ?? 'Unknown error')
                break
            }
          } catch { /* malformed JSON, skip */ }
          currentEvent = ''
        }
      }
    }
  })()

  return controller
}
