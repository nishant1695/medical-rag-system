import { useState } from 'react'
import { Search, ExternalLink } from 'lucide-react'
import { clsx } from 'clsx'
import { api } from '../api'
import type { SearchResponse } from '../types'
import { EvidenceBadge, EvidenceSummary } from './EvidenceBadge'

interface SearchTabProps {
  workspaceId: number
}

export function SearchTab({ workspaceId }: SearchTabProps) {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.chat.search(workspaceId, query.trim(), topK)
      setResult(res)
    } catch (e: unknown) {
      setError((e as Error).message ?? 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Search bar */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            placeholder="Search the literature, e.g. 'DIEP flap complications'"
            className="w-full pl-9 pr-4 py-2.5 border border-slate-300 rounded-xl text-sm focus:outline-none focus:border-medical-400 focus:ring-2 focus:ring-medical-100"
          />
        </div>
        <select
          value={topK}
          onChange={e => setTopK(Number(e.target.value))}
          className="border border-slate-300 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:border-medical-400 bg-white"
        >
          {[3, 5, 8, 10].map(n => (
            <option key={n} value={n}>Top {n}</option>
          ))}
        </select>
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className={clsx(
            'px-5 py-2.5 rounded-xl text-sm font-medium transition-colors',
            query.trim() && !loading
              ? 'bg-medical-600 hover:bg-medical-700 text-white'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed',
          )}
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="flex-1 overflow-y-auto space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm text-slate-600">
              <span className="font-medium">{result.chunks.length}</span> results for{' '}
              <em>"{result.query}"</em>
            </p>
            <EvidenceSummary summary={result.evidence_summary} />
          </div>

          {result.chunks.map((chunk, i) => (
            <div
              key={chunk.chunk_id ?? i}
              className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-2"
            >
              {/* Header */}
              <div className="flex items-start justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="citation-badge">{chunk.index}</span>
                  <EvidenceBadge level={chunk.evidence_level} />
                  {(chunk.paper_url || chunk.pmid) && (
                    <a
                      href={chunk.paper_url ?? `https://pubmed.ncbi.nlm.nih.gov/${chunk.pmid}/`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-0.5 text-xs text-medical-600 hover:underline font-mono"
                    >
                      {chunk.pmid ? `PMID ${chunk.pmid}` : 'View paper'}
                      <ExternalLink size={10} />
                    </a>
                  )}
                </div>
                <span className="text-xs text-slate-400">
                  score {chunk.score.toFixed(3)}
                </span>
              </div>

              {/* Heading path */}
              {chunk.heading_path?.length > 0 && (
                <p className="text-xs text-slate-400">
                  {chunk.heading_path.join(' › ')}
                </p>
              )}

              {/* Content */}
              <p className="text-sm text-slate-700 leading-relaxed line-clamp-4">
                {chunk.content}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
          Enter a query above to search the breast surgery knowledge base
        </div>
      )}
    </div>
  )
}
