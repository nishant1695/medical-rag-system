import { useState } from 'react'
import { clsx } from 'clsx'
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react'
import type { SourceChunk } from '../types'
import { EvidenceBadge } from './EvidenceBadge'

interface CitationPanelProps {
  sources: SourceChunk[]
  className?: string
}

export function CitationPanel({ sources, className }: CitationPanelProps) {
  const [expanded, setExpanded] = useState(false)
  const [openIdx, setOpenIdx] = useState<number | null>(null)

  if (!sources.length) return null

  return (
    <div className={clsx('rounded-lg border border-slate-200 bg-slate-50 text-xs', className)}>
      {/* Header */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-slate-100 rounded-t-lg transition-colors"
      >
        <span className="font-medium text-slate-700">
          {sources.length} source{sources.length !== 1 ? 's' : ''} cited
        </span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {/* Source list */}
      {expanded && (
        <ul className="divide-y divide-slate-200 border-t border-slate-200">
          {sources.map((src, i) => (
            <li key={src.chunk_id ?? i}>
              {/* Summary row */}
              <button
                onClick={() => setOpenIdx(openIdx === i ? null : i)}
                className="w-full flex items-start gap-2 px-3 py-2 text-left hover:bg-white transition-colors"
              >
                {/* Citation ID */}
                <span className="citation-badge shrink-0">{src.index}</span>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <EvidenceBadge level={src.evidence_level} />
                    {(src.paper_url || src.pmid) && (
                      <a
                        href={src.paper_url ?? `https://pubmed.ncbi.nlm.nih.gov/${src.pmid}/`}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={e => e.stopPropagation()}
                        className="flex items-center gap-0.5 text-medical-600 hover:underline font-mono"
                      >
                        {src.pmid ? `PMID ${src.pmid}` : 'View paper'}
                        <ExternalLink size={10} />
                      </a>
                    )}
                    <span className="text-slate-500">
                      score {src.score.toFixed(2)}
                    </span>
                  </div>
                  {src.heading_path?.length > 0 && (
                    <p className="text-slate-400 truncate mt-0.5">
                      {src.heading_path.join(' › ')}
                    </p>
                  )}
                </div>

                {openIdx === i ? <ChevronUp size={12} className="shrink-0 mt-0.5" /> : <ChevronDown size={12} className="shrink-0 mt-0.5" />}
              </button>

              {/* Expanded excerpt */}
              {openIdx === i && (
                <div className="px-3 pb-3 bg-white">
                  <p className="text-slate-700 leading-relaxed line-clamp-6 whitespace-pre-wrap">
                    {src.content}
                  </p>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
