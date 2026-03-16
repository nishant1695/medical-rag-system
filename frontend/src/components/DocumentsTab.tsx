import { useState, useEffect, useRef } from 'react'
import { FileText, Trash2, ExternalLink, RefreshCw, Upload } from 'lucide-react'
import { clsx } from 'clsx'
import { api } from '../api'
import type { Document } from '../types'
import { EvidenceBadge } from './EvidenceBadge'

interface DocumentsTabProps {
  workspaceId: number
}

const statusColors: Record<string, string> = {
  pending:  'bg-slate-100 text-slate-600',
  parsing:  'bg-blue-100 text-blue-700',
  indexing: 'bg-amber-100 text-amber-700',
  indexed:  'bg-emerald-100 text-emerald-700',
  failed:   'bg-red-100 text-red-700',
}

export function DocumentsTab({ workspaceId }: DocumentsTabProps) {
  const [docs, setDocs] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const list = await api.documents.list(workspaceId)
      setDocs(list)
    } catch (e: unknown) {
      setError((e as Error).message ?? 'Failed to load documents')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [workspaceId])

  const handleDelete = async (docId: number) => {
    if (!confirm('Delete this document from the knowledge base?')) return
    try {
      await api.documents.delete(workspaceId, docId)
      setDocs(d => d.filter(x => x.id !== docId))
    } catch (e: unknown) {
      alert((e as Error).message ?? 'Delete failed')
    }
  }

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      const doc = await api.documents.upload(workspaceId, file)
      setDocs(d => [doc, ...d])
    } catch (e: unknown) {
      alert((e as Error).message ?? 'Upload failed')
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm text-slate-600">
          <span className="font-medium">{docs.length}</span> documents indexed
        </p>
        <div className="flex gap-2">
          <button
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors"
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-medical-600 hover:bg-medical-700 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <Upload size={12} />
            {uploading ? 'Uploading…' : 'Upload PDF'}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            onChange={handleUpload}
            className="hidden"
          />
        </div>
      </div>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-32 text-slate-400 text-sm">
            Loading documents…
          </div>
        ) : docs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-slate-400 gap-2">
            <FileText size={24} />
            <p className="text-sm">No documents yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            {docs.map(doc => (
              <div
                key={doc.id}
                className="bg-white border border-slate-200 rounded-xl p-3 shadow-sm"
              >
                <div className="flex items-start justify-between gap-2">
                  {/* Left: metadata */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <span
                        className={clsx(
                          'text-xs px-1.5 py-0.5 rounded font-medium',
                          statusColors[doc.status] ?? statusColors.pending,
                        )}
                      >
                        {doc.status}
                      </span>
                      <EvidenceBadge level={doc.evidence_level} />
                      {(doc.paper_url || doc.pmid) && (
                        <a
                          href={doc.paper_url ?? `https://pubmed.ncbi.nlm.nih.gov/${doc.pmid}/`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-0.5 text-xs text-medical-600 hover:underline font-mono"
                        >
                          {doc.pmid ? `PMID ${doc.pmid}` : 'View paper'}
                          <ExternalLink size={10} />
                        </a>
                      )}
                    </div>

                    <p className="text-sm font-medium text-slate-800 truncate">
                      {doc.title ?? doc.original_filename}
                    </p>

                    <div className="mt-1 flex items-center gap-3 text-xs text-slate-500 flex-wrap">
                      {doc.authors && doc.authors.length > 0 && (
                        <span>{doc.authors.slice(0, 3).join(', ')}{doc.authors.length > 3 ? ' et al.' : ''}</span>
                      )}
                      {doc.journal && <span>{doc.journal}</span>}
                      {doc.publication_year && <span>{doc.publication_year}</span>}
                      {doc.study_design && (
                        <span className="capitalize">{doc.study_design.replace(/_/g, ' ')}</span>
                      )}
                      <span>{doc.chunk_count} chunks</span>
                    </div>
                  </div>

                  {/* Right: actions */}
                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="shrink-0 p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    title="Delete document"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
