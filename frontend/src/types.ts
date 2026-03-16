// ─── API Types ─────────────────────────────────────────────────────────────────

export interface Workspace {
  id: number
  name: string
  description?: string
  subspecialty?: string
  created_at: string
  document_count: number
}

export interface Document {
  id: number
  workspace_id: number
  original_filename: string
  file_type?: string
  status: 'pending' | 'parsing' | 'indexing' | 'indexed' | 'failed'
  page_count: number
  chunk_count: number
  pmid?: string
  title?: string
  authors?: string[]
  journal?: string
  publication_year?: number
  study_design?: string
  evidence_level?: 'I' | 'II' | 'III' | 'IV' | 'V'
  sample_size?: number
  created_at: string
}

export interface SourceChunk {
  index: string          // citation ID e.g. "a3x9"
  chunk_id: string
  content: string
  document_id: number
  page_no: number
  heading_path: string[]
  score: number
  pmid?: string
  evidence_level?: string
}

export type SafetyClass = 'literature' | 'patient_specific' | 'emergency'

export interface ChatResponse {
  answer: string
  sources: SourceChunk[]
  safety_classification: SafetyClass
  evidence_summary: Record<string, number>
  thinking?: string
}

export interface SearchResponse {
  query: string
  chunks: SourceChunk[]
  evidence_summary: Record<string, number>
}

export interface WorkspaceStats {
  workspace_id: number
  chunk_count: number
  name: string
  subspecialty?: string
}

// ─── UI Types ──────────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant'

export interface Message {
  id: string
  role: MessageRole
  content: string        // final streamed text
  sources: SourceChunk[]
  evidence_summary: Record<string, number>
  safety_classification?: SafetyClass
  thinking?: string
  streaming?: boolean    // true while tokens are arriving
  error?: string
}

export type Tab = 'chat' | 'search' | 'documents'
