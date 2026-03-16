import { useState, useEffect } from 'react'
import { MessageSquare, Search, FileText, Activity, ChevronLeft, ChevronRight, Stethoscope } from 'lucide-react'
import { clsx } from 'clsx'
import { api } from './api'
import type { Tab, Workspace, WorkspaceStats } from './types'
import { ChatPage } from './pages/ChatPage'
import { SearchTab } from './components/SearchTab'
import { DocumentsTab } from './components/DocumentsTab'

const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'chat',      label: 'Chat',      icon: <MessageSquare size={16} /> },
  { id: 'search',    label: 'Search',    icon: <Search size={16} /> },
  { id: 'documents', label: 'Documents', icon: <FileText size={16} /> },
]

export default function App() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([])
  const [activeWs, setActiveWs] = useState<number | null>(null)
  const [stats, setStats] = useState<WorkspaceStats | null>(null)
  const [tab, setTab] = useState<Tab>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [loadingWs, setLoadingWs] = useState(true)

  // Load workspaces on mount
  useEffect(() => {
    api.workspaces.list()
      .then(ws => {
        setWorkspaces(ws)
        if (ws.length > 0) setActiveWs(ws[0].id)
      })
      .catch(console.error)
      .finally(() => setLoadingWs(false))
  }, [])

  // Load stats when workspace changes
  useEffect(() => {
    if (!activeWs) return
    api.chat.stats(activeWs)
      .then(setStats)
      .catch(console.error)
  }, [activeWs])

  const currentWs = workspaces.find(w => w.id === activeWs)

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 overflow-hidden">
      {/* ─── Sidebar ─────────────────────────────────────────────── */}
      <aside
        className={clsx(
          'flex flex-col border-r border-slate-200 bg-white transition-all duration-200',
          sidebarOpen ? 'w-56' : 'w-14',
        )}
      >
        {/* Logo */}
        <div className={clsx('flex items-center gap-2 px-3 py-4 border-b border-slate-200', !sidebarOpen && 'justify-center')}>
          <Stethoscope size={20} className="text-medical-600 shrink-0" />
          {sidebarOpen && (
            <span className="font-bold text-sm text-medical-700 leading-tight">
              Medical<br/>RAG
            </span>
          )}
        </div>

        {/* Workspace list */}
        <div className="flex-1 overflow-y-auto py-2">
          {sidebarOpen && (
            <p className="px-3 py-1 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Workspaces
            </p>
          )}
          {loadingWs ? (
            <div className="px-3 py-2 text-xs text-slate-400">Loading…</div>
          ) : (
            workspaces.map(ws => (
              <button
                key={ws.id}
                onClick={() => setActiveWs(ws.id)}
                title={ws.name}
                className={clsx(
                  'w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors',
                  activeWs === ws.id
                    ? 'bg-medical-50 text-medical-700 font-medium'
                    : 'text-slate-600 hover:bg-slate-50',
                )}
              >
                <div
                  className={clsx(
                    'w-2 h-2 rounded-full shrink-0',
                    activeWs === ws.id ? 'bg-medical-500' : 'bg-slate-300',
                  )}
                />
                {sidebarOpen && (
                  <span className="truncate">{ws.name}</span>
                )}
              </button>
            ))
          )}
        </div>

        {/* Stats */}
        {sidebarOpen && stats && (
          <div className="border-t border-slate-200 px-3 py-3 space-y-1">
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <Activity size={12} />
              <span>{stats.chunk_count} chunks indexed</span>
            </div>
            {stats.subspecialty && (
              <p className="text-xs text-slate-400 capitalize">{stats.subspecialty}</p>
            )}
          </div>
        )}

        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarOpen(v => !v)}
          className="flex items-center justify-center py-3 border-t border-slate-200 text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-colors"
        >
          {sidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
        </button>
      </aside>

      {/* ─── Main panel ──────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-slate-200 bg-white shrink-0">
          <div>
            <h1 className="text-sm font-semibold text-slate-800">
              {currentWs?.name ?? 'Medical Research Assistant'}
            </h1>
            {currentWs?.description && (
              <p className="text-xs text-slate-400 mt-0.5">{currentWs.description}</p>
            )}
          </div>

          {/* Tab switcher */}
          <nav className="flex gap-1 bg-slate-100 rounded-xl p-1">
            {tabs.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={clsx(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                  tab === t.id
                    ? 'bg-white text-medical-700 shadow-sm'
                    : 'text-slate-500 hover:text-slate-700',
                )}
              >
                {t.icon}
                {t.label}
              </button>
            ))}
          </nav>
        </header>

        {/* Content area */}
        <div className="flex-1 overflow-hidden">
          {!activeWs ? (
            <div className="flex items-center justify-center h-full text-slate-400 text-sm">
              Select a workspace to get started
            </div>
          ) : (
            <>
              {tab === 'chat' && (
                <div className="h-full">
                  <ChatPage workspaceId={activeWs} />
                </div>
              )}
              {tab === 'search' && (
                <div className="h-full overflow-y-auto p-4">
                  <SearchTab workspaceId={activeWs} />
                </div>
              )}
              {tab === 'documents' && (
                <div className="h-full overflow-y-auto p-4">
                  <DocumentsTab workspaceId={activeWs} />
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  )
}
