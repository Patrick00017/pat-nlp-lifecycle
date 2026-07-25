import { useState } from 'react'
import Tabs from './components/Tabs'
import DiagnosisChat from './components/DiagnosisChat'
import RagChatPanel from './components/RagChatPanel'
import OrderMatchTimeline from './components/OrderMatchTimeline'
import KbManager from './components/KbManager'
import { useTheme } from './theme'

const tabs = [
  { key: 'fsm', label: '事件诊断' },
  { key: 'rag', label: '文档助手' },
  { key: 'order', label: '订单匹配' },
]

function SettingsModal({ onClose }) {
  const { theme, toggleTheme } = useTheme()

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.4)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-card)',
          borderRadius: 10,
          width: 520,
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 8px 30px rgba(0,0,0,0.2)',
        }}
      >
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '14px 18px', borderBottom: '1px solid var(--border-color)',
        }}>
          <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>设置</span>
          <button
            onClick={onClose}
            style={{
              border: 'none', background: 'none', cursor: 'pointer',
              fontSize: 16, color: 'var(--text-muted)', padding: '0 4px',
            }}
          >&#10005;</button>
        </div>

        <div style={{ flex: 1, overflow: 'auto' }}>

          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 10 }}>
              常规设置
            </div>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '10px 14px',
              background: 'var(--bg-body)',
              borderRadius: 8,
              border: '1px solid var(--border-color)',
            }}>
              <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>
                {theme === 'dark' ? '🌙 暗色模式' : '☀️ 亮色模式'}
              </span>
              <button
                onClick={toggleTheme}
                style={{
                  width: 44, height: 24, borderRadius: 12,
                  border: 'none', cursor: 'pointer',
                  background: theme === 'dark' ? '#3b82f6' : '#d1d5db',
                  position: 'relative',
                  transition: 'background 0.2s',
                }}
              >
                <span style={{
                  position: 'absolute', top: 2,
                  left: theme === 'dark' ? 22 : 2,
                  width: 20, height: 20, borderRadius: '50%',
                  background: '#fff',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                  transition: 'left 0.2s',
                }} />
              </button>
            </div>
          </div>

          <div style={{ padding: '14px 18px' }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 10 }}>
              知识库管理
            </div>
            <KbManager />
          </div>

        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('fsm')
  const [sharedThreadId, setSharedThreadId] = useState(null)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const gearBtn = (
    <button
      onClick={() => setSettingsOpen(true)}
      style={{
        border: 'none',
        background: 'none',
        cursor: 'pointer',
        fontSize: 18,
        color: 'var(--text-muted)',
        padding: '4px 8px',
      }}
      title="设置"
    >
      &#9881;
    </button>
  )

  return (
    <>
      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} headerRight={gearBtn}>
        <div style={{ height: '100%', display: activeTab === 'fsm' ? '' : 'none' }}><DiagnosisChat sharedThreadId={sharedThreadId} setSharedThreadId={setSharedThreadId} /></div>
        <div style={{ height: '100%', display: activeTab === 'rag' ? '' : 'none' }}><RagChatPanel /></div>
        <div style={{ height: '100%', display: activeTab === 'order' ? '' : 'none' }}><OrderMatchTimeline sharedThreadId={sharedThreadId} /></div>
      </Tabs>

      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
    </>
  )
}
