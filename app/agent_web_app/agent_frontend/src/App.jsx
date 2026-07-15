import { useState } from 'react'
import Tabs from './components/Tabs'
import DiagnosisChat from './components/DiagnosisChat'
import RagChatPanel from './components/RagChatPanel'
import OrderMatchTimeline from './components/OrderMatchTimeline'
import KbManager from './components/KbManager'

const tabs = [
  { key: 'fsm', label: '事件诊断' },
  { key: 'rag', label: '文档助手' },
  { key: 'order', label: '订单匹配' },
  { key: 'kb', label: '知识库管理' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('fsm')
  const [sharedThreadId, setSharedThreadId] = useState(null)
  return (
    <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab}>
      <div style={{ height: '100%', display: activeTab === 'fsm' ? '' : 'none' }}><DiagnosisChat sharedThreadId={sharedThreadId} setSharedThreadId={setSharedThreadId} /></div>
      <div style={{ height: '100%', display: activeTab === 'rag' ? '' : 'none' }}><RagChatPanel /></div>
      <div style={{ height: '100%', display: activeTab === 'order' ? '' : 'none' }}><OrderMatchTimeline sharedThreadId={sharedThreadId} /></div>
      <div style={{ height: '100%', display: activeTab === 'kb' ? '' : 'none' }}><KbManager /></div>
    </Tabs>
  )
}
