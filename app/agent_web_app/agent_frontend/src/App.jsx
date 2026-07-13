import { useState } from 'react'
import Tabs from './components/Tabs'
import DiagnosisChat from './components/DiagnosisChat'
import RagChatPanel from './components/RagChatPanel'
import OrderMatchTimeline from './components/OrderMatchTimeline'

const tabs = [
  { key: 'fsm', label: '事件诊断' },
  { key: 'rag', label: '文档助手' },
  { key: 'order', label: '订单匹配' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('fsm')
  const [sharedThreadId, setSharedThreadId] = useState(null)
  return (
    <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab}>
      <div style={{ height: '100%', display: activeTab === 'fsm' ? '' : 'none' }}><DiagnosisChat sharedThreadId={sharedThreadId} setSharedThreadId={setSharedThreadId} /></div>
      <div style={{ height: '100%', display: activeTab === 'rag' ? '' : 'none' }}><RagChatPanel /></div>
      <div style={{ height: '100%', display: activeTab === 'order' ? '' : 'none' }}><OrderMatchTimeline sharedThreadId={sharedThreadId} /></div>
    </Tabs>
  )
}
