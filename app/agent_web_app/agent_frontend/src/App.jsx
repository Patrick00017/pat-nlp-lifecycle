import { useState } from 'react'
import Tabs from './components/Tabs'
import ChatPanel from './components/ChatPanel'
import DataAnalysis from './components/DataAnalysis'
import OpenCodePanel from './components/OpenCodePanel'
import DiagnosisChat from './components/DiagnosisChat'

const tabs = [
  { key: 'chat', label: '智能助手' },
  { key: 'analysis', label: '数据分析' },
  { key: 'fsm', label: '事件诊断' },
  { key: 'opencode', label: 'Opencode 助手' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  return (
    <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab}>
      <div style={{ height: '100%', display: activeTab === 'chat' ? '' : 'none' }}><ChatPanel /></div>
      <div style={{ height: '100%', display: activeTab === 'analysis' ? '' : 'none' }}><DataAnalysis /></div>
      <div style={{ height: '100%', display: activeTab === 'fsm' ? '' : 'none' }}><DiagnosisChat /></div>
      <div style={{ height: '100%', display: activeTab === 'opencode' ? '' : 'none' }}><OpenCodePanel /></div>
    </Tabs>
  )
}
