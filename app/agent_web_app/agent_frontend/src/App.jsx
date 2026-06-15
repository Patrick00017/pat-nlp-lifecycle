import { useState } from 'react'
import Tabs from './components/Tabs'
import ChatPanel from './components/ChatPanel'
import DataAnalysis from './components/DataAnalysis'
import OpenCodePanel from './components/OpenCodePanel'

const tabs = [
  { key: 'chat', label: '智能助手' },
  { key: 'analysis', label: '数据分析' },
  { key: 'opencode', label: 'Opencode 助手' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  return (
    <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab}>
      {activeTab === 'chat' && <ChatPanel />}
      {activeTab === 'analysis' && <DataAnalysis />}
      {activeTab === 'opencode' && <OpenCodePanel />}
    </Tabs>
  )
}
