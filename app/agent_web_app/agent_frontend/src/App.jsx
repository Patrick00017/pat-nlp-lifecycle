import { useState } from 'react'
import Tabs from './components/Tabs'
import ChatPanel from './components/ChatPanel'
import DataAnalysis from './components/DataAnalysis'

const tabs = [
  { key: 'chat', label: '智能助手' },
  { key: 'analysis', label: '数据分析' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  return (
    <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab}>
      {activeTab === 'chat' && <ChatPanel />}
      {activeTab === 'analysis' && <DataAnalysis />}
    </Tabs>
  )
}
