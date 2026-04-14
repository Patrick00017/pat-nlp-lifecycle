import React, { useState } from 'react'
import { sendChat, resumeChat } from './api'

export default function App() {
  const [message, setMessage] = useState('')
  const [threadId, setThreadId] = useState(null)
  const [chatLog, setChatLog] = useState([])
  const [interrupt, setInterrupt] = useState(null)
  const [modifiedArgsText, setModifiedArgsText] = useState('{}')

  async function handleSend() {
    try {
      const data = await sendChat(message, threadId)
      if (data.thread_id) setThreadId(data.thread_id)

      if (data.interrupt) {
        setInterrupt(data.interrupt)
        setModifiedArgsText(JSON.stringify(data.interrupt.tool_args || {}, null, 2))
      } else {
        setChatLog((c) => [...c, { from: 'ai', text: data.response }])
        setInterrupt(null)
      }
    } catch (e) {
      setChatLog((c) => [...c, { from: 'error', text: String(e) }])
    }
  }

  async function handleApprove() {
    let modified = null
    try {
      modified = JSON.parse(modifiedArgsText)
    } catch (e) {
      alert('Modified args must be valid JSON')
      return
    }
    try {
      const data = await resumeChat(threadId, true, modified)
      setInterrupt(null)
      setChatLog((c) => [...c, { from: 'ai', text: data.response }])
    } catch (e) {
      setChatLog((c) => [...c, { from: 'error', text: String(e) }])
    }
  }

  async function handleReject() {
    try {
      const data = await resumeChat(threadId, false, null)
      setInterrupt(null)
      setChatLog((c) => [...c, { from: 'ai', text: data.response }])
    } catch (e) {
      setChatLog((c) => [...c, { from: 'error', text: String(e) }])
    }
  }

  return (
    <div className="container">
      <h1>Agent Chat</h1>

      <div className="chat">
        {chatLog.map((m, i) => (
          <div key={i} className={`msg ${m.from}`}>
            <pre>{m.text}</pre>
          </div>
        ))}
      </div>

      {interrupt && (
        <div className="interrupt">
          <h3>Tool call requires approval</h3>
          <div><strong>Tool:</strong> {interrupt.tool_name || interrupt.tool}</div>
          <div><strong>Args:</strong></div>
          <textarea
            value={modifiedArgsText}
            onChange={(e) => setModifiedArgsText(e.target.value)}
            rows={8}
          />
          <div className="buttons">
            <button onClick={handleApprove}>Approve & Run</button>
            <button onClick={handleReject}>Reject</button>
          </div>
        </div>
      )}

      <div className="composer">
        <textarea
          placeholder="Type your message..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={4}
        />
        <div className="buttons">
          <button onClick={handleSend}>Send</button>
        </div>
      </div>
    </div>
  )
}
