import React, { useState } from 'react'
import { sendChat, resumeChat } from './api'

function InterruptCheck({ interrupt, modifiedArgsText, setModifiedArgsText, onApprove, onReject, isLoading }) {
  if (!interrupt) return null
  return (
    <div className="interrupt-card">
      <div className="interrupt-header">
        <span className="interrupt-icon">⚠️</span>
        <h3>Tool Call Requires Approval</h3>
      </div>
      <div className="interrupt-body">
        <div className="interrupt-tool">
          <span className="label">Tool:</span>
          <span className="value">{interrupt.tool_name || interrupt.tool}</span>
        </div>
        <div className="interrupt-args">
          <span className="label">Arguments:</span>
          <textarea
            className="args-textarea"
            value={modifiedArgsText}
            onChange={(e) => setModifiedArgsText(e.target.value)}
            rows={8}
            disabled={isLoading}
          />
        </div>
      </div>
      <div className="interrupt-actions">
        <button className="btn btn-primary" onClick={onApprove} disabled={isLoading}>
          {isLoading ? <span className="spinner"></span> : 'Approve & Run'}
        </button>
        <button className="btn btn-secondary" onClick={onReject} disabled={isLoading}>
          {isLoading ? <span className="spinner"></span> : 'Reject'}
        </button>
      </div>
    </div>
  )
}

export default function App() {
  const [message, setMessage] = useState('')
  const [threadId, setThreadId] = useState(null)
  const [chatLog, setChatLog] = useState([])
  const [interrupt, setInterrupt] = useState(null)
  const [modifiedArgsText, setModifiedArgsText] = useState('{}')
  const [isLoading, setIsLoading] = useState(false)

  async function handleSend() {
    if (!message.trim() || isLoading) return
    const userMsg = message
    setMessage('')
    setIsLoading(true)
    try {
      setChatLog((c) => [...c, { from: 'user', text: userMsg }])
      const data = await sendChat(userMsg, threadId)
      if (data.thread_id) setThreadId(data.thread_id)

      if (data.interrupt) {
        setInterrupt(data.interrupt)
        setModifiedArgsText(JSON.stringify(data.interrupt.tool_args || {}, null, 2))
        setIsLoading(false)
      } else {
        setChatLog((c) => [...c, { from: 'ai', text: data.response }])
        setInterrupt(null)
        setIsLoading(false)
      }
    } catch (e) {
      setChatLog((c) => [...c, { from: 'error', text: String(e) }])
      setIsLoading(false)
    }
  }

  async function handleApprove() {
    let modified = null
    try {
      modified = JSON.parse(modifiedArgsText)
    } catch {
      alert('Modified args must be valid JSON')
      return
    }
    setIsLoading(true)
    try {
      const data = await resumeChat(threadId, true, modified)
      setInterrupt(null)
      setChatLog((c) => [...c, { from: 'ai', text: data.response }])
    } catch (e) {
      setChatLog((c) => [...c, { from: 'error', text: String(e) }])
    } finally {
      setIsLoading(false)
    }
  }

  async function handleReject() {
    setIsLoading(true)
    try {
      const data = await resumeChat(threadId, false, null)
      setInterrupt(null)
      setChatLog((c) => [...c, { from: 'ai', text: data.response }])
    } catch (e) {
      setChatLog((c) => [...c, { from: 'error', text: String(e) }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Agent Chat</h1>

      <div className="chat">
        {chatLog.map((m, i) => (
          <div key={i} className={`msg msg-${m.from}`}>
            <pre>{m.text}</pre>
          </div>
        ))}
        {interrupt && (
          <InterruptCheck
            interrupt={interrupt}
            modifiedArgsText={modifiedArgsText}
            setModifiedArgsText={setModifiedArgsText}
            onApprove={handleApprove}
            onReject={handleReject}
            isLoading={isLoading}
          />
        )}
      </div>

      <div className="composer">
        <textarea
          placeholder="Type your message..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={4}
          disabled={isLoading}
        />
        <div className="composer-actions">
          <button className="btn btn-primary" onClick={handleSend} disabled={isLoading}>
            {isLoading ? <span className="spinner"></span> : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
