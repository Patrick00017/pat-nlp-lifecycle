import React, { useState, useRef, useEffect } from 'react'
import { sendChat, resumeChat, sendChatStream, connectSSE } from './api'

function InterruptMessage({ interrupt, modifiedArgsText, setModifiedArgsText, onApprove, onReject, isLoading }) {
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
  const [threadId, setThreadId] = useState(crypto.randomUUID())
  const [chatLog, setChatLog] = useState([])
  const [modifiedArgsText, setModifiedArgsText] = useState('{}')
  const [isLoading, setIsLoading] = useState(false)
  const [mode, setMode] = useState('IPS')
  const [callMethod, setCallMethod] = useState("Invoke")
  const chatRef = useRef(null)
  const tokensRef = useRef("")

  const [tokens, setTokens] = useState("")
  const [isComplete, setIsComplete] = useState(false)

  const modules = {
    IPS: ['胶水参数分析', 'MP压力辊参数分析', '接纸机张力参数分析', '真空泵参数分析'],
    RAG: ['服务器硬件配置说明', '用户角色说明', '数据说明', '功能说明'],
  }

  const placeholderQuestions = [
    '分析一下在时间段2026.4.22上午9点至2026.4.22下午1点的胶水参数赋值情况',
    '分析一下在时间段2026.4.22上午9点至2026.4.22下午1点的MP压力辊赋值情况',
    '强换功能有哪两种方式？有什么区别？',
    '如何自定义未完工订单列表的显示？',
  ]

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [chatLog, tokens])

  async function handleSend() {
    if (!message.trim() || isLoading) return
    const userMsg = message
    setMessage('')
    setIsLoading(true)
    try {
      setChatLog((c) => [...c, { from: 'user', text: userMsg }])
      if (callMethod === "Stream"){
        const payload = { message }
        if (threadId){
          payload.thread_id = threadId
        }
        else{
          payload.thread_id = crypto.randomUUID()
        }

        connectSSE("http://localhost:8000/chat/stream", payload,
          (rawData) => {
              // rawData is like: {"type": "message", "content": "text"} or {"type": "thread_id", "value": "..."}
              // Sometimes it includes "data: " prefix, handle both cases
              let jsonStr = rawData
              try {
                const raw = JSON.parse(jsonStr)
                let msgStr = raw
                if (msgStr.startsWith('data: ')) {
                  msgStr = msgStr.slice(6).trim()
                }
                let data = JSON.parse(msgStr)

                if (data.type === 'message') {
                  // setTokens((prev) => prev + data.content)
                  setTokens((prev) => {
                    tokensRef.current = prev + data.content
                    return tokensRef.current
                  })
                } else if (data.type === 'interrupt') {
                  if (tokensRef.current !== ""){
                    setChatLog((c) => [...c, { from: 'ai', text: tokensRef.current }])
                    setTokens("")
                  }
                  setChatLog((c) => [...c, {
                    type: 'interrupt',
                    interrupt: {'tool_name': data.value.tool_name},
                    modifiedArgsText: JSON.stringify(data.value.tool_args || {}, null, 2)
                  }])
                  setModifiedArgsText(JSON.stringify(data.value.tool_args || {}, null, 2))
                  setIsLoading(false)
                } else if (data.type === 'done') {
                  console.log("done. tokens:" + tokensRef.current)
                  setChatLog((c) => [...c, { from: 'ai', text: tokensRef.current }])
                  setIsLoading(false)
                  setTokens("");
                  setIsComplete(true)
                }
              } catch (e) {
                console.error("Parse error:", e, "Raw:", rawData)
              }
          },
          (err) => {
            console.error("SSE error:", err)
            setChatLog((c) => [...c, { from: 'error', text: String(err) }])
            setIsLoading(false)
          }
        )
      } else {
        const data = await sendChat(userMsg, threadId)
        if (data.thread_id) setThreadId(data.thread_id)
        if (data.interrupt) {
          setChatLog((c) => [...c, {
            type: 'interrupt',
            interrupt: data.interrupt,
            modifiedArgsText: JSON.stringify(data.interrupt.tool_args || {}, null, 2)
          }])
          setModifiedArgsText(JSON.stringify(data.interrupt.tool_args || {}, null, 2))
          setIsLoading(false)
        } else {
          setChatLog((c) => [...c, { from: 'ai', text: data.response }])
          setIsLoading(false)
        }
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
      setChatLog((c) => {
        const newLog = c.filter((m) => m.type !== 'interrupt')
        const interruptMsg = c.find((m) => m.type === 'interrupt')
        const toolName = interruptMsg?.interrupt?.tool_name || interruptMsg?.interrupt?.tool || 'tool'
        const argsText = modifiedArgsText
        return [...newLog, { from: 'system', text: `[Approved] ${toolName}\nArgs: ${argsText}` }, { from: 'ai', text: data.response }]
      })
    } catch (e) {
      setChatLog((c) => {
        const newLog = c.filter((m) => m.type !== 'interrupt')
        return [...newLog, { from: 'error', text: String(e) }]
      })
    } finally {
      setIsLoading(false)
    }
  }

  async function handleReject() {
    setIsLoading(true)
    try {
      const data = await resumeChat(threadId, false, null)
      setChatLog((c) => {
        const newLog = c.filter((m) => m.type !== 'interrupt')
        const interruptMsg = c.find((m) => m.type === 'interrupt')
        const toolName = interruptMsg?.interrupt?.tool_name || interruptMsg?.interrupt?.tool || 'tool'
return [...newLog, { from: 'system', text: `[Rejected] ${toolName}` }, { from: 'ai', text: data.response }]
      })
    } catch (e) {
      setChatLog((c) => {
        const newLog = c.filter((m) => m.type !== 'interrupt')
        return [...newLog, { from: 'error', text: String(e) }]
      })
    } finally {
      setIsLoading(false)
    }
}

  return (
    <div className="container">
      <h1>对话</h1>

      <div className="chat" ref={chatRef}>
        {chatLog.length === 0 && (
          <div className="welcome-guide">
            <div className="welcome-header">
              <span className="welcome-icon">🛡️</span>
              <span>欢迎使用瓦楞辊产线助手</span>
            </div>
            <div className="welcome-modules">
              <h3>可用模块 ({mode})</h3>
              <ul>
                {modules[mode].map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
            <div className="welcome-questions">
              <h3>你可以这样问我:</h3>
              <div className="question-chips">
                {placeholderQuestions.map((q, i) => (
                  <button key={i} className="question-chip" onClick={() => setMessage(q)}>{q}</button>
                ))}
              </div>
            </div>
          </div>
        )}
        {chatLog.map((m, i) => (
          m.type === 'interrupt' ? (
            <div key={i} className="msg-wrapper msg-system-wrapper">
              <InterruptMessage
                interrupt={m.interrupt}
                modifiedArgsText={m.modifiedArgsText}
                setModifiedArgsText={(text) => {
                  setModifiedArgsText(text)
                  setChatLog((c) => c.map((item, idx) => idx === i ? { ...item, modifiedArgsText: text } : item))
                }}
                onApprove={handleApprove}
                onReject={handleReject}
                isLoading={isLoading}
              />
            </div>
          ) : (
            <div key={i} className={`msg-wrapper msg-${m.from}-wrapper`}>
              <div className={`msg msg-${m.from}`}>
                <pre>{m.text}</pre>
              </div>
            </div>
          )
        ))}
        {tokens && (
          <div className="msg-wrapper msg-ai-wrapper">
            <div className="msg msg-ai">
              <pre>{tokens}<span className="cursor">▋</span></pre>
            </div>
          </div>
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
          <div className="mode-selector">
            <label>工作流:</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={isLoading}>
              <option value="IPS">IPS</option>
              <option value="RAG">RAG</option>
            </select>
          </div>
          <div className="call-selector">
            <label>Call:</label>
            <select value={callMethod} onChange={(e) => setCallMethod(e.target.value)} disabled={isLoading}>
              <option value="Invoke">Invoke</option>
              <option value="Stream">Stream</option>
            </select>
          </div>
          <button className="btn btn-primary" onClick={handleSend} disabled={isLoading}>
            {isLoading ? <span className="spinner"></span> : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
