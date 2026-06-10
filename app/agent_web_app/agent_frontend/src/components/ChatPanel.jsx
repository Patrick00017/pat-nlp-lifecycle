import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { resumeChat, connectSSE, fetchTools } from '../api'
import InterruptMessage from './InterruptMessage'
import MessageComposer from './MessageComposer'

const API_ENDPOINTS = {
  IPS_STREAM: 'http://localhost:8000/chat/stream',
  IPS_INVOKE: 'http://localhost:8000/chat',
  RAG: 'http://localhost:8000/rag',
}

function parseSSEData(rawData) {
  try {
    const raw = JSON.parse(rawData)
    let msgStr = raw
    if (msgStr.startsWith('data: ')) {
      msgStr = msgStr.slice(6).trim()
    }
    return JSON.parse(msgStr)
  } catch (e) {
    console.error("Parse error:", e, "Raw:", rawData)
    return null
  }
}

function createInterruptPayload(data) {
  return {
    type: 'interrupt',
    interrupt: { tool_name: data.value.tool_name },
    modifiedArgsText: JSON.stringify(data.value.tool_args || {}, null, 2),
    modifiedArgsSchema: data.value.tool_args_schema || {},
    originalToolName: data.value.tool_name,
    originalArgsText: JSON.stringify(data.value.tool_args || {}, null, 2),
    originalSchema: data.value.tool_args_schema || {}
  }
}

function createSSECallbacks({
  setChatLog,
  setMessageTokens,
  setReasonTokens,
  setDocsTokens,
  setModifiedArgsText,
  setIsLoading,
  messageTokensRef,
  reasonTokensRef,
  docsTokensRef
}) {
  return {
    onReason: (content) => {
      reasonTokensRef.current += content
      setReasonTokens(reasonTokensRef.current)
    },
    onMessage: (content) => {
      messageTokensRef.current += content
      setMessageTokens(messageTokensRef.current)
    },
    onDocs: (content) => {
      docsTokensRef.current += content
      setDocsTokens(docsTokensRef.current)
    },
    onInterrupt: (data) => {
      const hasContent = messageTokensRef.current || reasonTokensRef.current || docsTokensRef.current
      if (hasContent) {
        setChatLog(c => [...c, { from: 'ai', reason: reasonTokensRef.current, docs: docsTokensRef.current, content: messageTokensRef.current, showReason: true, showDocs: false }])
        messageTokensRef.current = ""
        reasonTokensRef.current = ""
        docsTokensRef.current = ""
        setMessageTokens("")
        setReasonTokens("")
        setDocsTokens("")
      }
      setChatLog(c => [...c, createInterruptPayload(data)])
      setModifiedArgsText(createInterruptPayload(data).modifiedArgsText)
      setIsLoading(false)
    },
    onDone: () => {
      setChatLog(c => [...c, { from: 'ai', reason: reasonTokensRef.current, docs: docsTokensRef.current, content: messageTokensRef.current, showReason: true, showDocs: false }])
      setIsLoading(false)
      setMessageTokens("")
      setReasonTokens("")
      setDocsTokens("")
    },
    onError: (err) => {
      setChatLog(c => [...c, { from: 'error', text: String(err) }])
      setIsLoading(false)
    }
  }
}

export default function ChatPanel() {
  const [message, setMessage] = useState('')
  const [threadId, setThreadId] = useState(crypto.randomUUID())
  const [chatLog, setChatLog] = useState([])
  const [modifiedArgsText, setModifiedArgsText] = useState('{}')
  const [isLoading, setIsLoading] = useState(false)
  const [mode, setMode] = useState('IPS')
  const [tools, setTools] = useState([])
  const chatRef = useRef(null)
  const messageTokensRef = useRef("")
  const reasonTokensRef = useRef("")
  const docsTokensRef = useRef("")

  const [messageTokens, setMessageTokens] = useState("")
  const [reasonTokens, setReasonTokens] = useState("")
  const [docsTokens, setDocsTokens] = useState("")
  const [isAutoScroll, setIsAutoScroll] = useState(true)

  const modules = {
    IPS: ['胶水参数分析', 'MP压力辊参数分析', '接纸机张力参数分析', '真空泵参数分析'],
    RAG: ['服务器硬件配置说明', '用户角色说明', '数据说明', '功能说明'],
  }

const placeholderQuestions = [
    '分析一下在时间段2026.4.22上午9点至2026.4.22下午1点的胶水参数赋值情况',
    '分析一下在时间段2026.4.22上午9点至2026.4.22下午1点的MP压力辊参数赋值情况',
    '强换功能有哪两种方式？有什么区别？',
    '如何自定义未完工订单列表的显示？',
  ]

useEffect(() => {
    if (chatRef.current && isAutoScroll) {
      chatRef.current.scrollTo({
        top: chatRef.current.scrollHeight,
        behavior: 'smooth'
      })
    }
  }, [chatLog, messageTokens, reasonTokens, docsTokens, isAutoScroll])

  useEffect(() => {
    async function loadTools() {
      try {
        const data = await fetchTools()
        setTools(data.tools || [])
      } catch (e) {
        console.error('Failed to load tools:', e)
      }
    }
    loadTools()
  }, [])

  const handleScroll = () => {
    if (chatRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatRef.current
      const threshold = 100
      setIsAutoScroll(scrollHeight - scrollTop - clientHeight < threshold)
    }
  }

  async function handleSend() {
    if (!message.trim() || isLoading) return
    const userMsg = message
    setMessage('')
    setIsLoading(true)
    setMessageTokens('')
    setReasonTokens('')
    setDocsTokens('')
    messageTokensRef.current = ''
    reasonTokensRef.current = ''
    docsTokensRef.current = ''

    const callbacks = createSSECallbacks({
      setChatLog,
      setMessageTokens,
      setReasonTokens,
      setDocsTokens,
      setModifiedArgsText,
      setIsLoading,
      messageTokensRef,
      reasonTokensRef,
      docsTokensRef
    })

    try {
      if (mode === 'IPS') {
        setChatLog(c => [...c, { from: 'user', text: userMsg }])
        
        const payload = { message: userMsg }
        if (threadId) {
          payload.thread_id = threadId
        } else {
          const newThreadId = crypto.randomUUID()
          setThreadId(newThreadId)
          payload.thread_id = newThreadId
        }

        connectSSE(API_ENDPOINTS.IPS_STREAM, payload,
          (rawData) => {
            const data = parseSSEData(rawData)
            if (!data) return

            if (data.type === 'reason') callbacks.onReason(data.content)
            else if (data.type === 'message') callbacks.onMessage(data.content)
            else if (data.type === 'docs') callbacks.onDocs(data.content)
            else if (data.type === 'interrupt') callbacks.onInterrupt(data)
            else if (data.type === 'done') callbacks.onDone()
          },
          callbacks.onError
        )
      } else if (mode === 'RAG') {
        setChatLog(c => [...c, { from: 'user', text: userMsg }])
        const payload = { message: userMsg }

        connectSSE(API_ENDPOINTS.RAG, payload,
          (rawData) => {
            const data = parseSSEData(rawData)
            if (!data) return

            if (data.type === 'reason') callbacks.onReason(data.content)
            else if (data.type === 'message') callbacks.onMessage(data.content)
            else if (data.type === 'docs') callbacks.onDocs(data.content)
            else if (data.type === 'interrupt') callbacks.onInterrupt(data)
            else if (data.type === 'done') callbacks.onDone()
          },
          callbacks.onError
        )
      }
    } catch (e) {
      setChatLog(c => [...c, { from: 'error', text: String(e) }])
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

      <div className="chat" ref={chatRef} onScroll={handleScroll}>
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
        {chatLog.map((m, i) => {
          if (m.type === 'interrupt') {
            return (
              <div key={i} className="msg-wrapper msg-system-wrapper">
                <InterruptMessage
                  interrupt={m.interrupt}
                  modifiedArgsText={m.modifiedArgsText}
                  modifiedArgsSchema={m.modifiedArgsSchema || {}}
                  setModifiedArgsText={(text) => {
                    setModifiedArgsText(text)
                    setChatLog((c) => c.map((item, idx) => idx === i ? { ...item, modifiedArgsText: text, modifiedArgsSchema: item.modifiedArgsSchema } : item))
                  }}
                  onApprove={handleApprove}
                  onReject={handleReject}
                  isLoading={isLoading}
                  tools={tools}
                  onToolChange={(newToolName, newSchema) => {
                    setChatLog((c) => c.map((item, idx) => idx === i ? { ...item, interrupt: { ...item.interrupt, tool_name: newToolName }, modifiedArgsSchema: newSchema } : item))
                  }}
                  originalToolName={m.originalToolName || m.interrupt.tool_name || m.interrupt.tool}
                  originalArgsText={m.originalArgsText}
                  originalSchema={m.originalSchema || {}}
                />
              </div>
            )
          }
          return (
            <div key={i} className={`msg-wrapper msg-${m.from}-wrapper`}>
              <div className={`msg msg-${m.from}`}>
                {m.reason && (
                  <>
                    <button className="msg-toggle-btn" onClick={() => setChatLog(c => c.map((item, idx) => idx === i ? { ...item, showReason: !item.showReason } : item))}>
                      {m.showReason ? '▼' : '▶'} Reason
                    </button>
                    {m.showReason && (
                      <div className="msg-reason">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.reason}</ReactMarkdown>
                      </div>
                    )}
                  </>
                )}
                {m.docs && (
                  <>
                    <button className="msg-toggle-btn" onClick={() => setChatLog(c => c.map((item, idx) => idx === i ? { ...item, showDocs: !item.showDocs } : item))}>
                      {m.showDocs ? '▼' : '▶'} Docs
                    </button>
                    {m.showDocs && (
                      <div className="msg-docs">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.docs}</ReactMarkdown>
                      </div>
                    )}
                  </>
                )}
                {(m.content || m.text) && (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content || m.text}</ReactMarkdown>
                )}
              </div>
            </div>
          )
        })}
        {(messageTokens || reasonTokens) && (
          <div className="msg-wrapper msg-ai-wrapper">
            <div className="msg msg-ai">
              {reasonTokens && (
                <div className="msg-reason">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{reasonTokens}</ReactMarkdown>
                </div>
              )}
              {messageTokens && (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{messageTokens}</ReactMarkdown>
              )}
              <span className="cursor">▋</span>
            </div>
          </div>
        )}
      </div>
      <MessageComposer
        message={message}
        setMessage={setMessage}
        mode={mode}
        setMode={setMode}
        isLoading={isLoading}
        onSend={handleSend}
      />
    </div>
  )
}
