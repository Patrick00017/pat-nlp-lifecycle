import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { connectSSE, listOpenCodeAgents } from '../api'

const OPENCODE_STREAM_URL = 'http://localhost:8000/opencode/chat/stream'

const PLACEHOLDER_QUESTIONS = [
  '读取当前项目目录结构',
  '帮我创建一个新的React组件',
  '分析项目中存在哪些代码问题',
  '帮我添加错误处理逻辑',
]

const toAgentObj = (a) => typeof a === 'string' ? { name: a } : a

export default function OpenCodePanel() {
  const [message, setMessage] = useState('')
  const [threadId, setThreadId] = useState(null)
  const [chatLog, setChatLog] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [agent, setAgent] = useState('general')
  const [agents, setAgents] = useState([{ name: 'general' }])

  const chatRef = useRef(null)
  const messageTokensRef = useRef('')
  const reasonTokensRef = useRef('')
  const [messageTokens, setMessageTokens] = useState('')
  const [reasonTokens, setReasonTokens] = useState('')
  const [isAutoScroll, setIsAutoScroll] = useState(true)

  useEffect(() => {
    if (chatRef.current && isAutoScroll) {
      chatRef.current.scrollTo({
        top: chatRef.current.scrollHeight,
        behavior: 'smooth',
      })
    }
  }, [chatLog, messageTokens, reasonTokens, isAutoScroll])

  useEffect(() => {
    async function loadAgents() {
      try {
        const data = await listOpenCodeAgents()
        if (data.agents && data.agents.length > 0) {
          setAgents(data.agents.map(toAgentObj))
        }
      } catch (e) {
        console.error('Failed to load agents:', e)
      }
    }
    loadAgents()
  }, [])

  const handleScroll = () => {
    if (chatRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatRef.current
      setIsAutoScroll(scrollHeight - scrollTop - clientHeight < 100)
    }
  }

  async function handleSend() {
    if (!message.trim() || isLoading) return
    const userMsg = message
    setMessage('')
    setIsLoading(true)
    setMessageTokens('')
    setReasonTokens('')
    messageTokensRef.current = ''
    reasonTokensRef.current = ''

    setChatLog(c => [...c, { from: 'user', text: userMsg }])

    const payload = { message: userMsg, agent }
    if (threadId) {
      payload.thread_id = threadId
    }

    connectSSE(OPENCODE_STREAM_URL, payload,
      (rawData) => {
        let data
        try {
          data = JSON.parse(rawData)
        } catch {
          return
        }

        if (data.type === 'thread_id') {
          setThreadId(data.value)
        } else if (data.type === 'reason') {
          reasonTokensRef.current += data.content
          setReasonTokens(reasonTokensRef.current)
        } else if (data.type === 'message') {
          messageTokensRef.current += data.content
          setMessageTokens(messageTokensRef.current)
        } else if (data.type === 'done') {
          setChatLog(c => [...c, {
            from: 'ai',
            reason: reasonTokensRef.current || null,
            content: messageTokensRef.current || '(无返回内容)',
          }])
          setIsLoading(false)
          setMessageTokens('')
          setReasonTokens('')
        } else if (data.type === 'error') {
          setChatLog(c => [...c, { from: 'error', text: data.error }])
          setIsLoading(false)
        }
      },
      (err) => {
        setChatLog(c => [...c, { from: 'error', text: String(err) }])
        setIsLoading(false)
      }
    )
  }

  function startNewSession() {
    setThreadId(null)
    setChatLog([])
    setMessageTokens('')
    setReasonTokens('')
    messageTokensRef.current = ''
    reasonTokensRef.current = ''
  }

  return (
    <div className="container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h1 style={{ margin: 0 }}>Opencode 助手</h1>
        {threadId && (
          <button className="btn btn-secondary" onClick={startNewSession} style={{ fontSize: 12, padding: '6px 12px' }}>
            + 新会话
          </button>
        )}
      </div>

      <div className="chat" ref={chatRef} onScroll={handleScroll}>
        {chatLog.length === 0 && (
          <div className="welcome-guide">
            <div className="welcome-header">
              <span className="welcome-icon">🤖</span>
              <span>欢迎使用 Opencode 编码助手</span>
            </div>
            <div className="welcome-modules">
              <h3>可用 Agents</h3>
              <ul>
                {agents.map(a => <li key={a.name}>{a.name}</li>)}
              </ul>
            </div>
            <div className="welcome-questions">
              <h3>你可以这样问我:</h3>
              <div className="question-chips">
                {PLACEHOLDER_QUESTIONS.map((q, i) => (
                  <button key={i} className="question-chip" onClick={() => setMessage(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {threadId && chatLog.length > 0 && (
          <div style={{ padding: '6px 12px', fontSize: 12, color: '#94a3b8', borderBottom: '1px solid #f1f5f9' }}>
            会话: {threadId}
          </div>
        )}

        {chatLog.map((m, i) => {
          if (m.from === 'system') {
            return (
              <div key={i} className="msg-wrapper msg-system-wrapper">
                <div className="msg msg-system">{m.text}</div>
              </div>
            )
          }
          return (
            <div key={i} className={`msg-wrapper msg-${m.from}-wrapper`}>
              <div className={`msg msg-${m.from}`}>
                {m.reason && (
                  <div className="msg-reason">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.reason}</ReactMarkdown>
                  </div>
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

      <div className="composer">
        <textarea
          style={{
            width: '100%', boxSizing: 'border-box', padding: 10,
            border: '1px solid #e2e8f0', borderRadius: 8,
            fontFamily: 'inherit', fontSize: 14, resize: 'none', height: 52,
          }}
          placeholder="输入消息..."
          value={message}
          onChange={e => setMessage(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
          disabled={isLoading}
        />
        <div className="composer-actions">
          <div className="mode-selector">
            <label>Agent:</label>
            <select value={agent} onChange={e => setAgent(e.target.value)} disabled={isLoading}>
              {agents.map(a => (
                <option key={a.name} value={a.name}>{a.name}</option>
              ))}
            </select>
          </div>
          <button className="btn btn-primary" onClick={handleSend} disabled={isLoading || !message.trim()}>
            {isLoading ? <span className="spinner" /> : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
