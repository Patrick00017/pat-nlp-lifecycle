import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { connectSSE, BASE } from '../api'

const STREAM_URL = `${BASE}/opencode/chat/stream`

export default function RagChatPanel() {
  const [message, setMessage] = useState('')
  const [threadId, setThreadId] = useState(null)
  const [chatLog, setChatLog] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const chatRef = useRef(null)
  const messageTokensRef = useRef('')
  const reasonTokensRef = useRef('')
  const [messageTokens, setMessageTokens] = useState('')
  const [reasonTokens, setReasonTokens] = useState('')
  const [isAutoScroll, setIsAutoScroll] = useState(true)

  useEffect(() => {
    if (chatRef.current && isAutoScroll) {
      chatRef.current.scrollTo({ top: chatRef.current.scrollHeight, behavior: 'smooth' })
    }
  }, [chatLog, messageTokens, reasonTokens, isAutoScroll])

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

    const payload = { message: userMsg, agent: 'rag' }
    if (threadId) payload.thread_id = threadId

    connectSSE(STREAM_URL, payload,
      (rawData) => {
        let data
        try { data = JSON.parse(rawData) } catch { return }
        if (data.type === 'thread_id') {
          setThreadId(data.value)
        } else if (data.type === 'reason') {
          reasonTokensRef.current += data.content
          setReasonTokens(reasonTokensRef.current)
        } else if (data.type === 'message') {
          messageTokensRef.current += data.content
          setMessageTokens(messageTokensRef.current)
        } else if (data.type === 'done') {
          setChatLog(c => [...c, { from: 'ai', reason: reasonTokensRef.current || null, content: messageTokensRef.current || '(无返回内容)' }])
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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: '#fff', borderRadius: 8, border: '1px solid #e5e7eb', overflow: 'hidden' }}>
      <div style={{ padding: '8px 12px', fontSize: 12, color: '#94a3b8', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between' }}>
        <span>文档助手</span>
        {threadId && <span style={{ fontFamily: 'monospace' }}>{threadId.slice(0, 8)}...</span>}
      </div>

      <div ref={chatRef} onScroll={handleScroll} style={{ flex: 1, overflowY: 'auto', padding: 12, display: 'flex', flexDirection: 'column' }}>
        {chatLog.length === 0 && !isLoading && (
          <div className="welcome-guide">
            <div className="welcome-header">
              <span className="welcome-icon">📖</span>
              <span>欢迎使用文档助手</span>
            </div>
            <div className="welcome-questions">
              <h3>你可以这样问我:</h3>
              <div className="question-chips">
                {['IPS系统如何配置参数'].map((q, i) => (
                  <button key={i} className="question-chip" onClick={() => setMessage(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {chatLog.map((m, i) => (
          <div key={i} style={{
            display: 'flex', flexDirection: 'column',
            alignItems: m.from === 'user' ? 'flex-end' : 'flex-start',
            marginBottom: 8,
          }}>
            <div style={{
              maxWidth: '90%', padding: '8px 12px', borderRadius: 8, fontSize: 13, lineHeight: 1.5,
              background: m.from === 'user' ? '#3b82f6' : '#f1f5f9',
              color: m.from === 'user' ? '#fff' : '#1e293b',
            }}>
              {m.reason && (
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4, padding: '4px 8px', background: '#e2e8f0', borderRadius: 4 }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.reason}</ReactMarkdown>
                </div>
              )}
              {(m.content || m.text) && (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content || m.text}</ReactMarkdown>
              )}
            </div>
          </div>
        ))}
        {(messageTokens || reasonTokens) && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', marginBottom: 8 }}>
            <div style={{ maxWidth: '90%', padding: '8px 12px', borderRadius: 8, fontSize: 13, lineHeight: 1.5, background: '#f1f5f9', color: '#1e293b' }}>
              {reasonTokens && (
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4, padding: '4px 8px', background: '#e2e8f0', borderRadius: 4 }}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{reasonTokens}</ReactMarkdown>
                </div>
              )}
              {messageTokens && <ReactMarkdown remarkPlugins={[remarkGfm]}>{messageTokens}</ReactMarkdown>}
              <span style={{ animation: 'blink 1s step-end infinite', fontSize: 16 }}>▋</span>
            </div>
          </div>
        )}
      </div>

      <div style={{ borderTop: '1px solid #e5e7eb', padding: 8, display: 'flex', gap: 8 }}>
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
          placeholder="输入消息..."
          disabled={isLoading}
          style={{
            flex: 1, padding: '8px 10px', border: '1px solid #d1d5db', borderRadius: 6,
            fontFamily: 'inherit', fontSize: 13, resize: 'none', height: 36, lineHeight: '20px',
          }}
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !message.trim()}
          style={{
            padding: '4px 16px', borderRadius: 6, border: 'none', fontSize: 13, cursor: 'pointer',
            background: isLoading || !message.trim() ? '#d1d5db' : '#3b82f6',
            color: '#fff', fontWeight: 500, whiteSpace: 'nowrap',
          }}
        >
          {isLoading ? '...' : '发送'}
        </button>
      </div>
    </div>
  )
}
