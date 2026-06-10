import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { funcQuery, funcCall, analysisInit, analysisStep, analysisStepback } from '../api'

const ANALYSIS_TOOLS = {
  get_material_change_in_log:       { label: '材料变更分析', entryNode: 'mc_timeline' },
  get_glue_set_func_call_in_log:    { label: '胶量设定分析', entryNode: 'glue_list' },
  track_material_in_log:            { label: '生命周期跟踪', entryNode: 'track_lifecycle' },
  get_pressroll_mp_set_func_call_in_log: { label: 'MP压力辊分析', entryNode: 'press_list' },
}

export default function DataAnalysis() {
  const [inputMessage, setInputMessage] = useState('')
  const [chatLog, setChatLog] = useState([])
  const [messages, setMessages] = useState([])
  const [isQuerying, setIsQuerying] = useState(false)
  const [isExecuting, setIsExecuting] = useState(false)
  const [isStepping, setIsStepping] = useState(false)
  const chatRef = useRef(null)

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTo({
        top: chatRef.current.scrollHeight,
        behavior: 'smooth',
      })
    }
  }, [chatLog])

  function updateEntry(index, updates) {
    setChatLog(c => c.map((item, i) => (i === index ? { ...item, ...updates } : item)))
  }

  async function handleSend() {
    const text = inputMessage.trim()
    if (!text || isQuerying) return

    setInputMessage('')
    setChatLog(c => [...c, { type: 'user', text }])
    setIsQuerying(true)

    const nextMessages = [...messages, { role: 'user', content: text }]

    try {
      const resp = await funcQuery(nextMessages)
      const toolCalls = resp.tool_calls || []
      const assistantContent = resp.content || ''

      setMessages([...nextMessages, { role: 'assistant', content: assistantContent }])

      if (toolCalls.length > 0) {
        const tc = toolCalls[0]
        const toolDef = ANALYSIS_TOOLS[tc.name]
        if (!toolDef) {
          setChatLog(c => [...c, { type: 'info', text: `工具 "${tc.name}" 不支持分析` }])
          setIsQuerying(false)
          return
        }
        setChatLog(c => [...c, { type: 'tool_call', toolCall: tc, toolDef }])
      } else {
        setChatLog(c => [...c, { type: 'ai', text: assistantContent || '无法识别为分析操作' }])
      }
    } catch (e) {
      setChatLog(c => [...c, { type: 'error', text: `查询失败: ${e.message}` }])
    }
    setIsQuerying(false)
  }

  function handleToolCallArgChange(entryIndex, key, value) {
    updateEntry(entryIndex, {
      toolCall: {
        ...chatLog[entryIndex].toolCall,
        arguments: { ...chatLog[entryIndex].toolCall.arguments, [key]: value },
      },
    })
  }

  async function handleExecuteTool(entryIndex) {
    const entry = chatLog[entryIndex]
    if (!entry || entry.type !== 'tool_call') return
    setIsExecuting(true)

    try {
      const callResp = await funcCall([{ name: entry.toolCall.name, arguments: entry.toolCall.arguments }])
      const result = callResp.results?.[0]
      if (result?.error) {
        setChatLog(c => [...c, { type: 'error', text: `工具执行失败: ${result.error}` }])
        setIsExecuting(false)
        return
      }

      const toolResult = result?.result ?? ''
      setChatLog(c => [...c, { type: 'tool_result', result: toolResult }])

      const initResp = await analysisInit(entry.toolCall.name, entry.toolCall.arguments, toolResult)
      setChatLog(c => [...c, {
        type: 'analysis',
        stateId: initResp.state_id,
        context: initResp.context,
        executionPath: [],
        analysisResult: '',
        availableNodes: initResp.available_nodes,
        isTerminal: initResp.available_nodes.length === 0,
      }])
    } catch (e) {
      setChatLog(c => [...c, { type: 'error', text: `执行失败: ${e.message}` }])
    }
    setIsExecuting(false)
  }

  async function handleStep(entryIndex, nodeId) {
    setIsStepping(true)
    const entry = chatLog[entryIndex]
    try {
      const resp = await analysisStep(entry.stateId, nodeId)
      updateEntry(entryIndex, {
        analysisResult: resp.result,
        executionPath: resp.execution_path,
        availableNodes: resp.available_nodes,
        isTerminal: resp.is_terminal,
      })
    } catch (e) {
      setChatLog(c => [...c, { type: 'error', text: `分析步骤失败: ${e.message}` }])
    }
    setIsStepping(false)
  }

  async function handleStepback(entryIndex) {
    setIsStepping(true)
    const entry = chatLog[entryIndex]
    try {
      const resp = await analysisStepback(entry.stateId)
      updateEntry(entryIndex, {
        analysisResult: resp.previous_result || '',
        executionPath: resp.execution_path,
        availableNodes: resp.available_nodes,
        isTerminal: resp.is_terminal,
      })
    } catch (e) {
      setChatLog(c => [...c, { type: 'error', text: `回退失败: ${e.message}` }])
    }
    setIsStepping(false)
  }

  function renderEntry(entry, index) {
    switch (entry.type) {
      case 'user':
        return (
          <div key={index} className="msg-wrapper msg-user-wrapper">
            <div className="msg msg-user"><ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.text}</ReactMarkdown></div>
          </div>
        )

      case 'tool_call':
        return (
          <div key={index} className="msg-wrapper msg-system-wrapper">
            <div className="interrupt-card">
              <div className="interrupt-header">
                <span className="interrupt-icon">🔍</span>
                <h3>识别到分析操作</h3>
              </div>
              <div className="interrupt-body">
                <div className="interrupt-tool">
                  <span className="label">工具</span>
                  <div className="tool-badge">
                    <span className="tool-name">{entry.toolDef?.label || entry.toolCall.name}</span>
                  </div>
                </div>
                <div className="interrupt-args">
                  <span className="label">参数</span>
                  <div className="arg-fields">
                    {Object.entries(entry.toolCall.arguments).map(([key, val]) => (
                      <div key={key} className="arg-row">
                        <div className="arg-label">
                          <span className="arg-key">{key}</span>
                        </div>
                        <input
                          className="arg-input"
                          value={val ?? ''}
                          onChange={e => handleToolCallArgChange(index, key, e.target.value)}
                          disabled={isExecuting}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="interrupt-actions">
                <button className="btn btn-approve" onClick={() => handleExecuteTool(index)} disabled={isExecuting}>
                  {isExecuting ? <span className="spinner" /> : '▶ 执行分析'}
                </button>
              </div>
            </div>
          </div>
        )

      case 'tool_result':
        return (
          <div key={index} className="msg-wrapper msg-ai-wrapper">
            <div className="msg msg-ai"><ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.result}</ReactMarkdown></div>
          </div>
        )

      case 'analysis':
        return (
          <div key={index} className="msg-wrapper msg-ai-wrapper">
            <div className="msg msg-ai">
              {entry.context && (
                <div style={{ fontSize: 13, color: '#64748b', marginBottom: 8, padding: '4px 0' }}>
                  查询范围: {entry.context.start_time || '?'} ~ {entry.context.end_time || '?'}
                  {entry.context.material ? ` | 材料: ${entry.context.material}` : ''}
                </div>
              )}

              {entry.executionPath.length > 0 && (
                <div className="analysis-path">
                  <span className="analysis-path-label">分析路径</span>
                  {entry.executionPath.map((nodeId, i) => (
                    <span key={nodeId} style={{ display: 'contents' }}>
                      {i > 0 && <span className="analysis-path-arrow">→</span>}
                      <span className="analysis-path-chip">{nodeId}</span>
                    </span>
                  ))}
                </div>
              )}

              {entry.analysisResult && (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.analysisResult}</ReactMarkdown>
              )}

              {entry.isTerminal && (
                <div className="analysis-done-badge">✓ 分析完成</div>
              )}

              {!entry.isTerminal && entry.availableNodes.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 13, color: '#64748b', marginBottom: 8, fontWeight: 500 }}>可选下一步:</div>
                  <div className="analysis-node-list">
                    {entry.availableNodes.map(node => (
                      <button
                        key={node.id}
                        className="analysis-node-btn"
                        onClick={() => handleStep(index, node.id)}
                        disabled={isStepping}
                      >
                        <span className="node-name">{node.name}</span>
                        <span className="node-desc">{node.description}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div style={{ marginTop: 8 }}>
                <button
                  className="analysis-stepback-btn"
                  onClick={() => handleStepback(index)}
                  disabled={entry.executionPath.length === 0 || isStepping}
                >
                  ⟲ 上一步
                </button>
              </div>
            </div>
          </div>
        )

      case 'ai':
        return (
          <div key={index} className="msg-wrapper msg-ai-wrapper">
            <div className="msg msg-ai"><ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.text}</ReactMarkdown></div>
          </div>
        )

      case 'error':
        return (
          <div key={index} className="msg-wrapper msg-error-wrapper">
            <div className="msg msg-error"><ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.text}</ReactMarkdown></div>
          </div>
        )

      case 'info':
        return (
          <div key={index} className="msg-wrapper msg-system-wrapper">
            <div className="msg msg-system">{entry.text}</div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="container">
      <h1>数据分析</h1>

      <div className="chat" ref={chatRef}>
        {chatLog.length === 0 && (
          <div className="welcome-guide">
            <div className="welcome-header">
              <span className="welcome-icon">📊</span>
              <span>数据分析</span>
            </div>
            <div className="welcome-questions">
              <h3>输入分析问题，例如:</h3>
              <div className="question-chips">
                {[
                  '分析一下材料变更情况',
                  '查询胶量设定记录',
                  '追踪材料生命周期',
                  '查看MP压力辊参数',
                ].map((q, i) => (
                  <button key={i} className="question-chip" onClick={() => setInputMessage(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {chatLog.map((entry, i) => renderEntry(entry, i))}
      </div>

      <div className="composer">
        <div className="composer-actions" style={{ flexDirection: 'column', gap: 8 }}>
          <textarea
            style={{ width: '100%', boxSizing: 'border-box', padding: 10, border: '1px solid #e2e8f0', borderRadius: 8, fontFamily: 'inherit', fontSize: 14, resize: 'none', height: 52 }}
            placeholder="输入分析问题..."
            value={inputMessage}
            onChange={e => setInputMessage(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }}
            disabled={isQuerying || isExecuting}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', width: '100%' }}>
            <button className="btn btn-primary" onClick={handleSend} disabled={isQuerying || isExecuting || !inputMessage.trim()}>
              {isQuerying ? <span className="spinner" /> : '发送'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
