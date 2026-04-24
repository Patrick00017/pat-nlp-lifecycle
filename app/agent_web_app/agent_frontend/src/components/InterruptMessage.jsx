import React from 'react'

function InterruptMessage({ interrupt, modifiedArgsText, setModifiedArgsText, modifiedArgsSchema, onApprove, onReject, isLoading, tools, onToolChange, originalToolName, originalArgsText, originalSchema }) {
  const argsObj = React.useMemo(() => {
    try { return JSON.parse(modifiedArgsText) } catch { return {} }
  }, [modifiedArgsText])

  const handleFieldChange = (key, value) => {
    const newArgs = { ...argsObj, [key]: value }
    setModifiedArgsText(JSON.stringify(newArgs, null, 2))
  }

  const getTypeBadge = (type) => {
    if (type.includes("bool")) return <span className="type-badge type-bool">bool</span>
    if (type.includes("int")) return <span className="type-badge type-int">int</span>
    if (type.includes("float")) return <span className="type-badge type-float">float</span>
    return <span className="type-badge type-str">str</span>
  }

  const handleToolChange = (newToolName) => {
    const selectedTool = tools.find(t => t.name === newToolName)
    if (selectedTool && onToolChange) {
      const defaultArgs = {}
      Object.keys(selectedTool.schema || {}).forEach(key => {
        defaultArgs[key] = ''
      })
      setModifiedArgsText(JSON.stringify(defaultArgs, null, 2))
      onToolChange(newToolName, selectedTool.schema)
    }
  }

  const handleResetToOriginal = () => {
    setModifiedArgsText(originalArgsText)
    onToolChange(originalToolName, originalSchema)
  }

  const currentToolName = interrupt.tool_name || interrupt.tool || ''

  const renderField = (key, value, type) => {
    const baseType = type.includes("bool") ? "bool" : type.includes("int") || type.includes("float") ? "number" : "str"

    if (baseType === "bool") {
      return (
        <label className="arg-toggle">
          <input
            type="checkbox"
            checked={value || false}
            onChange={(e) => handleFieldChange(key, e.target.checked)}
            disabled={isLoading}
          />
          <span className="toggle-slider"></span>
          <span className="toggle-label">{value ? "ON" : "OFF"}</span>
        </label>
      )
    }
    if (baseType === "number") {
      return (
        <input
          className="arg-input"
          type="number"
          value={value ?? ""}
          onChange={(e) => handleFieldChange(key, parseFloat(e.target.value) || 0)}
          disabled={isLoading}
        />
      )
    }
    return (
      <textarea
        className="arg-textarea"
        value={value ?? ""}
        onChange={(e) => handleFieldChange(key, e.target.value)}
        rows={1}
        disabled={isLoading}
      />
    )
  }

  return (
    <div className="interrupt-card">
      <div className="interrupt-header">
        <span className="interrupt-icon">⚠️</span>
        <h3>Tool Call Requires Approval</h3>
      </div>
      <div className="interrupt-body">
        <div className="interrupt-tool">
          <span className="label">Tool</span>
          {tools && tools.length > 0 ? (
            <div className="tool-select-row">
              <select
                className="tool-select"
                value={currentToolName}
                onChange={(e) => handleToolChange(e.target.value)}
                disabled={isLoading}
              >
                {tools.map(tool => (
                  <option key={tool.name} value={tool.name}>{tool.name}</option>
                ))}
              </select>
              {originalToolName && (
                <button
                  className="btn btn-reset"
                  onClick={handleResetToOriginal}
                  disabled={isLoading}
                  title="Reset to original tool and arguments"
                >
                  ↺ Reset
                </button>
              )}
            </div>
          ) : (
            <div className="tool-badge">
              <svg className="tool-icon" viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
                <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd" />
              </svg>
              <span className="tool-name">{currentToolName}</span>
            </div>
          )}
        </div>
        <div className="interrupt-args">
          <span className="label">Arguments</span>
          {modifiedArgsSchema && Object.keys(modifiedArgsSchema).length > 0 ? (
            <div className="arg-fields">
              {Object.entries(modifiedArgsSchema).map(([key, type]) => (
                <div key={key} className="arg-row">
                  <div className="arg-label">
                    <span className="arg-key">{key} {getTypeBadge(type)}</span>
                  </div>
                  {renderField(key, argsObj[key], type)}
                </div>
              ))}
            </div>
          ) : (
            <textarea
              className="args-textarea"
              value={modifiedArgsText}
              onChange={(e) => setModifiedArgsText(e.target.value)}
              rows={4}
              disabled={isLoading}
            />
          )}
        </div>
      </div>
      <div className="interrupt-actions">
        <button className="btn btn-reject" onClick={onReject} disabled={isLoading}>
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
          {isLoading ? <span className="spinner"></span> : 'Reject'}
        </button>
        <button className="btn btn-approve" onClick={onApprove} disabled={isLoading}>
          <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
          {isLoading ? <span className="spinner"></span> : 'Approve & Run'}
        </button>
      </div>
    </div>
  )
}

export default InterruptMessage