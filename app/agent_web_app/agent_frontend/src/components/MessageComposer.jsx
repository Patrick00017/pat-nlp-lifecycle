export function MessageComposer({ message, setMessage, mode, setMode, isLoading, onSend }) {
  return (
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
        <button className="btn btn-primary" onClick={onSend} disabled={isLoading}>
          {isLoading ? <span className="spinner"></span> : 'Send'}
        </button>
      </div>
    </div>
  )
}

export default MessageComposer