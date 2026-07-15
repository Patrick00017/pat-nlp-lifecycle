import { useState, useEffect, useRef } from 'react'
import { getKBFiles, uploadKBFiles, buildKB, deleteKBFile } from '../api'

const COLORS = {
  primary: '#3b82f6',
  success: '#10b981',
  danger: '#ef4444',
  warn: '#f59e0b',
  border: '#e5e7eb',
  bg: '#f9fafb',
  text: '#374151',
  muted: '#6b7280',
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

export default function KbManager() {
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [building, setBuilding] = useState(false)
  const [status, setStatus] = useState(null)
  const fileInputRef = useRef(null)

  useEffect(() => { refreshFiles() }, [])

  async function refreshFiles() {
    setLoading(true)
    try {
      const res = await getKBFiles()
      setFiles(res.files || [])
    } catch (e) {
      setStatus({ type: 'error', text: '获取文件列表失败: ' + e.message })
    } finally {
      setLoading(false)
    }
  }

  async function handleSelectFiles() {
    fileInputRef.current?.click()
  }

  async function handleFilesChosen(e) {
    const selected = Array.from(e.target.files || [])
    if (selected.length === 0) return
    setUploading(true)
    setStatus(null)
    try {
      const res = await uploadKBFiles(selected)
      setStatus({ type: 'ok', text: `已保存 ${res.files.length} 个文件` })
      refreshFiles()
    } catch (e) {
      setStatus({ type: 'error', text: '上传失败: ' + e.message })
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  async function handleDelete(name) {
    try {
      await deleteKBFile(name)
      refreshFiles()
      setStatus({ type: 'ok', text: `已删除 ${name}` })
    } catch (e) {
      setStatus({ type: 'error', text: '删除失败: ' + e.message })
    }
  }

  async function handleBuild() {
    if (files.length === 0) {
      setStatus({ type: 'warn', text: '请先上传文件' })
      return
    }
    setBuilding(true)
    setStatus({ type: 'info', text: '正在构建知识库 ...' })
    try {
      const res = await buildKB()
      if (res.status === 'ok') {
        setStatus({ type: 'ok', text: `构建完成! 共生成了 ${res.chunk_count} 个文档块` })
      } else {
        setStatus({ type: 'error', text: '构建失败: ' + (res.detail || '未知错误') })
      }
    } catch (e) {
      setStatus({ type: 'error', text: '构建请求失败: ' + e.message })
    } finally {
      setBuilding(false)
    }
  }

  async function handleDrop(e) {
    e.preventDefault()
    const dropped = Array.from(e.dataTransfer.files || [])
    if (dropped.length === 0) return
    setUploading(true)
    setStatus(null)
    try {
      const res = await uploadKBFiles(dropped)
      setStatus({ type: 'ok', text: `已保存 ${res.files.length} 个文件` })
      refreshFiles()
    } catch (e) {
      setStatus({ type: 'error', text: '上传失败: ' + e.message })
    } finally {
      setUploading(false)
    }
  }

  function handleDragOver(e) {
    e.preventDefault()
  }

  const statusBg = status?.type === 'ok' ? '#d1fae5' :
    status?.type === 'error' ? '#fee2e2' :
    status?.type === 'warn' ? '#fef3c7' :
    '#dbeafe'
  const statusColor = status?.type === 'ok' ? '#065f46' :
    status?.type === 'error' ? '#991b1b' :
    status?.type === 'warn' ? '#92400e' :
    '#1e40af'

  if (loading) {
    return (
      <div style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: COLORS.muted,
        fontSize: 13,
      }}>
        <span style={{
          display: 'inline-block',
          width: 18,
          height: 18,
          border: '2px solid #3b82f6',
          borderTopColor: 'transparent',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
          marginBottom: 8,
        }} />
        加载中 ...
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    )
  }

  return (
    <div style={{ padding: '8px 12px', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ marginBottom: 4, fontSize: 13, color: COLORS.muted }}>
        知识库管理
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        style={{
          border: `2px dashed ${COLORS.border}`,
          borderRadius: 8,
          padding: '16px 12px',
          textAlign: 'center',
          background: COLORS.bg,
          cursor: 'pointer',
          marginBottom: 10,
          transition: 'border-color 0.2s',
        }}
        onClick={handleSelectFiles}
      >
        <div style={{ fontSize: 28, marginBottom: 4 }}>&#128206;</div>
        <div style={{ fontSize: 12, color: COLORS.muted }}>
          拖拽文件到此处，或点击选择
        </div>
        <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 2 }}>
          支持 PDF / TXT / MD / DOCX
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md,.docx"
          onChange={handleFilesChosen}
          style={{ display: 'none' }}
        />
      </div>

      {files.length > 0 && (
        <div style={{
          flex: 1,
          overflow: 'auto',
          border: `1px solid ${COLORS.border}`,
          borderRadius: 6,
          marginBottom: 10,
        }}>
          {files.map((f, i) => (
            <div key={f.name} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '6px 10px',
              borderBottom: i < files.length - 1 ? `1px solid ${COLORS.border}` : 'none',
              fontSize: 12,
            }}>
              <span style={{ color: COLORS.text }}>{f.name}</span>
              <span style={{ color: COLORS.muted, marginLeft: 8, flex: 1, textAlign: 'right' }}>
                {fmtSize(f.size)}
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); handleDelete(f.name) }}
                style={{
                  marginLeft: 8,
                  background: 'none',
                  border: 'none',
                  color: COLORS.danger,
                  cursor: 'pointer',
                  fontSize: 14,
                  padding: '0 4px',
                  lineHeight: 1,
                }}
              >&#10005;</button>
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleSelectFiles}
          disabled={uploading || building}
          style={{
            flex: 1,
            padding: '7px 0',
            border: `1px solid ${COLORS.border}`,
            borderRadius: 6,
            background: '#fff',
            color: COLORS.text,
            cursor: uploading || building ? 'not-allowed' : 'pointer',
            fontSize: 12,
            opacity: uploading || building ? 0.5 : 1,
          }}
        >
          {uploading ? '上传中 ...' : '📎 选择文件'}
        </button>
        <button
          onClick={handleBuild}
          disabled={uploading || building || files.length === 0}
          style={{
            flex: 1,
            padding: '7px 0',
            border: 'none',
            borderRadius: 6,
            background: building ? COLORS.muted : COLORS.primary,
            color: '#fff',
            cursor: (uploading || building || files.length === 0) ? 'not-allowed' : 'pointer',
            fontSize: 12,
            fontWeight: 600,
            opacity: (uploading || building || files.length === 0) ? 0.5 : 1,
          }}
        >
          {building ? '构建中 ...' : '🔨 构建知识库'}
        </button>
      </div>

      {status && (
        <div style={{
          marginTop: 8,
          padding: '6px 10px',
          borderRadius: 6,
          background: statusBg,
          color: statusColor,
          fontSize: 12,
          lineHeight: 1.5,
          whiteSpace: 'pre-wrap',
          maxHeight: 80,
          overflow: 'auto',
        }}>
          {building && <span style={{
            display: 'inline-block',
            width: 10,
            height: 10,
            border: '2px solid #3b82f6',
            borderTopColor: 'transparent',
            borderRadius: '50%',
            animation: 'spin 0.8s linear infinite',
            marginRight: 6,
            verticalAlign: 'middle',
          }} />}
          {status.text}
        </div>
      )}

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
