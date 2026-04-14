const BASE = 'http://localhost:8000'

export async function sendChat(message, threadId = null) {
  const body = { message }
  if (threadId) body.thread_id = threadId

  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`)
  return res.json()
}

export async function resumeChat(threadId, approved = true, modified_args = null) {
  const body = { thread_id: threadId, approved }
  if (modified_args) body.modified_args = modified_args

  const res = await fetch(`${BASE}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Resume request failed: ${res.status}`)
  return res.json()
}

export async function health() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}
