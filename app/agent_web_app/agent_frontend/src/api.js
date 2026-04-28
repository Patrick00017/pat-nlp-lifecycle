import { fetchEventSource } from '@microsoft/fetch-event-source';

const BASE = 'http://localhost:8000'

export async function sendChat(message, threadId = null, mode = 'IPS') {
  const body = { message }
  if (threadId){
    body.thread_id = threadId
  }
  else{
    body.thread_id = crypto.randomUUID()
  }

  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Chat request failed: ${res.status}`)
  return res.json()
}

export async function sendChatStream(message, threadId = null, mode = 'IPS') {
  const body = { message }
  if (threadId){
    body.thread_id = threadId
  }
  else{
    body.thread_id = crypto.randomUUID()
  }

  const res = await fetch(`${BASE}/chat/stream`, {
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
  console.log("resumeChat -> body: " + JSON.stringify(body))
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

export async function fetchTools() {
  const res = await fetch(`${BASE}/tool/list`)
  if (!res.ok) throw new Error(`Fetch tools failed: ${res.status}`)
  return res.json()
}

export async function connectSSE(url, payload, onMessage, onError) {
  try {
    await fetchEventSource(url, {
      method: 'POST',
      openWhenHidden: true,
      headers: {
        // 'Authorization': 'Bearer YOUR_TOKEN',
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload),
      onopen: async (response) => {
        console.log('SSE connection opened:', response);
        if (response.ok && response.headers.get('content-type')?.includes('text/event-stream')) {
          console.log('Connection is ready to receive events.');
        } else {
          throw new Error('Unexpected response when opening SSE connection.');
        }
      },
      onmessage: (event) => {
        // console.log('Received message:', event.data);
        // 处理服务器发送的数据
        onMessage(event.data)
      },
      onerror: (err) => {
        // console.error('SSE error occurred:', err);
        // 可在此重试连接
        onError(err)
      },
      onclose: () => {
        console.log('SSE connection closed');
      },
    });
  } catch (error) {
    console.error('Failed to establish SSE connection:', error);
  }
};
