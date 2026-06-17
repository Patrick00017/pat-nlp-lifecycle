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

export async function funcQuery(messages, maxTokens = 128) {
  const res = await fetch(`${BASE}/func/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, max_tokens: maxTokens }),
  })
  if (!res.ok) throw new Error(`Func query failed: ${res.status}`)
  return res.json()
}

export async function funcCall(toolCalls) {
  const res = await fetch(`${BASE}/func/call`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool_calls: toolCalls }),
  })
  if (!res.ok) throw new Error(`Func call failed: ${res.status}`)
  return res.json()
}

export async function analysisInit(toolName, toolArgs, toolResult = null) {
  const res = await fetch(`${BASE}/analysis/init`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool_name: toolName, tool_args: toolArgs, tool_result: toolResult }),
  })
  if (!res.ok) throw new Error(`Analysis init failed: ${res.status}`)
  return res.json()
}

export async function analysisStep(stateId, nodeId, args = {}) {
  const body = { state_id: stateId, node_id: nodeId };
  if (Object.keys(args).length > 0) body.args = args;
  const res = await fetch(`${BASE}/analysis/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`Analysis step failed: ${res.status}`)
  return res.json()
}

export async function analysisStepback(stateId) {
  const res = await fetch(`${BASE}/analysis/stepback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state_id: stateId }),
  })
  if (!res.ok) throw new Error(`Analysis stepback failed: ${res.status}`)
  return res.json()
}

export async function listOpenCodeAgents() {
  const res = await fetch(`${BASE}/opencode/agents`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!res.ok) throw new Error(`List agents failed: ${res.status}`)
  return res.json()
}

export async function createOpenCodeSession(agent = 'general') {
  const res = await fetch(`${BASE}/opencode/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ agent }),
  })
  if (!res.ok) throw new Error(`Create session failed: ${res.status}`)
  return res.json()
}

export async function listOpenCodeSessions(limit = 20) {
  const res = await fetch(`${BASE}/opencode/sessions?limit=${limit}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!res.ok) throw new Error(`List sessions failed: ${res.status}`)
  return res.json()
}

export async function getOpenCodeSession(sessionId) {
  const res = await fetch(`${BASE}/opencode/session/${sessionId}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!res.ok) throw new Error(`Get session failed: ${res.status}`)
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
