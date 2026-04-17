# API参考

## 概述

IPS系统提供基于FastAPI的RESTful API接口，支持与外部系统集成。

## 基本信息

| 属性 | 值 |
|------|-----|
| 框架 | FastAPI |
| 文档地址 | /docs |
| CORS | 启用 |
| 中间件 | Human-in-the-loop |

---

## 基础配置

### CORS配置

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 模型配置

```python
ChatOpenAI(
    temperature=0.5,
    openai_api_base="http://127.0.0.1:8080/v1",
    openai_api_key="ed"
)
```

---

## 端点总览

### 主端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/chat` | POST | 聊天接口 |
| `/docs` | GET | API文档 |
| `/openapi.json` | GET | OpenAPI schema |

---

## 聊天接口

### POST /chat

**功能**

与RAG系统的聊天接口，支持人类介入确认。

**请求头**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| Content-Type | string | 是 | application/json |

**请求体**

```json
{
  "message": "用户问题",
  "thread_id": "可选的线程ID"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| message | string | 是 | 用户发送的消息 |
| thread_id | string | 否 | 会话线程ID，用于多轮对话 |

**响应 (正常)**

```json
{
  "response": "系统回复内容",
  "thread_id": "线程ID"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| response | string | 系统回复 |
| thread_id | string | 线程ID |

**响应 (需要确认)**

```json
{
  "status": "awaiting_approval",
  "tool_name": "execute_code",
  "tool_args": {
    "code": "要执行的代码"
  },
  "approval_required": true
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | awaiting_approval |
| tool_name | string | 工具名称 |
| tool_args | object | 工具参数 |
| approval_required | boolean | 需要确认 |

**示例请求**

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "解释一下强换操作"}'
```

**示例响应**

```json
{
  "response": "强换功能用于在生管内强换至下一单...",
  "thread_id": "abc-123"
}
```

---

## 工具定义

### execute_code

**工具名称**

```
execute_code
```

**功能**

执行Python代码（需要用户确认）。

**参数模式**

```python
class CodeInput(BaseModel):
    code: str = Field(description="Code to execute")
```

**参数说明**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| code | string | 是 | 要执行的Python代码 |

**返回值**

```json
{
  "result": "执行结果"
}
```

**中断确认**

调用此工具时会触发中断，需要用户确认：

```json
{
  "approved": true,
  "modified_args": {
    "code": "用户修改后的代码"
  }
}
```

或拒绝：

```json
{
  "approved": false
}
```

---

## 状态类型

### AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
```

**字段**

| 字段 | 类型 | 说明 |
|------|------|------|
| messages | List[BaseMessage] | 消息列表 |

---

## ���误响应

### 400 - 请求错误

```json
{
  "detail": "错误详情"
}
```

### 500 - 服务器错误

```json
{
  "detail": "服务器内部错误"
}
```

---

## 使用示例

### Python调用示例

```python
import requests

url = "http://localhost:8000/chat"
payload = {"message": "如何进行强换操作"}
response = requests.post(url, json=payload)
print(response.json())
```

### JavaScript调用示例

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message: '解释一下强换操作'})
});
const data = await response.json();
console.log(data.response);
```

---

## WebSocket支持

### 描述

系统支持WebSocket连接用于流式响应。

### 连接方式

```
ws://localhost:8000/ws?thread_id=<thread_id>
```

---

## 速率限制

### 说明

| 限制项 | 值 |
|--------|------|
| 免费用户 | 60请求/小时 |
| 认证用户 | 1000请求/小时 |

---

## 版本信息

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2024-01 | 初始版本 |
| v1.1 | 2024-06 | 增加Human-in-the-loop |

---

## 相关文档

| 文档 | 说明 |
|------|------|
| `01_system_overview.md` | 系统概述 |
| `06_rag_pipeline.md` | RAG系统架构 |