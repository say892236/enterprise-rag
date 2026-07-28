// client.ts


import axios from 'axios'

import type {
  ApiResponse,
  ChatRequest,
  ChatResponse,
  ChatStreamDoneData,
  ChatStreamErrorData,
  ChatStreamHandlers,
  ChatStreamSourcesData,
  ChatStreamStatusData,
  ChatStreamTokenData,
  CreateSessionResponse,
  DeleteSessionResponse,
  GetSessionMessagesResponse,
  GetSessionsResponse,
  HealthResponse,
  LoginResponse,
  ReadinessResponse,
} from '../types/api'

const API_BASE_URL = 'http://127.0.0.1:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
})

/**
 * 请求拦截器。
 *
 * 每次请求后端前，从 localStorage 读取 token，
 * 然后放到 Authorization 请求头里。
 *
 * 后端 /chat 依赖这个请求头：
 * Authorization: Bearer <access_token>
 */
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')

  if (token && token !== 'undefined' && token !== 'null') {
    config.headers.Authorization = `Bearer ${token}`
  }

  return config
})

/**
 * 登录。
 *
 * 后端 /auth/login 当前接收 JSON：
 * {
 *   username: string,
 *   password: string
 * }
 *
 * 兼容两种返回格式：
 * 1. 直接返回 LoginResponse
 * 2. 返回统一 ApiResponse<LoginResponse>
 */
export async function login(
  username: string,
  password: string,
): Promise<LoginResponse> {
  const response = await apiClient.post<LoginResponse | ApiResponse<LoginResponse>>(
    '/auth/login',
    {
      username,
      password,
    },
    {
      headers: {
        'Content-Type': 'application/json',
      },
    },
  )

  const responseData = response.data

  if ('success' in responseData) {
    if (!responseData.success || !responseData.data) {
      throw new Error(responseData.message || '登录失败')
    }

    return responseData.data
  }

  return responseData
}

/**
 * 企业知识库问答。
 */
export async function askQuestion(request: ChatRequest): Promise<ChatResponse> {
  const response = await apiClient.post<ApiResponse<ChatResponse>>('/chat', request)

  if (!response.data.success || !response.data.data) {
    throw new Error(response.data.message || '问答请求失败')
  }

  return response.data.data
}


/**
 * 解析单个 SSE 事件块。
 *
 * 后端返回格式：
 *
 * event: token
 * data: {"content":"员工"}
 *
 * 每个事件之间使用空行分隔。
 */
function parseSSEBlock(
  block: string,
): {
  event: string
  data: string
} | null {
  let eventName = 'message'
  const dataLines: string[] = []

  for (const rawLine of block.split('\n')) {
    const line = rawLine.trimEnd()

    // 空行和 SSE 注释行不需要处理
    if (!line || line.startsWith(':')) {
      continue
    }

    const separatorIndex = line.indexOf(':')

    const field = separatorIndex === -1
      ? line
      : line.slice(0, separatorIndex)

    let value = separatorIndex === -1
      ? ''
      : line.slice(separatorIndex + 1)

    // SSE 规范允许冒号后存在一个空格
    if (value.startsWith(' ')) {
      value = value.slice(1)
    }

    if (field === 'event') {
      eventName = value
    }

    if (field === 'data') {
      dataLines.push(value)
    }
  }

  if (dataLines.length === 0) {
    return null
  }

  return {
    event: eventName,
    data: dataLines.join('\n'),
  }
}


/**
 * 从非 2xx HTTP 响应中提取错误信息。
 */
async function getHTTPErrorMessage(
  response: Response,
): Promise<string> {
  const defaultMessage = `流式问答请求失败，HTTP 状态码：${response.status}`

  try {
    const errorData = await response.json() as {
      message?: string
      detail?: string | {
        message?: string
      }
    }

    if (errorData.message) {
      return errorData.message
    }

    if (typeof errorData.detail === 'string') {
      return errorData.detail
    }

    if (
      typeof errorData.detail === 'object'
      && errorData.detail?.message
    ) {
      return errorData.detail.message
    }

    return defaultMessage
  } catch {
    return defaultMessage
  }
}


/**
 * 企业知识库 SSE 流式问答。
 *
 * 由于该接口需要：
 * 1. POST 请求
 * 2. JSON 请求体
 * 3. Authorization 请求头
 *
 * 因此不能直接使用浏览器原生 EventSource，
 * 这里使用 fetch + ReadableStream 手动读取 SSE。
 */
export async function askQuestionStream(
  request: ChatRequest,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const token = localStorage.getItem('token')

  const headers: Record<string, string> = {
    Accept: 'text/event-stream',
    'Content-Type': 'application/json',
  }

  if (token && token !== 'undefined' && token !== 'null') {
    headers.Authorization = `Bearer ${token}`
  }

  const response = await fetch(
    `${API_BASE_URL}/chat/stream`,
    {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
      signal,
    },
  )

  // 401、403、422、500 等普通 HTTP 错误
  if (!response.ok) {
  const message = await getHTTPErrorMessage(response)

  const error = new Error(message) as Error & {
    status?: number
  }

  error.status = response.status

  throw error
}
  if (!response.body) {
    throw new Error('浏览器未获取到流式响应内容')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')

  // 保存尚未组成完整 SSE 事件的数据
  let buffer = ''

  // done 事件中会返回最终完整响应
  let finalResponse: ChatResponse | null = null

  /**
   * 处理一个完整 SSE 事件。
   */
  const handleEventBlock = (block: string): void => {
    const parsedEvent = parseSSEBlock(block)

    if (!parsedEvent) {
      return
    }

    let eventData: unknown

    try {
      eventData = JSON.parse(parsedEvent.data)
    } catch {
      throw new Error(
        `SSE 数据解析失败：${parsedEvent.data}`,
      )
    }

    switch (parsedEvent.event) {
      case 'status': {
        handlers.onStatus?.(
          eventData as ChatStreamStatusData,
        )
        break
      }

      case 'token': {
        handlers.onToken?.(
          eventData as ChatStreamTokenData,
        )
        break
      }

      case 'sources': {
        handlers.onSources?.(
          eventData as ChatStreamSourcesData,
        )
        break
      }

      case 'done': {
        const doneData = eventData as ChatStreamDoneData

        finalResponse = doneData.data
        handlers.onDone?.(doneData)
        break
      }

      case 'error': {
        const errorData = eventData as ChatStreamErrorData

        handlers.onError?.(errorData)

        // 让调用方的 catch 同样能够捕获 SSE 业务错误
        throw new Error(
          errorData.message || '流式问答执行失败',
        )
      }

      default:
        // 未知事件暂时忽略，避免后端未来新增事件时前端直接崩溃
        break
    }
  }

  try {
    while (true) {
      const {
        value,
        done,
      } = await reader.read()

      if (done) {
        // 将 TextDecoder 中可能残留的字符取出
        buffer += decoder.decode()
        break
      }

      buffer += decoder.decode(
        value,
        {
          stream: true,
        },
      )

      /*
       * 统一换行符。
       *
       * 后端通常使用 \n\n 分隔事件，
       * 但部分代理或系统可能返回 \r\n\r\n。
       */
      buffer = buffer.replace(/\r\n/g, '\n')

      let separatorIndex = buffer.indexOf('\n\n')

      while (separatorIndex !== -1) {
        const eventBlock = buffer.slice(
          0,
          separatorIndex,
        )

        buffer = buffer.slice(
          separatorIndex + 2,
        )

        handleEventBlock(eventBlock)

        separatorIndex = buffer.indexOf('\n\n')
      }
    }

    // 处理连接关闭前可能残留的最后一个事件
    if (buffer.trim()) {
      handleEventBlock(buffer.trim())
    }
  } finally {
    reader.releaseLock()
  }

  if (!finalResponse) {
    throw new Error(
      '流式连接已结束，但没有收到 done 事件',
    )
  }

  return finalResponse
}

/**
 * 健康检查。
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health')
  return response.data
}

/**
 * 就绪检查。
 */
export async function getReadiness(): Promise<ReadinessResponse> {
  const response = await apiClient.get<ReadinessResponse>('/readiness')
  return response.data
}




export async function getSessions(): Promise<GetSessionsResponse> {

  const response = await apiClient.get<GetSessionsResponse>('/chat/sessions')

  return response.data
}


// 创建新聊天
export async function createSession(): Promise<CreateSessionResponse> {

  const response = await apiClient.post<CreateSessionResponse>('/chat/session/new')

  return response.data
}


// 删除聊天
export async function deleteSession(
  sessionId: string,
): Promise<DeleteSessionResponse> {

  const response = await apiClient.delete<DeleteSessionResponse>(
    `/chat/session/${sessionId}`,
  )

  return response.data
}

export async function getSessionMessages(
  sessionId: string,
): Promise<GetSessionMessagesResponse> {
  const response = await apiClient.get<GetSessionMessagesResponse>(
    `/chat/session/${sessionId}`,
  )

  return response.data
}
