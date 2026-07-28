// src/types/api.ts

/**
 * 登录接口返回的数据结构。
 *
 * 后端一般会返回 access_token 和 token_type。
 * 这里额外兼容 username、role 等字段，方便后续扩展。
 */
export interface LoginResponse {
  access_token: string
  token_type?: string
  username?: string
  user_id?: string
  role?: string
}

/**
 * API 统一响应结构。
 *
 * 后端 ChatAPIResponse 通常是：
 * {
 *   success: true,
 *   data: {...}
 * }
 */
export interface ApiResponse<T> {
  success: boolean
  data?: T
  error_code?: string
  message?: string
  request_id?: string
}

/**
 * 引用来源结构。
 *
 * 对应后端 SourceItem。
 * 字段较多，这里只定义前端当前会展示的核心字段。
 */
export interface SourceItem {
  source?: string
  file_path?: string
  file_type?: string
  header_1?: string
  header_2?: string
  chunk_id?: string
  chunk_index?: number
  content_preview?: string
  content?: string
  page_content?: string
  score?: number
  bm25_score?: number
  hybrid_score?: number
  rerank_score?: number
  raw_rerank_score?: number
 rerank_boost?: number
 rerank_boost_reasons?: string[]
}

/**
 * Query Router 信息。
 */
export interface QueryRouteInfo {
  query_type?: string
  enable_rewrite?: boolean
  enable_rerank?: boolean
  reason?: string
}

/**
 * RAG 各阶段耗时。
 */
export interface RAGTimingInfo {
  route_seconds?: number
  rewrite_seconds?: number
  retrieve_seconds?: number
  rerank_seconds?: number
  confidence_seconds?: number
  compression_seconds?: number
  context_build_seconds?: number
  llm_seconds?: number
  total_chain_seconds?: number
}

/**
 * Context Compression 信息。
 */
export interface ContextCompressionInfo {
  enabled?: boolean
  original_chars?: number
  compressed_chars?: number
  compression_ratio?: number
}

/**
 * /chat 返回的数据结构。
 */
export interface ChatResponse {
  request_id: string
  username: string
  user_id: string
  user_role: string
  question: string
  rewritten_question?: string | null
  answer: string
  sources: SourceItem[]
  metadata_filter?: Record<string, unknown> | null
  query_route?: QueryRouteInfo | null
  timings?: RAGTimingInfo | null
  compression?: ContextCompressionInfo | null
  cost_time_seconds: number
}

/**
 * /chat 请求体。
 */
export interface ChatRequest {
  /**
   * 当前问题。
   */
  question: string
   /**
   * 当前聊天会话 ID。
   *
   * 前端左侧每一个历史会话都有自己的 session id。
   * 后端用它区分不同对话的 Redis 历史记录。
   */
  session_id?: string

  retrieve_top_k: number
  rerank_top_k: number
  enable_rewrite: boolean
  enable_rerank: boolean
  enable_compression: boolean
  metadata_filter?: Record<string, unknown> | null
}

/**
 * /health 返回结构。
 */
export interface HealthResponse {
  status: string
  service: string
}

/**
 * /readiness 返回结构。
 */
export interface ReadinessResponse {
  status: string
  service: string
  checks: Record<string, unknown>
}


/**
 * SSE 当前执行阶段。
 */
export type ChatStreamStage = 'retrieving' | 'generating'

/**
 * status 事件数据。
 *
 * retrieving：正在检索知识库
 * generating：正在生成回答
 */
export interface ChatStreamStatusData {
  stage: ChatStreamStage
  message: string
}

/**
 * token 事件数据。
 *
 * content 是大模型本次增量生成的文本，
 * 前端需要不断追加到同一条 AI 消息中。
 */
export interface ChatStreamTokenData {
  content: string
}

/**
 * sources 事件数据。
 */
export interface ChatStreamSourcesData {
  sources: SourceItem[]
}

/**
 * done 事件数据。
 *
 * data 与普通 /chat 接口返回的 ChatResponse 保持一致。
 */
export interface ChatStreamDoneData {
  success: boolean
  data: ChatResponse
}

/**
 * error 事件数据。
 */
export interface ChatStreamErrorData {
  success: false
  error_code?: string
  message: string
  request_id?: string
  error?: string
}

/**
 * 前端处理 SSE 事件时使用的回调。
 */
export interface ChatStreamHandlers {
  /**
   * 接收到检索或生成状态。
   */
  onStatus?: (data: ChatStreamStatusData) => void

  /**
   * 接收到大模型增量文本。
   */
  onToken?: (data: ChatStreamTokenData) => void

  /**
   * 接收到引用来源。
   */
  onSources?: (data: ChatStreamSourcesData) => void

  /**
   * 流式回答全部完成。
   */
  onDone?: (data: ChatStreamDoneData) => void

  /**
   * 后端通过 SSE 返回业务错误。
   */
  onError?: (data: ChatStreamErrorData) => void
}

/**
 * Redis 会话列表项。
 */
export interface RedisSession {
  session_id: string
  title: string
  time: string
}

/**
 * Redis 中保存的单条聊天消息。
 */
export interface RedisChatMessage {
  role: 'user' | 'assistant'
  content: string
  time?: string
}

/**
 * 创建会话接口响应。
 */
export interface CreateSessionResponse {
  success: boolean
  session_id: string
}

/**
 * 获取会话列表接口响应。
 */
export interface GetSessionsResponse {
  success: boolean
  sessions: RedisSession[]
}

/**
 * 获取会话历史接口响应。
 */
export interface GetSessionMessagesResponse {
  success: boolean
  messages: RedisChatMessage[]
}

/**
 * 删除会话接口响应。
 */
export interface DeleteSessionResponse {
  success: boolean
  message: string
}