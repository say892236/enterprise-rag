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
  rerank_score?: number
  hybrid_score?: number
  bm25_score?: number
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
  question: string
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