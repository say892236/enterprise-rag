// src/api/client.ts

import axios from 'axios'
import type {
  ApiResponse,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  LoginResponse,
  ReadinessResponse,
} from '../types/api'

/**
 * 后端 API 地址。
 *
 * 当前开发阶段：
 * - FastAPI: http://127.0.0.1:8000
 * - React:   http://localhost:5173
 */
const API_BASE_URL = 'http://127.0.0.1:8000'

/**
 * 创建 axios 实例，统一配置后端地址和超时时间。
 */
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
})

/**
 * 请求拦截器。
 *
 * 每次请求前，从 localStorage 读取 token。
 * 如果 token 存在，就自动加上 Authorization 请求头。
 */
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')

  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }

  return config
})

/**
 * 登录。
 *
 * 后端如果使用 OAuth2PasswordRequestForm，
 * 请求格式必须是 application/x-www-form-urlencoded，
 * 不能直接传 JSON。
 */
/**
 * 登录。
 *
 * 当前后端 /auth/login 使用 JSON 请求体：
 * {
 *   "username": "...",
 *   "password": "..."
 * }
 *
 * 这里同时兼容两种返回格式：
 * 1. 直接返回 token 数据：
 *    { access_token: "...", token_type: "bearer" }
 *
 * 2. 统一响应格式：
 *    { success: true, data: { access_token: "...", token_type: "bearer" } }
 */
export async function login(username: string, password: string): Promise<LoginResponse> {
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
 * 调用企业知识库问答接口。
 */
export async function askQuestion(request: ChatRequest): Promise<ChatResponse> {
  const response = await apiClient.post<ApiResponse<ChatResponse>>('/chat', request)

  if (!response.data.success || !response.data.data) {
    throw new Error(response.data.message || '问答请求失败')
  }

  return response.data.data
}

/**
 * 调用 /health 存活检查。
 */
export async function getHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health')
  return response.data
}

/**
 * 调用 /readiness 就绪检查。
 */
export async function getReadiness(): Promise<ReadinessResponse> {
  const response = await apiClient.get<ReadinessResponse>('/readiness')
  return response.data
}