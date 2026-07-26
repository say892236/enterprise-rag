import axios from 'axios'

import type {
  ApiResponse,
  ChatRequest,
  ChatResponse,
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