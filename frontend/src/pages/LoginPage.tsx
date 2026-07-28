// src/pages/LoginPage.tsx

import { useState } from 'react'
import { login } from '../api/client'


interface LoginPageProps {
  onLoginSuccess: () => void
}

/**
 * 后端写入 JWT 的用户身份数据。
 */
interface JwtPayload {
  sub?: string
  user_id?: string | number
  role?: string
  exp?: number
}

/**
 * 解析 JWT Payload。
 *
 * 后端已经把 username、user_id 和 role 写入 Token，
 * 前端直接使用 Token 中的真实角色，不再根据用户名猜测。
 */
function decodeJwtPayload(token: string): JwtPayload {
  const payloadPart = token.split('.')[1]

  if (!payloadPart) {
    throw new Error('登录 Token 格式无效')
  }

  // JWT 使用 Base64URL，需要先转换成普通 Base64
  const base64 = payloadPart
    .replace(/-/g, '+')
    .replace(/_/g, '/')

  // 补齐 Base64 末尾的等号
  const paddedBase64 = base64.padEnd(
    Math.ceil(base64.length / 4) * 4,
    '=',
  )

  // 转成 UTF-8 字符串后解析 JSON
  const bytes = Uint8Array.from(
    atob(paddedBase64),
    (char) => char.charCodeAt(0),
  )

  const jsonText = new TextDecoder().decode(bytes)

  return JSON.parse(jsonText) as JwtPayload
}


/**
 * 登录页面。
 *
 * 用户输入 username/password 后，请求后端 /auth/login。
 * 登录成功后，将 access_token 保存到 localStorage。
 */
export function LoginPage({ onLoginSuccess }: LoginPageProps) {
  const [username, setUsername] = useState('say')
  const [password, setPassword] = useState('123456')
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  async function handleLogin() {
    setLoading(true)
    setErrorMessage('')

    try {
  const loginResult = await login(username, password)

if (!loginResult.access_token) {
  throw new Error('后端未返回 access_token')
}

// 从后端签发的 JWT 中读取真实身份和角色
const tokenPayload = decodeJwtPayload(
  loginResult.access_token,
)

const loginUsername =
  loginResult.username
  || tokenPayload.sub
  || username

const loginRole =
  loginResult.role
  || tokenPayload.role

if (!loginRole) {
  throw new Error('登录 Token 中缺少用户角色')
}

localStorage.setItem(
  'token',
  loginResult.access_token,
)

localStorage.setItem(
  'username',
  loginUsername,
)

localStorage.setItem(
  'role',
  loginRole,
)

// ===============================
// 登录成功后创建新的聊天会话
// 避免进入系统后直接加载上一次聊天
// ===============================

onLoginSuccess()

} catch (error) {
  console.error(error)
  setErrorMessage('登录失败，请检查用户名、密码或后端服务状态')
} finally {
  setLoading(false)
    }
  }



  return (
    <div className="login-page">
      <div className="login-card">
         <div className="login-header">
  <h1 className="login-title">企业知识库问答系统</h1>
  <p className="login-subtitle">
    登录后可查询企业制度、差旅报销、请假与安全合规内容
  </p>
    </div>
        <div className="form-group">
          <label>用户名</label>
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            placeholder="请输入用户名"
          />
        </div>

        <div className="form-group">
          <label>密码</label>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="请输入密码"
          />
        </div>

        {errorMessage && <div className="error-box">{errorMessage}</div>}

        <button className="primary-button" onClick={handleLogin} disabled={loading}>
          {loading ? '登录中...' : '登录'}
        </button>
      </div>
    </div>
  )
}
