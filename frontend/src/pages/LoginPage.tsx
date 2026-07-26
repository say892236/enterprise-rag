// src/pages/LoginPage.tsx

import { useState } from 'react'
import { login } from '../api/client'

interface LoginPageProps {
  onLoginSuccess: () => void
}

/**
 * 登录页面。
 *
 * 用户输入 username/password 后，请求后端 /auth/login。
 * 登录成功后，将 access_token 保存到 localStorage。
 */
export function LoginPage({ onLoginSuccess }: LoginPageProps) {
  const [username, setUsername] = useState('ay')
  const [password, setPassword] = useState('123456')
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  async function handleLogin() {
    setLoading(true)
    setErrorMessage('')

    try {
      const data = await login(username, password)

      localStorage.setItem('access_token', data.access_token)

      if (data.username) {
        localStorage.setItem('username', data.username)
      }

      if (data.role) {
        localStorage.setItem('user_role', data.role)
      }

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