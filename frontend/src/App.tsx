// src/App.tsx

import { useState } from 'react'
import { ChatPage } from './pages/ChatPage'
import { LoginPage } from './pages/LoginPage'
import './App.css'

/**
 * 应用入口组件。
 *
 * 根据 localStorage 中是否存在 token，
 * 判断当前展示登录页还是问答页。
 */
function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(
    Boolean(localStorage.getItem('token')),
  )

  function handleLoginSuccess() {
    setIsLoggedIn(true)
  }

  function handleLogout() {
    localStorage.removeItem('token')
    localStorage.removeItem('access_token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')
    localStorage.removeItem('user_role')
    setIsLoggedIn(false)
  }

  if (!isLoggedIn) {
    return <LoginPage onLoginSuccess={handleLoginSuccess} />
  }

  return <ChatPage onLogout={handleLogout} />
}

export default App
