import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { askQuestion } from '../api/client'
import type { ChatResponse, SourceItem } from '../types/api'

type ChatPageProps = {
  onLogout: () => void
}

/**
 * 单条聊天消息。
 *
 * role:
 * - user：用户消息
 * - assistant：AI 回复
 *
 * response:
 * - 只有 assistant 消息才可能携带后端完整响应
 * - 用于展示引用来源、调试信息等
 */
type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
}

/**
 * 一个前端会话。
 *
 * 当前阶段：
 * - 会话只保存在浏览器 localStorage
 * - 刷新页面后历史会话仍然存在
 *
 * 注意：
 * - 这还不是后端多轮记忆
 * - 后端暂时仍然是单轮 RAG 问答
 */
type ChatSession = {
  id: string
  title: string
  messages: ChatMessage[]
  updatedAt: number
}

const CHAT_SESSIONS_STORAGE_KEY = 'enterprise_rag_chat_sessions'

/**
 * 默认欢迎语。
 *
 * 每个新会话都会从这条助手消息开始。
 */
function createInitialMessages(): ChatMessage[] {
  return [
    {
      role: 'assistant',
      content:
        '你好，我是企业知识库问答助手。你可以询问公司概览、差旅、报销、请假、员工手册、IT 安全和权限合规相关问题。',
    },
  ]
}

/**
 * 创建一个前端会话 ID。
 *
 * 当前阶段使用前端生成的 session_id。
 * 后续如果接入后端多轮记忆，可以直接把这个 ID 传给后端。
 */
function createSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

/**
 * 创建一个新会话。
 */
function createNewSession(): ChatSession {
  return {
    id: createSessionId(),
    title: '新聊天',
    messages: createInitialMessages(),
    updatedAt: Date.now(),
  }
}

/**
 * 从 localStorage 读取历史会话。
 *
 * 如果没有历史记录，自动创建一个默认新会话。
 */
function loadChatSessions(): ChatSession[] {
  const rawValue = localStorage.getItem(CHAT_SESSIONS_STORAGE_KEY)

  if (!rawValue) {
    return [createNewSession()]
  }

  try {
    const parsedSessions = JSON.parse(rawValue) as ChatSession[]

    if (!Array.isArray(parsedSessions) || parsedSessions.length === 0) {
      return [createNewSession()]
    }

    return parsedSessions
  } catch {
    return [createNewSession()]
  }
}

/**
 * 保存历史会话到 localStorage。
 */
function saveChatSessions(sessions: ChatSession[]) {
  localStorage.setItem(CHAT_SESSIONS_STORAGE_KEY, JSON.stringify(sessions))
}

/**
 * 初始化聊天状态。
 *
 * 这里统一加载一次 sessions，避免 useState 分别初始化时生成两个不同的新会话。
 */
function loadInitialChatState() {
  const sessions = loadChatSessions()

  return {
    sessions,
    activeSessionId: sessions[0].id,
  }
}

/**
 * 根据用户第一次提问生成会话标题。
 *
 * 例如：
 * - 员工出差住宿标准是什么？
 * - 我们公司名字叫什么？
 */
function buildSessionTitle(question: string): string {
  const cleanQuestion = question.trim()

  if (!cleanQuestion) {
    return '新聊天'
  }

  return cleanQuestion.length > 18 ? `${cleanQuestion.slice(0, 18)}...` : cleanQuestion
}

/**
 * 打字机效果用的延迟函数。
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

/**
 * 聊天主页面。
 *
 * 页面结构：
 * - 左侧：新聊天 + 历史会话列表
 * - 右侧：当前会话内容 + 输入框
 * - 右上角：当前用户信息 + 退出
 */
export function ChatPage({ onLogout }: ChatPageProps) {
  /**
   * 只在首次渲染时读取 localStorage。
   */
  const initialChatStateRef = useRef<ReturnType<typeof loadInitialChatState> | null>(null)

  if (initialChatStateRef.current === null) {
    initialChatStateRef.current = loadInitialChatState()
  }

  /**
   * 所有前端历史会话。
   */
  const [sessions, setSessions] = useState<ChatSession[]>(
    initialChatStateRef.current.sessions,
  )

  /**
   * 当前正在查看的会话 ID。
   */
  const [activeSessionId, setActiveSessionId] = useState<string>(
    initialChatStateRef.current.activeSessionId,
  )

  /**
   * 输入框内容。
   */
  const [question, setQuestion] = useState('')

  /**
   * 请求状态。
   */
  const [asking, setAsking] = useState(false)

  /**
   * 页面错误提示。
   */
  const [errorMessage, setErrorMessage] = useState('')

  /**
   * 消息列表底部锚点，用于自动滚动到底部。
   */
  const messageEndRef = useRef<HTMLDivElement | null>(null)

  /**
   * 当前激活的会话。
   */
  const activeSession = sessions.find((session) => session.id === activeSessionId)

  /**
   * 当前会话的消息列表。
   */
  const messages = activeSession?.messages || createInitialMessages()

  /**
   * 当消息变化时，自动滚动到底部。
   */
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'end',
    })
  }, [messages])

  /**
   * 更新指定会话的消息。
   *
   * 这里不直接 setMessages，因为现在消息属于某个 session。
   */
  function updateSessionMessages(
    sessionId: string,
    nextMessages: ChatMessage[],
    nextTitle?: string,
  ) {
    setSessions((prevSessions) => {
      const nextSessions = prevSessions.map((session) => {
        if (session.id !== sessionId) {
          return session
        }

        return {
          ...session,
          title: nextTitle || session.title,
          messages: nextMessages,
          updatedAt: Date.now(),
        }
      })

      /**
       * 最近更新的会话排在最上面。
       */
      const sortedSessions = [...nextSessions].sort((a, b) => b.updatedAt - a.updatedAt)

      saveChatSessions(sortedSessions)

      return sortedSessions
    })
  }

  /**
   * 新建会话。
   */
  function handleNewChat() {
    const newSession = createNewSession()

    setSessions((prevSessions) => {
      const nextSessions = [newSession, ...prevSessions]
      saveChatSessions(nextSessions)
      return nextSessions
    })

    setActiveSessionId(newSession.id)
    setQuestion('')
    setErrorMessage('')
  }

  /**
   * 切换左侧历史会话。
   */
  function handleSelectSession(sessionId: string) {
    setActiveSessionId(sessionId)
    setQuestion('')
    setErrorMessage('')
  }

  /**
   * 发送问题到后端。
   *
   * 当前阶段：
   * - 前端保存会话历史
   * - 每次请求后端仍然只发送当前 question
   *
   * 后续做真正多轮记忆时，可以在 ChatRequest 中增加 session_id。
   */
  async function handleAsk() {
    const cleanQuestion = question.trim()

    if (!cleanQuestion || asking) {
      return
    }

    const currentSessionId = activeSessionId
    const currentSession = sessions.find((session) => session.id === currentSessionId)
    const currentMessages = currentSession?.messages || createInitialMessages()

    const userMessage: ChatMessage = {
      role: 'user',
      content: cleanQuestion,
    }

    const nextMessages = [...currentMessages, userMessage]

    /**
     * 如果当前会话还是“新聊天”，用用户第一句话生成标题。
     */
    const nextTitle =
      currentSession?.title === '新聊天' ? buildSessionTitle(cleanQuestion) : undefined

    updateSessionMessages(currentSessionId, nextMessages, nextTitle)

    setQuestion('')
    setErrorMessage('')
    setAsking(true)

    try {
      const response = await askQuestion({
        question: cleanQuestion,
        retrieve_top_k: 5,
        rerank_top_k: 5,
        enable_rewrite: true,
        enable_rerank: true,
        enable_compression: true,
      })

      await typeAssistantAnswer(
        currentSessionId,
        nextMessages,
        response.answer,
        response,
      )
    }  catch (error) {
  console.error(error)

  /**
   * Axios 请求失败时，401 表示登录态失效。
   * 这里不继续留在聊天页，而是清理登录信息并回到登录页。
   */
  const status = (error as { response?: { status?: number } }).response?.status

  if (status === 401) {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    localStorage.removeItem('role')

    setErrorMessage('登录已失效，请重新登录')
    onLogout()
    return
  }

  const errorAssistantMessage: ChatMessage = {
    role: 'assistant',
    content: '请求失败，请检查后端服务状态或稍后重试。',
  }

  updateSessionMessages(currentSessionId, [...nextMessages, errorAssistantMessage])
  setErrorMessage('请求失败，请检查后端服务状态或稍后重试')
}finally {
      setAsking(false)
    }
  }

  /**
   * 前端打字机效果。
   *
   * 注意：
   * - 这不是真正的后端流式输出
   * - 后端完整回答返回后，前端逐字显示
   */
  async function typeAssistantAnswer(
    sessionId: string,
    baseMessages: ChatMessage[],
    answer: string,
    response: ChatResponse,
  ) {
    let currentText = ''

    /**
     * 先插入一条空的 assistant 消息。
     */
    updateSessionMessages(sessionId, [
      ...baseMessages,
      {
        role: 'assistant',
        content: '',
        response,
      },
    ])

    for (let index = 0; index < answer.length; index += 1) {
      currentText += answer[index]

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: currentText,
        response,
      }

      updateSessionMessages(sessionId, [...baseMessages, assistantMessage])

      await sleep(12)
    }
  }

  /**
   * 输入框快捷键：
   * - Enter：发送
   * - Shift + Enter：换行
   */
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleAsk()
    }
  }

  /**
   * 点击示例问题时，填入输入框。
   */
  function handleExampleClick(exampleQuestion: string) {
    setQuestion(exampleQuestion)
  }

  return (
    <div className="chat-layout">
      <aside className="chat-sidebar">
        <button
          type="button"
          className="sidebar-new-chat-button"
          onClick={handleNewChat}
        >
          ＋ 新聊天
        </button>

        <div className="sidebar-section-title">最近</div>

        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              type="button"
              className={
                session.id === activeSessionId ? 'session-item active' : 'session-item'
              }
              onClick={() => handleSelectSession(session.id)}
              title={session.title}
            >
              {session.title}
            </button>
          ))}
        </div>
      </aside>

      <div className="chat-shell">
        <header className="chat-header">
          <div className="chat-title-block">
            <h1>企业知识库问答系统</h1>
            <p>基于 Enterprise-RAG 的企业制度与知识问答助手</p>
          </div>

          <div className="header-actions">
            <UserBadge />

            <button type="button" className="logout-button" onClick={onLogout}>
              退出
            </button>
          </div>
        </header>

        <main className="chat-main">
          <div className="message-list">
            {messages.map((message, index) => (
              <MessageBubble key={`${activeSessionId}-${index}`} message={message} />
            ))}

            {asking && messages[messages.length - 1]?.role === 'user' && (
              <MessageBubble
                message={{
                  role: 'assistant',
                  content: '正在查询知识库...',
                }}
              />
            )}

            <div ref={messageEndRef} />
          </div>
        </main>

        <section className="chat-input-area">
          {errorMessage && <div className="error-message">{errorMessage}</div>}

          <div className="example-row">
            <button
              type="button"
              onClick={() => handleExampleClick('员工出差住宿标准是什么？')}
            >
              差旅住宿标准
            </button>

            <button type="button" onClick={() => handleExampleClick('报销多久内提交？')}>
              报销提交时限
            </button>

            <button
              type="button"
              onClick={() => handleExampleClick('出差可以住五星级酒店吗？')}
            >
              五星级酒店
            </button>

            <button
              type="button"
              onClick={() => handleExampleClick('我们公司名字叫什么？')}
            >
              公司名称
            </button>

            <button
              type="button"
              onClick={() => handleExampleClick('公司食堂几点开门？')}
            >
              无答案测试
            </button>
          </div>

          <div className="input-box">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题，按 Enter 发送，Shift + Enter 换行"
              disabled={asking}
            />

            <button
              type="button"
              className="send-button"
              onClick={handleAsk}
              disabled={asking || !question.trim()}
            >
              {asking ? '生成中' : '发送'}
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}

/**
 * 当前登录用户信息。
 *
 * username / role 来自登录成功后写入的 localStorage。
 */
function UserBadge() {
  const username = localStorage.getItem('username') || 'unknown'
  const role = localStorage.getItem('role') || 'employee'

  const roleTextMap: Record<string, string> = {
    admin: '管理员',
    employee: '普通员工',
    hr: 'HR',
    it: 'IT',
    finance: '财务',
    manager: '经理',
  }

  const roleText = roleTextMap[role] || role
  const avatarText = username.slice(0, 1).toUpperCase()

  return (
    <div className="user-badge">
      <div className="user-avatar">{avatarText}</div>

      <div className="user-info">
        <div className="user-name">{username}</div>
        <div className="user-role">{roleText}</div>
      </div>
    </div>
  )
}

type MessageBubbleProps = {
  message: ChatMessage
}

/**
 * 单条消息气泡。
 */
function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={isUser ? 'message-row user' : 'message-row assistant'}>
      <div className="message-avatar">{isUser ? '我' : 'AI'}</div>

      <div className="message-content">
        <div className="message-bubble">
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        </div>

        {message.response && <MessageExtra response={message.response} />}
      </div>
    </div>
  )
}

type MessageExtraProps = {
  response: ChatResponse
}

/**
 * 消息附加信息。
 *
 * 普通用户：
 * - 只显示引用来源
 *
 * admin：
 * - 显示引用来源
 * - 显示调试信息
 *
 * 无答案场景：
 * - sources = []
 * - 不展示引用来源和调试信息
 */
function MessageExtra({ response }: MessageExtraProps) {
  const hasSources = response.sources.length > 0
  const canViewDebug = response.user_role === 'admin'

  if (!hasSources) {
    return null
  }

  return (
    <div className="message-extra">
      <details>
        <summary>引用来源（{response.sources.length}）</summary>
        <SourceList sources={response.sources} />
      </details>

      {canViewDebug && (
        <details>
          <summary>调试信息</summary>

          <div className="debug-grid">
            <DebugBlock title="请求信息">
              <pre>
                {JSON.stringify(
                  {
                    request_id: response.request_id,
                    username: response.username,
                    user_role: response.user_role,
                    cost_time_seconds: response.cost_time_seconds,
                    rewritten_question: response.rewritten_question,
                  },
                  null,
                  2,
                )}
              </pre>
            </DebugBlock>

            <DebugBlock title="Query Route">
              <pre>{JSON.stringify(response.query_route, null, 2)}</pre>
            </DebugBlock>

            <DebugBlock title="Timings">
              <pre>{JSON.stringify(response.timings, null, 2)}</pre>
            </DebugBlock>

            <DebugBlock title="Compression">
              <pre>{JSON.stringify(response.compression, null, 2)}</pre>
            </DebugBlock>
          </div>
        </details>
      )}
    </div>
  )
}

type SourceListProps = {
  sources: SourceItem[]
}

/**
 * 引用来源列表。
 */
function SourceList({ sources }: SourceListProps) {
  return (
    <div className="source-list">
      {sources.map((source, index) => {
        const sourceTitle = source.source || source.file_path || '未知来源'
        const sourceLocation = [source.header_1, source.header_2]
          .filter(Boolean)
          .join(' / ')

        return (
          <div key={source.chunk_id || `${sourceTitle}-${index}`} className="source-card">
            <div className="source-title">{sourceTitle}</div>

            {sourceLocation && <div className="source-meta">{sourceLocation}</div>}

            {source.chunk_id && <div className="source-meta">chunk_id：{source.chunk_id}</div>}

            {source.content_preview && (
              <div className="source-preview">{source.content_preview}</div>
            )}
          </div>
        )
      })}
    </div>
  )
}

type DebugBlockProps = {
  title: string
  children: ReactNode
}

/**
 * admin 调试信息块。
 */
function DebugBlock({ title, children }: DebugBlockProps) {
  return (
    <div className="debug-block">
      <div className="debug-block-title">{title}</div>
      {children}
    </div>
  )
}