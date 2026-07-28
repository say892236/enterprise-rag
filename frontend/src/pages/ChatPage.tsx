import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import "../assets/markdown.css";

import {
  askQuestionStream,
  createSession,
  deleteSession,
  getSessionMessages,
  getSessions,
} from '../api/client'

import type {
  ChatResponse,
  ChatStreamStage,
  RedisChatMessage,
  RedisSession,
  SourceItem,
} from '../types/api'

type ChatPageProps = {
  onLogout: () => void
}

type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
  loading?: boolean
}



function toChatMessages(historyMessages: RedisChatMessage[]): ChatMessage[] {
  return historyMessages.map(({ role, content }) => ({ role, content }))
}

export function ChatPage({ onLogout }: ChatPageProps) {
  const [sessions, setSessions] = useState<RedisSession[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
const [asking, setAsking] = useState(false)

// 当前后端真实执行阶段：检索或生成
const [streamStage, setStreamStage] = useState<ChatStreamStage | null>(null)

const [errorMessage, setErrorMessage] = useState('')


  const showExampleQuestions =
  messages.length === 0
  && sessionId === null
  && !asking


  const messageEndRef = useRef<HTMLDivElement | null>(null)
  const currentSessionIdRef = useRef<string | null>(null)
  const initialisedRef = useRef(false)
  const sessionLoadVersionRef = useRef(0)


  useEffect(() => {
    messageEndRef.current?.scrollIntoView({
      // Keep pace with the typewriter updates; smooth scrolling is used once
      // the response is complete (and for session/history changes).
      behavior: asking ? 'auto' : 'smooth',
      block: 'end',
    })
  }, [asking, messages, sessionId])



const handleRequestError = useCallback(
  (
    error: unknown,
    fallbackMessage: string,
  ): void => {
    console.error(error)

    // 同时兼容 Axios 错误和 fetch 流式请求错误
    const status =
      (
        error as {
          response?: {
            status?: number
          }
          status?: number
        }
      ).response?.status
      ?? (
        error as {
          status?: number
        }
      ).status

    if (status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('username')
      localStorage.removeItem('role')

      setErrorMessage('登录已失效，请重新登录')
      onLogout()
      return
    }

    // SSE 抛出的错误通常包含后端的具体错误消息
    const errorText =
      error instanceof Error
        ? error.message
        : ''

    setErrorMessage(
      errorText && !errorText.startsWith('Request failed')
        ? errorText
        : fallbackMessage,
    )
  },
  [onLogout],
)

  const loadSessions = useCallback(async (): Promise<void> => {
    const result = await getSessions()

    if (!result.success) {
      throw new Error('加载聊天历史失败')
    }

    setSessions(result.sessions)
  }, [])

  useEffect(() => {
    if (initialisedRef.current) {
      return
    }

    initialisedRef.current = true

    void loadSessions().catch((error: unknown) => {
      handleRequestError(error, '加载聊天历史失败，请稍后重试')
    })
  }, [handleRequestError, loadSessions])

  function handleNewChat(): void {
    sessionLoadVersionRef.current += 1
    currentSessionIdRef.current = null
    setSessionId(null)
    setMessages([])
    setQuestion('')
    setErrorMessage('')
  }

  async function handleSelectSession(nextSessionId: string): Promise<void> {
    if (asking || nextSessionId === sessionId) {
      return
    }

    const loadVersion = sessionLoadVersionRef.current + 1
    sessionLoadVersionRef.current = loadVersion
    currentSessionIdRef.current = nextSessionId
    setSessionId(nextSessionId)
    setMessages([])
    setQuestion('')
    setErrorMessage('')

    try {
      const result = await getSessionMessages(nextSessionId)

      if (!result.success) {
        throw new Error('加载聊天记录失败')
      }

      if (
        currentSessionIdRef.current === nextSessionId &&
        sessionLoadVersionRef.current === loadVersion
      ) {
        setMessages(toChatMessages(result.messages))
      }
    } catch (error) {
      if (currentSessionIdRef.current === nextSessionId) {
        handleRequestError(error, '加载聊天记录失败，请稍后重试')
      }
    }
  }

  async function handleDeleteSession(targetSessionId: string): Promise<void> {
    if (asking || !window.confirm('确定删除这个聊天吗？')) {
      return
    }

    try {
      const result = await deleteSession(targetSessionId)

      if (!result.success) {
        throw new Error('删除失败')
      }

      if (targetSessionId === currentSessionIdRef.current) {
        sessionLoadVersionRef.current += 1
        currentSessionIdRef.current = null
        setSessionId(null)
        setMessages([])
        setQuestion('')
        setErrorMessage('')
      }

      await loadSessions()
    } catch (error) {
      handleRequestError(error, '删除失败')
    }
  }

  /**
 * 更新消息列表最后一条 AI 消息。
 *
 * 本次流式回答期间，最后一条消息始终是正在生成的 AI 消息。
 * status 和 token 事件都会通过这个方法更新它。
 */
function updateLastAssistantMessage(
  targetSessionId: string,
  updater: (message: ChatMessage) => ChatMessage,
): void {
  // 用户已经切换会话时，忽略旧请求返回的数据
  if (currentSessionIdRef.current !== targetSessionId) {
    return
  }

  setMessages((currentMessages) => {
    const lastIndex = currentMessages.length - 1
    const lastMessage = currentMessages[lastIndex]

    if (!lastMessage || lastMessage.role !== 'assistant') {
      return currentMessages
    }

    const nextMessages = [...currentMessages]
    nextMessages[lastIndex] = updater(lastMessage)

    return nextMessages
  })
}

  async function handleAsk(): Promise<void> {
  const cleanQuestion = question.trim()

  if (!cleanQuestion || asking) {
    return
  }

  let targetSessionId = sessionId

  setQuestion('')
  setErrorMessage('')
  setStreamStage('retrieving')
  setAsking(true)

  try {
    // 当前没有会话时，先创建新的 Redis 会话
    if (!targetSessionId) {
      const result = await createSession()

      if (!result.success || !result.session_id) {
        throw new Error('创建聊天失败')
      }

      targetSessionId = result.session_id
      currentSessionIdRef.current = targetSessionId
      setSessionId(targetSessionId)
    }

    if (!targetSessionId) {
      throw new Error('未能获取聊天会话 ID')
    }
    // 加在这里：经过上面的判断后，它一定是 string
const activeSessionId = targetSessionId

    /*
     * 立即插入两条消息：
     * 1. 用户问题
     * 2. AI 占位消息
     *
     * 后续 status 和 token 事件都会持续更新这条 AI 消息。
     */
    setMessages([
      ...messages,
      {
        role: 'user',
        content: cleanQuestion,
      },
      {
        role: 'assistant',
        content: '正在检索知识库...',
        loading: true,
      },
    ])

    const response = await askQuestionStream(
      {
        question: cleanQuestion,
        session_id: targetSessionId,
        retrieve_top_k: 5,
        rerank_top_k: 5,
        enable_rewrite: true,
        enable_rerank: true,
        enable_compression: true,
        metadata_filter: null,
      },



      {

        /**
         * 后端状态事件：
         * retrieving -> generating
         */
        onStatus: (data) => {
          setStreamStage(data.stage)

          updateLastAssistantMessage(
    activeSessionId,
    (message) => {
      if (!message.loading) {
        return message
      }

      return {
        ...message,
        content: data.message,
        loading: true,
      }
    },
  )
},

        /**
         * 后端大模型增量文本。
         */
        onToken: (data) => {
          updateLastAssistantMessage(
            activeSessionId,
            (message) => {
              /*
               * 第一个 token 到达时：
               * 清除“正在生成结果...”状态文字。
               *
               * 后续 token 到达时：
               * 继续追加到已有回答。
               */
              const currentContent = message.loading
                ? ''
                : message.content

              return {
                ...message,
                content: currentContent + data.content,
                loading: false,
              }
            },
          )
        },

        /**
         * SSE 业务错误。
         *
         * askQuestionStream 随后还会抛出异常，
         * 最终由 catch 统一处理。
         */
        onError: (data) => {
          setErrorMessage(data.message)
        },
      },
    )

    /*
     * done 事件已经到达。
     *
     * 使用后端返回的完整答案校正最终文本，
     * 并挂载 sources、耗时和管理员调试信息。
     */
    updateLastAssistantMessage(
      targetSessionId,
      (message) => ({
        ...message,
        content: response.answer,
        response,
        loading: false,
      }),
    )

    // 第一轮问答完成后，后端可能已经更新会话标题
    await loadSessions()
  } catch (error) {
    if (targetSessionId) {
      updateLastAssistantMessage(
        targetSessionId,
        (message) => {
          /*
           * 尚未收到任何 token：
           * 将状态占位文字替换为失败提示。
           *
           * 已收到部分 token：
           * 保留已生成内容，并标记生成中断。
           */
          if (message.loading) {
            return {
              ...message,
              content: '回答生成失败，请稍后重试。',
              loading: false,
            }
          }

          return {
            ...message,
            content: `${message.content}\n\n> 回答生成中断，请稍后重试。`,
            loading: false,
          }
        },
      )
    }

    handleRequestError(
      error,
      '请求失败，请检查后端服务状态或稍后重试',
    )
  } finally {
    setAsking(false)
    setStreamStage(null)
  }
}


  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void handleAsk()
    }
  }

  function handleExampleClick(exampleQuestion: string): void {
    setQuestion(exampleQuestion)
  }

  return (
    <div className="chat-layout">
      <aside className="chat-sidebar">
        <button
          type="button"
          className="sidebar-new-chat-button"
          onClick={() => void handleNewChat()}
          disabled={asking}
        >
          + 新聊天
        </button>

        <div className="sidebar-section-title">最近</div>

        <div className="session-list">
          {sessions.length === 0 ? (

<div className="empty-history">
    暂无历史消息
</div>

) : (
            sessions.map((session) => (
            <button
              key={session.session_id}
              type="button"
              className={session.session_id === sessionId ? 'session-item active' : 'session-item'}
              onClick={() => void handleSelectSession(session.session_id)}
              onContextMenu={(event) => {
                event.preventDefault()
                void handleDeleteSession(session.session_id)
              }}
              title={session.title}
            >
              {session.title}
            </button>
            )))
          }
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
           {
          messages.length === 0 ? (

           <div className="empty-chat">
               👋 你好，我是企业知识库问答助手
你可以咨询：公司概览、差旅、报销、请假、员工手册、IT安全权限、合规相关问题
                 </div>
              ) : (
                  messages.map((message,index)=>(
             <MessageBubble
             key={`${index}`}
               message={message}
              />
))
)
}


            <div ref={messageEndRef} className="messages-end-anchor" />
          </div>
        </main>

        <section className="chat-input-area">
          {errorMessage && <div className="error-message">{errorMessage}</div>}

          {showExampleQuestions && (
  <div className="example-row">
    <button
      type="button"
      onClick={() => handleExampleClick('员工出差住宿标准是什么？')}
    >
      差旅住宿标准
    </button>

    <button
      type="button"
      onClick={() => handleExampleClick('报销需要在多久内提交？')}
    >
      报销提交时限
    </button>

    <button
      type="button"
      onClick={() => handleExampleClick('员工请假需要经过哪些流程？')}
    >
      员工请假流程
    </button>
  </div>
)}

          <div className="input-box">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入问题 | Enter发送，Shift+Enter换行"
              disabled={asking}
            />

            <button
              type="button"
              className="send-button"
              onClick={() => void handleAsk()}
              disabled={asking || !question.trim()}
            >
              {!asking
             ? '发送'
           : streamStage === 'retrieving'
          ? '检索中...'
            : '生成中...'}
            </button>
          </div>
        </section>
      </div>
    </div>
  )
}

function UserBadge() {
  const username = localStorage.getItem('username') || 'unknown'
  const role = localStorage.getItem('role') || 'employee'

  const roleTextMap: Record<string, string> = {
  admin: '系统管理员',
  employee: '普通员工',
  hr: '人力资源',
  finance: '财务',
  engineering: '研发',
  administration: '行政',
  legal: '法务',
  management: '管理层',

  // 兼容之前可能签发的旧 Token
  it: 'IT',
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

function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={isUser ? 'message-row user' : 'message-row assistant'}>
      <div className="message-avatar">{isUser ? '我' : 'AI'}</div>

      <div className="message-content">
        <div className="message-bubble">
          <div className="markdown-body">
            {message.loading ? (
   <div className="loading-text">
      {message.content}
   </div>
 )
 :
 (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
 )}
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

function MessageExtra({ response }: MessageExtraProps) {
  const [showSources, setShowSources] = useState(false)

  // 管理员最多查看 5 条，普通用户最多查看 3 条
  const sourceLimit = response.user_role === 'admin' ? 5 : 3

  const visibleSources = response.sources.slice(0, sourceLimit)
  const hasSources = visibleSources.length > 0
  const canViewDebug = response.user_role === 'admin'

  if (!hasSources) {
    return null
  }

  return (
  <div className="message-extra">

   <div
  className="source-title-bar"
  onClick={() => setShowSources(!showSources)}
>

  📚 参考依据（{visibleSources.length}）

  <span className="source-toggle">
    {showSources ? '收起' : '展开'}
  </span>

</div>


{showSources && (
  <SourceList sources={visibleSources} />
)}
      {canViewDebug && (
         <>
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
      <details>
  <summary>
    Reranker排序
  </summary>
  <RerankDebug sources={response.sources}/>
</details>
      </>
      )}
    </div>
  )
}

type SourceListProps = {
  sources: SourceItem[]
}

/**
 * 格式化参考依据名称。
 *
 * 作用：
 * 1. 去掉文件路径。
 * 2. 去掉 Markdown 的 .md 后缀。
 * 3. 清理首尾空白。
 */
function formatSourceName(sourceName?: string): string {
  if (!sourceName) {
    return '未知来源'
  }

  // 同时兼容 Windows 和 Linux 路径
  const fileName = sourceName
    .split(/[\\/]/)
    .pop()
    ?.trim()

  return (fileName || '未知来源')
    .replace(/\.md$/i, '')
    .trim()
}


/**
 * 清理引用片段中的 Markdown 标记。
 *
 * 这里只用于参考依据预览，不修改真正交给模型的知识库内容。
 */
function cleanSourcePreview(content?: string): string {
  if (!content) {
    return ''
  }

  return content
    // 删除出现在行首或空白后的 Markdown 标题符号
    .replace(/(^|\s)#{1,6}\s*/g, '$1')

    // 删除加粗、斜体和行内代码符号
    .replace(/[*_`]/g, '')

    // 删除引用符号
    .replace(/(^|\s)>\s*/g, '$1')

    // 表格竖线转换为空格
    .replace(/\|/g, ' ')

    // 合并连续空白
    .replace(/\s+/g, ' ')

    .trim()
}

function SourceList({ sources }: SourceListProps) {
  return (
    <div className="source-list">
      {sources.map((source, index) => {
        /*
         * 展示标题优先级：
         * 1. Markdown 一级标题 header_1
         * 2. 原始文件名 source
         * 3. 文件路径 file_path
         *
         * 无论使用哪个字段，都统一去掉 .md 后缀。
         */
        const sourceTitle = formatSourceName(
          source.header_1
          || source.source
          || source.file_path,
        )

        /*
         * 一级标题已经作为文档名称展示，
         * 所以这里不再重复显示 header_1，
         * 只显示具体章节 header_2。
         */
        const sourceLocation = cleanSourcePreview(
          source.header_2,
        )

        const preview = cleanSourcePreview(
          source.content_preview
          || source.content
          || source.page_content,
        )

        return (
          <div
            key={source.chunk_id || index}
            className="source-card"
          >
            <div className="source-file">
              📄 {sourceTitle}
            </div>

            {sourceLocation && (
              <div className="source-location">
                📌 {sourceLocation}
              </div>
            )}

            {preview && (
              <div className="source-preview">
                {preview}
              </div>
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

function DebugBlock({ title, children }: DebugBlockProps) {
  return (
    <div className="debug-block">
      <div className="debug-block-title">{title}</div>
      {children}
    </div>
  )
}


type RerankDebugProps = {
  sources: SourceItem[]
}


function RerankDebug({ sources }: RerankDebugProps) {
  return (
    <div className="rerank-debug">

      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>文档</th>
            <th>章节</th>
            <th>Score</th>
          </tr>
        </thead>

        <tbody>
          {sources.map((source, index) => (
            <tr key={`${source.chunk_id}-${index}`}>

              <td>
                {index + 1}
              </td>

              <td>
                {source.source}
              </td>

              <td>
                {source.header_2 || '-'}
              </td>

              <td>
                {
                  source.rerank_score !== undefined
                    ? source.rerank_score.toFixed(4)
                    : '-'
                }
              </td>

            </tr>
          ))}
        </tbody>

      </table>

    </div>
  )
}
