// src/pages/ChatPage.tsx

import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { askQuestion, getHealth, getReadiness } from '../api/client'
import type {
  ChatResponse,
  HealthResponse,
  ReadinessResponse,
  SourceItem,
} from '../types/api'

interface ChatPageProps {
  onLogout: () => void
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
}

/**
 * 极简聊天页面。
 *
 * 目标：
 * 1. 像 ChatGPT 一样输入问题。
 * 2. 回答用 Markdown 渲染。
 * 3. 答案采用前端打字机效果逐字显示。
 * 4. 引用来源和调试信息默认折叠，不干扰主体验。
 */
export function ChatPage({ onLogout }: ChatPageProps) {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        '你好，我是企业知识库问答助手。你可以询问差旅、报销、请假、员工手册和安全合规相关问题。',
    },
  ])

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null)

  const bottomRef = useRef<HTMLDivElement | null>(null)

  /**
   * 页面加载后检查后端状态。
   */
  useEffect(() => {
    refreshStatus()
  }, [])

  /**
   * 消息变化后自动滚动到底部。
   */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function refreshStatus() {
    try {
      const [healthData, readinessData] = await Promise.all([
        getHealth(),
        getReadiness(),
      ])

      setHealth(healthData)
      setReadiness(readinessData)
    } catch (error) {
      console.error(error)
      setHealth(null)
      setReadiness(null)
    }
  }

  /**
   * 发送问题。
   *
   * 注意：
   * 当前后端 /chat 还是一次性返回完整答案。
   * 这里通过前端打字机效果模拟“流式输出”的体验。
   */
  async function handleAsk() {
    const finalQuestion = question.trim()

    if (!finalQuestion || loading) {
      return
    }

    setQuestion('')
    setLoading(true)

    // 先把用户消息加入聊天记录。
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: finalQuestion,
      },
      {
        role: 'assistant',
        content: '',
      },
    ])

    try {
      const result = await askQuestion({
        question: finalQuestion,
        retrieve_top_k: 5,
        rerank_top_k: 5,
        enable_rewrite: true,
        enable_rerank: true,
        enable_compression: true,
        metadata_filter: null,
      })

      await typeAssistantAnswer(result.answer, result)
    } catch (error) {
      console.error(error)

      setMessages((prev) => {
        const copied = [...prev]
        const lastIndex = copied.length - 1

        copied[lastIndex] = {
          role: 'assistant',
          content: '请求失败，请检查登录状态、后端服务或浏览器控制台错误。',
        }

        return copied
      })
    } finally {
      setLoading(false)
    }
  }

  /**
   * 前端打字机效果。
   *
   * 后端返回完整 answer 后，前端逐字追加到最后一条 assistant 消息中。
   */
  async function typeAssistantAnswer(answer: string, response: ChatResponse) {
    let currentText = ''

    for (let index = 0; index < answer.length; index += 1) {
      currentText += answer[index]

      setMessages((prev) => {
        const copied = [...prev]
        const lastIndex = copied.length - 1

        copied[lastIndex] = {
          role: 'assistant',
          content: currentText,
          response,
        }

        return copied
      })

      await sleep(12)
    }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleAsk()
    }
  }

  function handleLogout() {
    onLogout()
  }

  const apiReady = health?.status === 'ok'
  const ragReady = readiness?.status === 'ready'

  return (
    <div className="chat-shell">
      <header className="chat-header">
        <div>
          <h1>企业知识库问答系统</h1>
          <p>面向企业制度文档的 RAG 问答助手</p>
        </div>

        <div className="header-actions">
          <div className="system-badge">
            <span className={apiReady ? 'dot ok' : 'dot bad'} />
            API {apiReady ? '正常' : '异常'}
          </div>

          <div className="system-badge">
            <span className={ragReady ? 'dot ok' : 'dot bad'} />
            RAG {ragReady ? '就绪' : '未就绪'}
          </div>

          <button className="ghost-button" onClick={refreshStatus}>
            刷新
          </button>

          <button className="ghost-button" onClick={handleLogout}>
            退出
          </button>
        </div>
      </header>

      <main className="chat-main">
        <div className="message-list">
          {messages.map((message, index) => (
            <MessageBubble key={index} message={message} />
          ))}

          {loading && (
            <div className="typing-line">
              <span className="typing-dot" />
              正在检索知识库并生成回答...
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="chat-input-area">
        <div className="example-row">
          <button onClick={() => setQuestion('员工出差住宿标准是什么？')}>
            差旅住宿标准
          </button>
          <button onClick={() => setQuestion('报销多久内提交？')}>
            报销提交时限
          </button>
          <button onClick={() => setQuestion('出差可以住五星级酒店吗？')}>
            五星级酒店
          </button>
          <button onClick={() => setQuestion('公司食堂几点开门？')}>
            无答案测试
          </button>
        </div>

        <div className="input-box">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入你的问题，按 Enter 发送，Shift + Enter 换行"
          />

          <button onClick={handleAsk} disabled={loading || !question.trim()}>
            {loading ? '生成中' : '发送'}
          </button>
        </div>
      </footer>
    </div>
  )
}

interface MessageBubbleProps {
  message: ChatMessage
}

/**
 * 单条消息展示。
 */
function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <div className={isUser ? 'message-row user' : 'message-row assistant'}>
      <div className="message-avatar">{isUser ? '我' : 'AI'}</div>

      <div className={isUser ? 'message-bubble user' : 'message-bubble assistant'}>
        <div className="markdown-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>

        {!isUser && message.response && (
          <MessageExtra response={message.response} />
        )}
      </div>
    </div>
  )
}

interface MessageExtraProps {
  response: ChatResponse
}

/**
 * 回答附加信息。
 *
 * 默认折叠，避免页面像调试后台。
 */
function MessageExtra({ response }: MessageExtraProps) {
  const hasSources = response.sources.length > 0

  /**
   * 只有 admin 可以查看调试信息。
   *
   * employee / hr / it / finance 等业务角色：
   * 只展示答案和引用来源，不展示 Query Route、Timings、Compression 等调试信息。
   */
  const canViewDebug = response.user_role === 'admin'

  /**
   * 无答案场景下，后端会返回 sources = []。
   * 此时页面只展示拒答内容，不展示引用来源和调试信息。
   */
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

interface DebugBlockProps {
  title: string
  children: React.ReactNode
}

/**
 * 调试信息块。
 */
function DebugBlock({ title, children }: DebugBlockProps) {
  return (
    <div className="debug-block">
      <h4>{title}</h4>
      {children}
    </div>
  )
}

interface SourceListProps {
  sources: SourceItem[]
}

/**
 * 引用来源列表。
 */
function SourceList({ sources }: SourceListProps) {
  if (sources.length === 0) {
    return <p className="muted">暂无引用来源</p>
  }

  return (
    <div className="source-list">
      {sources.map((source, index) => (
        <div className="source-card" key={`${source.source}-${source.chunk_id}-${index}`}>
          <div className="source-title">
            #{index + 1} {source.source || 'unknown source'}
          </div>

          <div className="source-subtitle">
            {source.header_2 || source.header_1 || '未命中标题'}
          </div>

          <p>
            {source.content_preview ||
              source.content ||
              source.page_content ||
              '暂无内容预览'}
          </p>
        </div>
      ))}
    </div>
  )
}

function sleep(ms: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}