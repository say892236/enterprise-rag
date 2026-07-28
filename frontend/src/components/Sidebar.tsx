import { useEffect, useState } from "react"

import {
  getSessions,
  createSession,
  deleteSession,
} from "../api/client"




interface SessionItem {

  session_id: string

  title: string

  time: string

}


interface SidebarProps {

  currentSessionId: string | null

  onChangeSession: (
    sessionId: string
  ) => void

  onNewChat: (
    sessionId: string
  ) => void

}



export function Sidebar(
  {
    currentSessionId,
    onChangeSession,
    onNewChat,
  }: SidebarProps
) {


  const [
    sessions,
    setSessions
  ] = useState<SessionItem[]>([])



  /**
   * 页面加载时获取历史聊天
   */
  async function loadSessions(){

    try {

      const result = await getSessions()


      setSessions(result.sessions)


    } catch(error){

      console.error(
        "获取聊天历史失败",
        error
      )

    }

  }



  useEffect(()=>{

    loadSessions()

  },[])



  /**
   * 创建新聊天
   */
  async function handleNewChat(){


    const result = await createSession()


    const sessionId = result.session_id



    // 更新列表
    await loadSessions()


    // 通知 ChatPage 切换
    onNewChat(
      sessionId
    )

  }



  /**
   * 删除聊天
   */
  async function handleDelete(
    sessionId:string
  ){

    const confirmDelete =
      window.confirm(
        "确定删除这个聊天吗？"
      )


    if(!confirmDelete){
      return
    }


    await deleteSession(
      sessionId
    )


    await loadSessions()


  }



  return (

    <aside className="sidebar">


      <button
        className="new-chat-button"
        onClick={handleNewChat}
      >

        + 新聊天

      </button>



      <div className="sidebar-title">

        最近

      </div>



      <div className="session-list">


        {
          sessions.map(
            session=>(


              <div

                key={
                  session.session_id
                }


                className={
                  currentSessionId === session.session_id
                    ?
                    "session-item active"
                    :
                    "session-item"
                }


                onClick={()=>{

                  onChangeSession(
                    session.session_id
                  )

                }}


                onContextMenu={(e)=>{

                  e.preventDefault()

                  handleDelete(
                    session.session_id
                  )

                }}

              >

                {session.title}


              </div>


            )
          )
        }


      </div>


    </aside>

  )

}
