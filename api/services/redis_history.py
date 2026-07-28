"""
Redis 对话历史管理。

负责：
1. 保存用户聊天记录
2. 获取最近几轮上下文
3. 限制历史长度，避免 Prompt 无限膨胀
"""

import uuid
import json
import logging
from datetime import datetime

from api.core.redis_client import get_redis_client


logger = logging.getLogger(__name__)


class RedisChatHistory:
    """
    Redis 聊天历史存储。

    一个用户一个 session。

    key:
        chat:history:{user_id}:{session_id}
    """

    def __init__(
        self,
        max_messages: int = 10,
        expire_seconds: int = 86400,
    ):
        self.redis = get_redis_client()

        # 最大保存消息数量
        self.max_messages = max_messages

        # 过期时间
        # 默认一天
        self.expire_seconds = expire_seconds


    def _build_key(
        self,
        user_id: str,
        session_id: str,
    ):
        """
        构造 Redis Key。
        """

        return f"chat:history:{user_id}:{session_id}"


    def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ):
        """
        添加一条聊天记录。
        """

        key = self._build_key(
            user_id,
            session_id,
        )


        message = {
            "role": role,
            "content": content,
            "time": datetime.now().isoformat(),
        }


        # Redis List 追加
        self.redis.rpush(
            key,
            json.dumps(
                message,
                ensure_ascii=False,
            )
        )


        # 保留最近 N 条
        self.redis.ltrim(
            key,
            -self.max_messages,
            -1,
        )


        # 设置过期时间
        self.redis.expire(
            key,
            self.expire_seconds,
        )


    def get_messages(
        self,
        user_id: str,
        session_id: str,
    ):
        """
        获取历史消息。
        """

        key = self._build_key(
            user_id,
            session_id,
        )


        messages = self.redis.lrange(
            key,
            0,
            -1,
        )


        return [
            json.loads(item)
            for item in messages
        ]

    def get_session_messages(
            self,
            user_id: str,
            session_id: str,
    ):
        """
        获取指定 session 的聊天记录。

        用于：
        - 加载历史聊天
        - 判断是否第一次聊天
        """

        return self.get_messages(
            user_id=user_id,
            session_id=session_id,
        )


    def clear(
        self,
        user_id: str,
        session_id: str,
    ):
        """
        清空历史。
        """

        key = self._build_key(
            user_id,
            session_id,
        )

        self.redis.delete(key)

    def create_session(
        self,
        user_id:str,
        title:str="新聊天",
    ):
        """
        创建新的聊天会话。

        返回:
        session_id
        """


        session_id = str(uuid.uuid4())


        session_info = {
            "session_id": session_id,
            "title": title,
            "time": datetime.now().isoformat(),
        }


        key = f"chat:sessions:{user_id}"


        self.redis.rpush(
            key,
            json.dumps(
                session_info,
                ensure_ascii=False,
            )
        )


        return session_id

    def get_sessions(
        self,
        user_id:str,
    ):

        """
        获取左侧历史聊天列表
        """


        key=f"chat:sessions:{user_id}"


        sessions=self.redis.lrange(
            key,
            0,
            -1,
        )


        return [
            json.loads(item)
            for item in sessions
        ]

    def delete_session(
        self,
        user_id:str,
        session_id:str,
    ):

        """
        删除某个聊天
        """


        # 删除消息
        self.clear(
            user_id,
            session_id
        )


        # 删除列表记录

        key=f"chat:sessions:{user_id}"


        sessions=self.get_sessions(user_id)


        for item in sessions:

            if item["session_id"] == session_id:

                self.redis.lrem(
                    key,
                    1,
                    json.dumps(
                        item,
                        ensure_ascii=False
                    )
                )

    def update_session_title(
            self,
            user_id: str,
            session_id: str,
            title: str,
    ):
        """
        更新聊天标题。
        """

        key = f"chat:sessions:{user_id}"

        # 获取所有 session
        sessions = self.redis.lrange(
            key,
            0,
            -1,
        )

        if not sessions:
            return

        # 删除旧列表
        self.redis.delete(key)

        for item in sessions:

            session = json.loads(item)

            if session["session_id"] == session_id:
                session["title"] = title

            # 重新写回 Redis List
            self.redis.rpush(
                key,
                json.dumps(
                    session,
                    ensure_ascii=False,
                )
            )