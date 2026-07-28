"""
Redis 聊天历史管理。

负责：
1. 保存聊天记录
2. 获取历史消息
3. 删除聊天记录

Redis 数据结构：

key:

chat:{conversation_id}


value:

[
 {
   "role":"user",
   "content":"上海住宿标准"
 },
 {
   "role":"assistant",
   "content":"普通员工600元"
 }
]

"""


import json
import os
from typing import List, Dict

import redis



class RedisChatHistory:

    def __init__(self):

        # Docker环境:
        # redis://redis:6379/0
        #
        # 本地运行:
        # redis://localhost:6379/0

        redis_url = os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0"
        )


        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True
        )


    def get_history(
        self,
        conversation_id:str
    ) -> List[Dict]:
        """
        获取历史消息
        """


        key = f"chat:{conversation_id}"


        data = self.client.get(key)


        if not data:
            return []


        return json.loads(data)



    def save_message(
        self,
        conversation_id:str,
        role:str,
        content:str
    ):
        """
        保存单条消息
        """


        key = f"chat:{conversation_id}"


        history = self.get_history(
            conversation_id
        )


        history.append(
            {
                "role":role,
                "content":content
            }
        )


        self.client.set(
            key,
            json.dumps(
                history,
                ensure_ascii=False
            ),

            # 聊天过期时间
            # 7天自动清理
            ex=60*60*24*7
        )



    def delete_chat(
        self,
        conversation_id:str
    ):
        """
        删除聊天
        """

        key=f"chat:{conversation_id}"

        self.client.delete(key)