"""
Redis 客户端封装。

当前用途：
1. 统一读取 REDIS_URL 配置。
2. 提供 Redis 客户端给后续业务复用。
3. 提供 ping_redis() 给 /readiness 做 Redis 就绪检查。

后续可以继续基于这个文件实现：
1. 会话历史缓存
2. RAG 问答结果缓存
3. 用户会话状态管理
"""

import logging
import os
from functools import lru_cache

from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_redis_client() -> Redis:
    """
    创建 Redis 客户端。

    说明：
    - Docker Compose 内部访问 Redis：redis://redis:6379/0
    - 本地直连 Redis：redis://127.0.0.1:6379/0

    decode_responses=True：
    - Redis 默认返回 bytes
    - 开启后自动返回 str，后续处理 JSON / 文本更方便
    """

    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    return Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


def ping_redis() -> bool:
    """
    检查 Redis 是否可用。

    返回：
    - True：Redis 正常
    - False：Redis 不可用

    注意：
    这里捕获 RedisError，避免 Redis 临时不可用时直接打崩 /readiness。
    """

    try:
        return bool(get_redis_client().ping())
    except RedisError as e:
        logger.warning("Redis ping failed: %s", e)
        return False