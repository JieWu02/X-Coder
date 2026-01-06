from app.libs.redis_queue import RedisQueue
import app.config as app_config


def connect_queue(is_async: bool = False) -> RedisQueue:
    return RedisQueue(
        redis_uri=app_config.REDIS_URI,
        queue_name=app_config.REDIS_WORK_QUEUE_NAME,
        socket_timeout=app_config.REDIS_SOCKET_TIMEOUT,
        is_async=is_async,
    )
