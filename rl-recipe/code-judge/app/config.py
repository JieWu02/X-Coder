import os
from app.version import __version__ as version


env = os.environ.get

ERROR_CASE_SAVE_PATH = env('ERROR_CASE_SAVE_PATH', '')  # default empty, which means not save error case

MAX_EXECUTION_TIME = int(env('MAX_EXECUTION_TIME', 10))  # default 10 seconds
MAX_STDOUT_ERROR_LENGTH = int(env('MAX_STDOUT_ERROR_LENGTH', 1000))
# default 15 seconds
# additional 5 seconds for communication between judge server and judge worker
MAX_QUEUE_WAIT_TIME = int(env('MAX_QUEUE_WAIT_TIME', MAX_EXECUTION_TIME + 5))
LONG_BATCH_MAX_QUEUE_WAIT_TIME = int(env('LONG_BATCH_MAX_QUEUE_WAIT_TIME', 60*60))  # default 1 hour
MAX_QUEUE_WORK_LIFE_TIME = int(env('MAX_QUEUE_WORK_LIFE_TIME', 4))  # default 4s
MAX_MEMORY = int(env('MAX_MEMORY', 256))  # default 256 MB
MAX_WORKERS = int(env('MAX_WORKERS', os.cpu_count())) or os.cpu_count()  # default os.cpu_count()

RUN_WORKERS = int(env('RUN_WORKERS', 0))  # default 0, which means run workers in a separate process

MAX_BATCH_CHUNK_SIZE = int(env('MAX_BATCH_CHUNK_SIZE', 2))  # 0 means no limit
MAX_LONG_BATCH_CHUNK_SIZE = int(env('MAX_LONG_BATCH_CHUNK_SIZE', 100))

PYTHON_EXECUTOR_PATH = env('PYTHON_EXECUTOR_PATH', 'python3')
CPP_COMPILER_PATH = env('CPP_COMPILER_PATH', 'g++')

# TODO: support fakeredis for testing.
REDIS_URI = env('REDIS_URI', '')
if not REDIS_URI:
    raise ValueError('REDIS_URI is not set')
REDIS_KEY_PREFIX = env('REDIS_KEY_PREFIX', 'js')
REDIS_RESULT_PREFIX = env('REDIS_RESULT_QUEUE_PREFIX', f'{REDIS_KEY_PREFIX}:{version}:result-queue:')
REDIS_RESULT_EXPIRE = int(env('REDIS_RESULT_EXPIRE', 60))  # default 1 minute
REDIS_RESULT_LONG_BATCH_EXPIRE = int(env('REDIS_RESULT_LONG_BATCH_EXPIRE', LONG_BATCH_MAX_QUEUE_WAIT_TIME))  # default 1 hour
REDIS_WORK_QUEUE_NAME = env('WORK_QUEUE_NAME', f'{REDIS_KEY_PREFIX}:{version}:work-queue')

REDIS_WORK_QUEUE_BLOCK_TIMEOUT = int(env('REDIS_WORK_QUEUE_BLOCK_TIMEOUT', 30))  # default 30 seconds
REDIS_WORKER_ID_PREFIX = env('REDIS_WORKER_ID_PREFIX', f'{REDIS_KEY_PREFIX}:{version}:work-ids:')
REDIS_WORKER_REGISTER_EXPIRE = int(env('REDIS_WORKER_REGISTER_TIMEOUT', 120))  # default 2 minute
if REDIS_WORKER_REGISTER_EXPIRE < REDIS_WORK_QUEUE_BLOCK_TIMEOUT:
    raise ValueError('REDIS_WORKER_REGISTER_EXPIRE must be bigger than REDIS_WORK_QUEUE_BLOCK_TIMEOUT')

# default 15 seconds
# additional 5 seconds for communication between judge server and judge worker
REDIS_SOCKET_TIMEOUT = int(env('REDIS_SOCKET_TIMEOUT', 60)) # default 1 minute
