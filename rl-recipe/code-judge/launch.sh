# set up Docker's apt repository

# Add Docker's official GPG key:
# sudo apt-get update
# sudo apt-get install ca-certificates curl
# sudo install -m 0755 -d /etc/apt/keyrings
# sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
# sudo chmod a+r /etc/apt/keyrings/docker.asc

# # Add the repository to Apt sources:
# echo \
#   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
#   $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
#   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# sudo apt-get update


# # Install Docker Engine:
# sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# # Verify
# sudo docker run hello-world

echo "*********** Run the project without docker ************"
# sudo apt-get update
# sudo apt-get install -y redis
# sudo apt install -y httpie
wget https://download.redis.io/redis-stable.tar.gz
tar -xzvf redis-stable.tar.gz
cd redis-stable
make
cd src && make install
pip install  httpie
cd ../../


# pip install -r requirements.txt

echo "***** RUN redis-server *****"

redis-server --daemonize yes
redis-cli ping  # 如果返回 PONG 说明 Redis 运行正常

echo "***** Checking Redis Connection *****"
# python -c "import redis; r = redis.from_url('rediss+cluster://:YOUR_REDIS_ACCESS_KEY@aicoder-judge.eastus2.redis.azure.net:10000'); print(r.ping())"
pip install redis 
pip install fastapi[standard] 
pip install uvicorn 
pip install psutil 
# pip install torch 
# pip install "numpy<2" --force-reinstall 

lscpu

# export PATH=/home/aiscuser/.local/bin:$PATH
# fastapi --help

# REDIS_URI=rediss+cluster://:YOUR_REDIS_ACCESS_KEY@aicoder-judge.eastus2.redis.azure.net:10000 uvicorn app.main:app --workers 4 --limit-max-requests 1000 --port 8005
# python compute_matrix.py &
# REDIS_URI=rediss+cluster://:YOUR_REDIS_ACCESS_KEY@aicoder-judge.eastus2.redis.azure.net:10000 python run_workers.py &
# echo "**** debug_api ****"
# REDIS_URI=rediss+cluster://:YOUR_REDIS_ACCESS_KEY@aicoder-judge.eastus2.redis.azure.net:10000 python debug_api.py


### local start

REDIS_URI=redis://localhost:6379 uvicorn app.main:app --workers 4 --limit-max-requests 1000 --port 8005 &
sleep 8s

REDIS_URI=redis://localhost:6379 python run_workers.py &
sleep 5s

echo "redis start !!!"

# REDIS_URI=redis://localhost:6379 uvicorn app.main:app --host 0.0.0.0 --port 8005 --workers 4 &
# sleep 10
# REDIS_URI=redis://localhost:6379 python run_workers.py &
# sleep 5
# REDIS_URI=rediss+cluster://:YOUR_REDIS_ACCESS_KEY@aicoder-judge.eastus2.redis.azure.net:10000 uvicorn app.main:app --host 0.0.0.0 --port 8005 --workers 4 --limit-max-requests 1000 > redis_output.log 2>&1 &

# REDIS_URI=rediss+cluster://:YOUR_REDIS_ACCESS_KEY@aicoder-codejudge-sa.eastus2.redis.azure.net:10000 uvicorn app.main:app --host 0.0.0.0 --port 8005 --workers 4 --limit-max-requests 1000
# REDIS_URI=rediss+cluster://:YOUR_REDIS_ACCESS_KEY@aicoder-codejudge-sa.eastus2.redis.azure.net:10000 python run_workers.py

# sleep 5
### local stop
# pkill -f run_workers.py

### 
# ps aux | grep main.py
# ps aux | grep run_workers.py

# kill $(lsof -t -i :8005)
# kill $(lsof -t -i :6379)

# wait
