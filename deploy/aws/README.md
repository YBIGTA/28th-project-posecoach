# AWS EC2 Docker Deployment (Demo)

This setup runs:
- `api` (FastAPI + pose analysis) on internal port `8000`
- `web` (Nginx + built frontend) on public port `80`

The frontend proxies:
- `/api/*` -> FastAPI
- `/static/frames/*` -> FastAPI static frames

## 1. EC2 prerequisites

- Ubuntu 22.04 (recommended)
- Security Group inbound:
  - `22` from your IP
  - `80` from `0.0.0.0/0`

## 2. Install Docker + Compose plugin

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $USER
newgrp docker
```

## 3. Clone and configure

```bash
git clone <YOUR_REPO_URL>
cd 28th-project-posecoach
```

Create `.env` in repo root:

```bash
cat > .env << 'EOF'
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
EOF
```

## 4. Run

```bash
docker compose build
docker compose up -d
docker compose ps
```

Open:
- `http://<EC2_PUBLIC_IP>`

## 5. Logs / restart / stop

```bash
docker compose logs -f api
docker compose logs -f web
docker compose restart
docker compose down
```

## 6. Persistence

`docker-compose.yml` mounts `./data` into `/app/data`, so these persist:
- uploaded videos
- extracted frame images
- SQLite DB (`data/posecoach.db`)

## Notes

- First API build may take time (OpenCV + ML deps).
- If you later add HTTPS, place ALB/CloudFront or Certbot-enabled Nginx in front.
