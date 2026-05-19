<div align="center">
  
# easyResearchAssistant

**High-Availability AI Inference Gateway**

*A production-ready, fault-tolerant AI assistant with distributed inference and local fallback capabilities*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Overview

**easyResearchAssistant** is a lightweight, high-availability AI inference gateway designed for families who want reliable access to AI assistants without single points of failure. The system intelligently routes requests across multiple distributed inference providers and automatically falls back to local GPU inference when cloud services are unavailable.

### Key Features

| Feature | Description |
|---------|-------------|
| **Distributed Inference** | Load balancing across multiple cloud inference nodes |
| **Automatic Failover** | Intelligent retry with exponential backoff on errors |
| **Local Fallback** | Seamless switch to local Ollama when cloud is exhausted |
| **Research Mode (RAG)** | Real-time web search with DuckDuckGo for up-to-date information |
| **Monitoring Dashboard** | Real-time observability with node status, metrics, and live logs |
| **Streaming Responses** | Real-time chat experience with SSE |
| **Access Control** | Token-based authentication |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client Applications                          │
│                  (Streamlit UI / API Consumers)                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       API Gateway (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   Auth      │  │  Request Router │  │  Streaming Handler      │  │
│  │   Layer     │  │  & Retry Logic  │  │  (SSE)                  │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                           │                                         │
│                           ▼                                         │
│              ┌─────────────────────────┐                            │
│              │  RAG Search Tool        │ ◄── Research Mode          │
│              │  (DuckDuckGo)           │                            │
│              └─────────────────────────┘                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Provider Manager                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Selection Strategies: Round Robin │ Random │ Least Used     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Health Monitoring │ Cooldown Management │ Auto-Recovery     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Inference      │  │  Inference      │  │  Inference      │
│  Node Alpha     │  │  Node Beta      │  │  Node Gamma     │
│  (Cloudflare)   │  │  (Cloudflare)   │  │  (Cloudflare)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
           │
           │ (All cloud nodes exhausted)
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Local Fallback (Ollama)                          │
│                 RTX 3050 • Llama 3.1 • Low-VRAM Mode                │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Principles

- **Redundancy**: Multiple inference providers ensure no single point of failure
- **Graceful Degradation**: Automatic fallback to local inference preserves availability
- **User Privacy**: Local processing option keeps sensitive queries on-premises
- **Resource Awareness**: Lightweight local inference respects GPU memory constraints

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (optional, for local fallback)
- Cloudflare Workers AI API access

### Installation

```bash
# Clone the repository
git clone https://github.com/hzjanuary/easyResearchAssistant.git
cd easyResearchAssistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env with your credentials
```

### Configuration

1. **Generate Access Token**:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Configure `.env`**:
   ```env
   # Security
   ACCESS_TOKEN=your_generated_token_here
   ADMIN_PASSWORD=your_admin_password_here
   
   # Add your Cloudflare credentials
   CLOUDFLARE_ACCOUNT_1_ID=your_account_id
   CLOUDFLARE_ACCOUNT_1_TOKEN=your_api_token
   CLOUDFLARE_ACCOUNT_1_NAME=Provider-Alpha
   CLOUDFLARE_MODEL=@cf/meta/llama-3.1-8b-instruct
   
   # Enable local fallback (optional)
   OLLAMA_ENABLED=true
   OLLAMA_MODEL=llama3.1
   
   # For Docker: automatically configured via docker-compose.yml
   # BACKEND_URL=http://backend:8000
   # LOG_FILE=/app/logs/system.log
   ```

3. **Start Local Fallback** (optional):
   ```bash
   ollama pull llama3.1
   ollama serve
   ```

### Running the Applications

The system consists of three components:

| Component | Purpose | Default Port |
|-----------|---------|--------------|
| API Gateway | FastAPI backend | 8000 |
| Chat UI | User-facing chat interface | 8501 |
| Admin Dashboard | System monitoring & health | 8502 |

#### Option 1: Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

Access:
- Chat UI: `http://localhost:8501`
- Admin Dashboard: `http://localhost:8502`
- API Gateway: `http://localhost:8000`

> **Note for Windows/Mac**: Ollama running on host is accessible via `host.docker.internal:11434`

#### Docker Volume for Logs

The `docker-compose.yml` creates a shared volume `app_logs` for centralized logging:
- Backend writes logs to `/app/logs/system.log`
- Admin UI reads logs from the same volume (read-only)

To view logs from Docker:
```bash
docker-compose logs -f backend
```

#### Option 2: Local Development

**Start the API Gateway:**
```bash
python api_gateway.py
```

**Start the Chat UI (Port 8501):**
```bash
streamlit run streamlit_app.py
```

**Start the Admin Dashboard (Port 8502):**
```bash
streamlit run admin_app.py --server.port 8502
```

**Run Both Apps Simultaneously (PowerShell):**
```powershell
Start-Process -NoNewWindow powershell -ArgumentList "streamlit run streamlit_app.py"
Start-Process -NoNewWindow powershell -ArgumentList "streamlit run admin_app.py --server.port 8502"
```

**Run Both Apps Simultaneously (Bash):**
```bash
streamlit run streamlit_app.py &
streamlit run admin_app.py --server.port 8502 &
```

Access:
- Chat UI: `http://localhost:8501`
- Admin Dashboard: `http://localhost:8502`

---

## API Reference

### Authentication

All endpoints (except `/health`) require Bearer token authentication:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/v1/status
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info (public) |
| `GET` | `/health` | Health check with provider status (public) |
| `POST` | `/v1/inference` | Main inference endpoint |
| `GET` | `/v1/status` | Detailed gateway status |
| `POST` | `/v1/providers/strategy/{strategy}` | Change selection strategy |
| `POST` | `/v1/providers/reset` | Reset all providers |
| `GET` | `/v1/monitoring/stats` | Comprehensive monitoring statistics |
| `GET` | `/v1/monitoring/logs` | Recent log entries |
| `GET` | `/v1/monitoring/health` | Lightweight health for polling (public) |

### Inference Request

```bash
curl -X POST http://localhost:8000/v1/inference \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum entanglement",
    "research_mode": true,
    "stream": true,
    "max_tokens": 2048,
    "temperature": 0.7
  }'
```

### Response Formats

**Streaming (SSE)**:
```
data: {"response": "Quantum "}
data: {"response": "entanglement "}
data: {"response": "is..."}
data: [DONE]
```

**Non-Streaming**:
```json
{
  "response": "Quantum entanglement is...",
  "provider": "Provider-Alpha"
}
```

---

## Research Mode (RAG)

Research Mode enables **Retrieval-Augmented Generation (RAG)** - the system performs real-time web searches before generating responses, ensuring up-to-date information.

### How It Works

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────┐
│ User Query  │───▶│ DuckDuckGo Search│───▶│ Augmented Prompt│───▶│ LLM Response│
└─────────────┘    └──────────────────┘    └─────────────────┘    └─────────────┘
                          │                        │
                          ▼                        ▼
                   [Search Results]    [System Prompt + Results + Query]
```

### Features

- **Real-Time Web Search**: Queries DuckDuckGo for current information
- **Source Citations**: Results include titles, snippets, and URLs
- **Intelligent Augmentation**: Search results are injected into the system prompt
- **Graceful Fallback**: If search fails, uses base LLM knowledge

### Example

Query: *"Who is the current President of Vietnam?"*

1. System searches DuckDuckGo for latest information
2. Search results are formatted and added to the prompt
3. LLM generates response using real-time data (2026)

### Enable via API:
```json
{"prompt": "Who won the 2026 World Cup?", "research_mode": true}
```

### Enable via UI:
Toggle "Enable Research Mode" in the Streamlit sidebar. A 🔍 badge will appear indicating RAG is active.

### Configuration

```env
# .env
RAG_MAX_RESULTS=3  # Number of search results to fetch
```

---

## Monitoring Dashboard

The Admin Dashboard (`admin_app.py`) provides real-time observability into your inference gateway, separate from the chat interface to avoid disrupting the user experience.

### Features

- **Node Status**: Visual indicators showing Active (green), Cooldown (yellow), Offline (red)
- **Request Metrics**: Total requests, success rates, and rate limit counts
- **Request Distribution**: Bar chart showing load distribution across providers
- **Live Logs**: Real-time streaming of gateway events (provider switching, errors, recoveries)
- **Live System Logs (File)**: Persistent log viewer reading from `system.log` file
- **Auto-refresh**: Dashboard updates every 10 seconds (only in admin app)
- **Strategy Control**: Change load balancing strategy on-the-fly

### Security

The Admin Dashboard requires two-factor authentication:

1. **ACCESS_TOKEN**: Same API token used for the chat app
2. **ADMIN_PASSWORD**: Additional password configured in `.env`

```env
ADMIN_PASSWORD=your_secure_admin_password
```

If `ADMIN_PASSWORD` is not set, only the ACCESS_TOKEN is required.

### Accessing the Dashboard

1. Start the admin app: `streamlit run admin_app.py --server.port 8502`
2. Open `http://localhost:8502`
3. Enter your Access Token and Admin Password
4. Click Login

### Monitoring API

```bash
# Get comprehensive stats
curl -H "Authorization: Bearer TOKEN" http://localhost:8000/v1/monitoring/stats

# Get recent logs
curl -H "Authorization: Bearer TOKEN" http://localhost:8000/v1/monitoring/logs?count=10

# Lightweight health check (for polling)
curl http://localhost:8000/v1/monitoring/health
```

### Cooldown Configuration

When a provider returns HTTP 429 (rate limited), it enters cooldown:

```env
COOLDOWN_MINUTES=30  # Default: 30 minutes
```

Nodes automatically recover after the cooldown period expires.

---

## Provider Configuration

### Environment Variables (Recommended)

```env
# Cloud providers
CLOUDFLARE_ACCOUNT_1_ID=abc123...
CLOUDFLARE_ACCOUNT_1_TOKEN=xyz789...
CLOUDFLARE_ACCOUNT_1_NAME=Provider-Alpha
CLOUDFLARE_MODEL=@cf/meta/llama-3.1-8b-instruct

# Networking (Docker sets these automatically)
BACKEND_URL=http://backend:8000  # Service name in Docker
LOG_FILE=/app/logs/system.log    # Shared volume path

# Local fallback
OLLAMA_ENDPOINT=http://host.docker.internal:11434  # Docker on Windows/Mac
```

### JSON Configuration (Alternative)

Create `providers.json` from the template:
```bash
cp providers.example.json providers.json
# Edit with your credentials
```

---

## Local Fallback Setup

For high availability, configure Ollama as a local fallback:

### Installing Ollama

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.1 (recommended for better accuracy)
ollama pull llama3.1
```

### Lightweight Configuration

The gateway automatically configures Ollama for constrained environments:

- **Token Limit**: 1024 (configurable via `OLLAMA_MAX_TOKENS`)
- **Single Concurrent Request**: Prevents VRAM exhaustion
- **CPU Thread Limit**: Preserves system responsiveness

---

## Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 6+ cores |
| RAM | 8 GB | 16 GB |
| GPU (for fallback) | 4GB VRAM | 8GB+ VRAM |
| Network | Stable broadband | Low-latency connection |

**Tested On**: Acer Aspire 7 with RTX 3050 (4GB VRAM), Pop!_OS

---

## Security Considerations

1. **Never commit credentials**: `.env` and `providers.json` are gitignored
2. **Rotate tokens regularly**: Generate new access tokens periodically
3. **Use HTTPS in production**: Deploy behind a reverse proxy (nginx/Caddy)
4. **Limit network exposure**: Bind to localhost unless needed externally

---

## Troubleshooting

### Common Issues

**"No available providers"**
- Check your Cloudflare API tokens are valid
- Verify account IDs are correct
- Run `/v1/providers/reset` to clear cooldowns

**"Local fallback failed"**
- Ensure Ollama is running: `ollama serve`
- Check the model is pulled: `ollama list`
- Verify endpoint: `curl http://localhost:11434/api/tags`

**"Connection refused"**
- Start the API gateway: `python api_gateway.py`
- Check the port isn't in use: `lsof -i :8000`

### Logs

The system uses centralized logging with both file and console output:

**Log File Location:**
- Local: `./system.log`
- Docker: `/app/logs/system.log` (shared volume between backend and admin-ui)

**Log Format:**
```
2026-03-11 14:30:00 | INFO | api_gateway | Attempt 1: Using Provider-Alpha
2026-03-11 14:30:01 | WARNING | provider_manager | Node Provider-Alpha rate limited (429), cooldown: 30min
2026-03-11 14:30:01 | INFO | api_gateway | Attempt 2: Using Provider-Beta
2026-03-11 14:30:05 | INFO | provider_manager | Falling back to local Ollama
```

**View Logs:**
- Admin Dashboard: Expand "📋 Live System Logs (File)" section
- Docker: `docker-compose logs -f backend`
- Local: `tail -f system.log`

---

## Development

### Project Structure

```
easyResearchAssistant/
├── api_gateway.py         # FastAPI backend (port 8000)
├── provider_manager.py    # Distributed provider orchestration
├── search_tool.py         # RAG web search utility (DuckDuckGo)
├── streamlit_app.py       # Chat UI (port 8501)
├── admin_app.py           # Admin monitoring dashboard (port 8502)
├── docker-compose.yml     # Docker multi-service configuration
├── Dockerfile             # Container image definition
├── requirements.txt       # Python dependencies
├── .env.template          # Configuration template
├── providers.example.json # Provider config template (alternative)
├── system.log             # Centralized log file (auto-generated)
└── README.md              # This file
```

### Running Tests

```bash
pytest tests/ -v
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.
