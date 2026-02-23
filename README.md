# GitHub MCP Server

A self-hosted MCP HTTP server that connects to **Claude.ai as a custom connector**, providing full CRUD operations and semantic search over a single GitHub repository. Designed to run on TrueNAS Scale behind a Cloudflare Tunnel.

---

## Architecture

```
Claude.ai (Custom Connector)
        │  HTTPS (Streamable HTTP / MCP protocol)
        │
Cloudflare Tunnel  ─────────────────────────────────────────┐
        │                                                    │
TrueNAS Scale (Docker Container)                            │
  ┌─────────────────────────────────────────────────┐       │
  │  MCP Server  (FastMCP + Starlette + Uvicorn)     │       │
  │                                                 │       │
  │  ┌───────────────┐  ┌──────────────────────┐    │       │
  │  │  15 MCP Tools │  │  OAuth 2.1 Provider  │◄───┘       │
  │  │ (CRUD + Search)│  │ (PKCE, JWT tokens)   │            │
  │  └──────┬────────┘  └──────────────────────┘            │
  │         │                                               │
  │  ┌──────▼────────────────────────────────────────────┐  │
  │  │            Local Git Clone  (/data/repo)           │  │
  │  │  • All CRUD = local filesystem ops (no GitHub API) │  │
  │  │  • git pull  every 5 min                           │  │
  │  │  • git push  2 min after last write (debounced)    │  │
  │  └───────────────────────┬───────────────────────────┘  │
  │                          │                              │
  │  ┌───────────────────────▼───────────────────────────┐  │
  │  │  Semantic Search Engine  (/data/index)             │  │
  │  │  • nomic-embed-text-v1.5   (768-dim embeddings)    │  │
  │  │  • FAISS IndexIDMap        (cosine similarity)     │  │
  │  │  • cross-encoder reranker  (ms-marco-MiniLM-L6)    │  │
  │  │  • Incremental: only re-index changed files        │  │
  │  └───────────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────────┘
```

---

## Features

### 15 MCP Tools

| Category | Tool | Description |
|----------|------|-------------|
| **Read** | `list_folder` | List folders and files in a directory |
| | `list_files` | List files (optionally recursive) |
| | `read_file` | Read full file content |
| | `read_range` | Read a specific character range `start-end` |
| | `search_files` | Grep-style text search across files |
| **Write** | `add_folder` | Create a new folder |
| | `add_file` | Create a new file with content |
| | `edit_file` | Replace entire file content |
| | `append_to_file` | Append content to end of file |
| | `insert_at_position` | Insert content at a character offset |
| **Manage** | `rename_file` | Rename / move a file |
| | `rename_folder` | Rename / move a folder |
| | `delete_file` | Delete a file |
| | `delete_folder` | Delete a folder and all contents |
| **Search** | `semantic_search` | AI-powered semantic search with file + position results |

### Semantic Search Response Format
```json
[
  {
    "file_name": "src/config.py",
    "position_range": "0-1000",
    "score": 8.432,
    "preview": "Application configuration loaded from environment variables..."
  }
]
```

---

## Quick Start (Local Development)

### 1. Prerequisites
```bash
python 3.11+
git
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env with your real values
```

The minimum required variables:
```
GITHUB_REPO_URL=https://github.com/your-user/your-repo.git
GITHUB_TOKEN=ghp_your_personal_access_token
GITHUB_BRANCH=main
OAUTH_CLIENT_ID=local-test-client
OAUTH_CLIENT_SECRET=local-test-secret
OAUTH_ISSUER_URL=http://localhost:8000
JWT_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
```

### 4. Run the server
```bash
python -m src.main
# or
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Verify
```bash
curl http://localhost:8000/health
curl http://localhost:8000/.well-known/oauth-authorization-server
```

### 6. Test with MCP Inspector
```bash
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

---

## Docker Deployment

### Build and run
```bash
# Build image (downloads ML models during build ~500MB)
docker build -f deploy/Dockerfile -t github-mcp-server:latest .

# Run with docker-compose
cp .env.example .env   # fill in your values
docker compose -f deploy/docker-compose.yml up -d

# Verify
curl http://localhost:8000/health
```

---

## TrueNAS Scale Deployment

### Step 1: Build and push Docker image
```bash
docker build -f deploy/Dockerfile -t YOUR_DOCKERHUB_USER/github-mcp-server:latest .
docker push YOUR_DOCKERHUB_USER/github-mcp-server:latest
```

### Step 2: Install custom app on TrueNAS Scale
1. Open TrueNAS Scale web UI → **Apps** → **Discover Apps** → **Custom App**
2. Open `deploy/truenas-app.yaml` and edit:
   - Replace `your-dockerhub-user/github-mcp-server:latest` with your image
   - Fill in all env var values (GitHub token, OAuth credentials, etc.)
3. Paste the YAML into the configuration field and click **Install**

### Step 3: Cloudflare Tunnel
1. Install `cloudflared` on TrueNAS or use the Docker Compose option in `deploy/cloudflare-tunnel.yml`
2. Authenticate: `cloudflared tunnel login`
3. Create tunnel: `cloudflared tunnel create github-mcp-server`
4. Edit `deploy/cloudflare-tunnel.yml` — replace `<YOUR_TUNNEL_ID>` and your hostname
5. Run: `cloudflared tunnel --config deploy/cloudflare-tunnel.yml run`

---

## Connecting to Claude.ai

### Step 1: Set OAuth Issuer URL
Set `OAUTH_ISSUER_URL` in your `.env` to your public domain (e.g. `https://mcp.yourdomain.com`).

### Step 2: Add Claude Custom Connector
1. Go to **Claude.ai** → **Settings** → **Integrations** → **Add Custom Connector**
2. Enter your MCP server URL: `https://mcp.yourdomain.com/mcp`
3. Claude will fetch OAuth metadata from `/.well-known/oauth-authorization-server`
4. A browser window opens to your server's `/authorize` page — click **Authorize**
5. You're connected! All 15 tools are now available in Claude

---

## Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_REPO_URL` | ✅ | — | HTTPS clone URL of the target repo |
| `GITHUB_TOKEN` | ✅ | — | GitHub Personal Access Token (repo scope) |
| `GITHUB_BRANCH` | — | `main` | Branch to track |
| `OAUTH_CLIENT_ID` | ✅ | — | OAuth client ID for Claude connector |
| `OAUTH_CLIENT_SECRET` | ✅ | — | OAuth client secret |
| `OAUTH_ISSUER_URL` | ✅ | — | Public base URL of this server |
| `JWT_SECRET_KEY` | ✅ | — | Secret for signing JWT tokens |
| `HOST` | — | `0.0.0.0` | Bind host |
| `PORT` | — | `8000` | Bind port |
| `LOG_LEVEL` | — | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `REPO_DIR` | — | `/data/repo` | Local path for git clone |
| `INDEX_DIR` | — | `/data/index` | Local path for FAISS index |
| `PULL_INTERVAL` | — | `300` | Seconds between periodic git pulls |
| `PUSH_DEBOUNCE` | — | `120` | Seconds after last write before git push |

---

## Running Tests
```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Project Structure

```
src/
├── main.py              # App entry point, lifespan, routing
├── config.py            # Pydantic settings from env vars
├── git_sync/
│   ├── manager.py       # GitPython clone/pull/push/diff
│   └── debouncer.py     # 2-minute write debounce for push
├── tools/
│   ├── crud.py          # 14 CRUD MCP tool definitions
│   ├── search.py        # semantic_search MCP tool
│   └── validators.py    # Path traversal security
├── semantic/
│   ├── engine.py        # Search orchestrator (init, search, on_file_changed)
│   ├── chunker.py       # 1000-char chunks with 200-char overlap
│   ├── embedder.py      # nomic-embed-text-v1.5 wrapper
│   ├── reranker.py      # cross-encoder/ms-marco-MiniLM-L-6-v2 wrapper
│   └── faiss_index.py   # FAISS IndexIDMap: incremental add/remove/search
└── oauth/
    ├── provider.py      # OAuth 2.1 server (authorize, token, discovery)
    ├── tokens.py        # JWT access tokens + refresh tokens
    └── middleware.py    # Bearer token validation middleware

deploy/
├── Dockerfile           # Multi-stage build with pre-downloaded models
├── docker-compose.yml   # Local/Docker deployment
├── truenas-app.yaml     # TrueNAS Scale custom app Kubernetes YAML
└── cloudflare-tunnel.yml # Cloudflare Tunnel configuration template

tests/
├── test_crud.py         # CRUD and path validation tests
├── test_semantic.py     # Chunker and FAISS tests
├── test_oauth.py        # Token and PKCE tests
└── test_git_sync.py     # Debouncer and change detection tests
```

---

## Resource Requirements

| Component | Memory | Notes |
|-----------|--------|-------|
| nomic-embed-text-v1.5 | ~270 MB | Embedding model (CPU) |
| cross-encoder reranker | ~90 MB | Reranking model (CPU) |
| FAISS index | Scales with repo | ~1 MB per 1,000 chunks |
| Git clone | Scales with repo | Full clone stored locally |
| **Recommended total** | **4 GB RAM** | For a large repo with >100K files |

First startup takes 2-5 minutes to build the semantic index. Subsequent starts load the saved index instantly.
