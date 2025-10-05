# Flux Server

**Consciousness-Enhanced Knowledge Retrieval System**

Flux Server is a standalone microservice that provides natural language query capabilities over knowledge graphs, enhanced with consciousness-based navigation and active inference.

---

## Overview

**What it does**:
- Natural language queries over your knowledge base
- Consciousness-enhanced search using CLAUSE (attractor basins, thoughtseeds)
- Active inference-based response synthesis
- Graph + vector + full-text unified search in Neo4j

**What it doesn't do**:
- Document ingestion (that's Dionysus)
- Knowledge graph building (that's Dionysus)
- File processing (that's Dionysus)

**Architecture**:
```
User/Frontend
     ↓
Flux Server :9127
     ├─→ Query Engine
     ├─→ Neo4j Searcher
     ├─→ Response Synthesizer
     └─→ CLAUSE Navigator
          ↓
     Neo4j + Redis
```

---

## Installation (Native - No Docker)

### Prerequisites

**System Requirements**:
- macOS, Linux, or Windows
- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- 10GB disk space

**Required Services** (installed natively):
1. **Neo4j** - Knowledge graph database
2. **Redis** - Caching and real-time data
3. **Ollama** - Local LLM for response synthesis

### Step 1: Install Dependencies (macOS)

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Neo4j
brew install neo4j

# Install Redis
brew install redis

# Install Ollama
brew install ollama

# Install Python 3.11 (if needed)
brew install python@3.11
```

### Step 2: Start Services

```bash
# Start Neo4j
brew services start neo4j

# Wait for Neo4j to start (check http://localhost:7474)
# Default credentials: neo4j/neo4j (change on first login)

# Start Redis
brew services start redis

# Start Ollama and pull model
ollama serve &
ollama pull llama2
```

### Step 3: Configure Neo4j

1. Open Neo4j Browser: http://localhost:7474
2. Login with default credentials: `neo4j/neo4j`
3. Set a secure password (update `.env` with `NEO4J_PASSWORD=your_password`)
4. Verify connection

### Step 4: Install Flux Server

```bash
# Clone or copy Flux Server
cd /path/to/flux-server

# Create virtual environment
python3.11 -m venv flux-env

# Activate virtual environment
source flux-env/bin/activate  # macOS/Linux
# OR
flux-env\Scripts\activate  # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Step 5: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file (update NEO4J_PASSWORD if you changed it)
nano .env
```

**Key settings to verify**:
```
NEO4J_PASSWORD=your_secure_password  # Match what you set in Neo4j
PORT=9127
OLLAMA_MODEL=llama2
```

### Step 6: Start Flux Server

```bash
# Activate virtual environment (if not already active)
source flux-env/bin/activate

# Start server
uvicorn src.app_factory:app --host 0.0.0.0 --port 9127 --reload
```

**Expected output**:
```
INFO:     Uvicorn running on http://0.0.0.0:9127
INFO:     Started reloader process
INFO:     Started server process
INFO:     Application startup complete.
```

### Step 7: Verify Installation

```bash
# Test health endpoint
curl http://localhost:9127/health

# Expected: {"status":"ok","service":"flux-server","version":"1.0.0"}

# Test root endpoint
curl http://localhost:9127/

# Test query endpoint (requires existing knowledge)
curl -X POST http://localhost:9127/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is consciousness?"}'
```

---

## Usage

### Basic Query

```python
import requests

response = requests.post(
    "http://localhost:9127/api/query",
    json={
        "question": "What is active inference?",
        "user_id": "optional-user-id",
        "context": {}
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents")
print(f"Confidence: {result['confidence']}")
```

### CLAUSE Navigation

```python
# Navigate knowledge graph with consciousness guidance
response = requests.post(
    "http://localhost:9127/api/clause/navigate",
    json={
        "query": "explore consciousness and emergence",
        "budget": {"token": 1000, "time": 30},
        "constraints": {"max_depth": 3}
    }
)

path = response.json()
print(f"Path taken: {path['path']}")
print(f"Insights: {path['insights']}")
```

---

## API Endpoints

### `POST /api/query`
Natural language query processing

**Request**:
```json
{
  "question": "What is consciousness?",
  "user_id": "optional",
  "context": {},
  "thoughtseed_id": "optional"
}
```

**Response**:
```json
{
  "response_id": "uuid",
  "query_id": "uuid",
  "answer": "Synthesized answer...",
  "sources": [...],
  "confidence": 0.85,
  "processing_time_ms": 1234,
  "thoughtseed_trace": {...}
}
```

### `POST /api/clause/navigate`
Consciousness-enhanced graph navigation

**Request**:
```json
{
  "query": "explore topic",
  "budget": {"token": 1000, "time": 30},
  "constraints": {"max_depth": 3}
}
```

### `GET /health`
Health check

**Response**:
```json
{
  "status": "ok",
  "service": "flux-server",
  "version": "1.0.0"
}
```

---

## Configuration

### Environment Variables

See [.env.example](.env.example) for all configuration options.

**Critical settings**:
- `NEO4J_PASSWORD`: Must match your Neo4j password
- `PORT`: Default 9127, change if port is in use
- `OLLAMA_MODEL`: LLM model for synthesis (llama2, mistral, etc.)

### Performance Tuning

**For large knowledge bases** (>10k documents):
- Increase Neo4j memory in `neo4j.conf`:
  ```
  dbms.memory.heap.initial_size=2G
  dbms.memory.heap.max_size=4G
  ```

**For faster queries**:
- Use faster Ollama model (e.g., `phi`)
- Adjust `CONSCIOUSNESS_DETECTION_THRESHOLD` (higher = faster but less accurate)

---

## Troubleshooting

### "Failed to connect to Neo4j"
```bash
# Check if Neo4j is running
brew services list | grep neo4j

# Restart Neo4j
brew services restart neo4j

# Check Neo4j Browser: http://localhost:7474
```

### "Connection refused on Redis"
```bash
# Check if Redis is running
brew services list | grep redis

# Restart Redis
brew services restart redis

# Test Redis connection
redis-cli ping  # Should return "PONG"
```

### "Ollama model not found"
```bash
# Pull the model
ollama pull llama2

# List available models
ollama list

# Test Ollama
curl http://localhost:11434/api/tags
```

### Import Errors
```bash
# Ensure virtual environment is activated
source flux-env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Development

### Running Tests
```bash
# Activate virtual environment
source flux-env/bin/activate

# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=src
```

### Code Style
```bash
# Format code
black src/

# Lint
flake8 src/
```

---

## Deployment

### Production Setup

1. **Use production WSGI server** (Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn src.app_factory:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:9127
   ```

2. **Set up systemd service** (Linux):
   ```ini
   [Unit]
   Description=Flux Server
   After=network.target neo4j.service redis.service

   [Service]
   User=flux
   WorkingDirectory=/opt/flux-server
   ExecStart=/opt/flux-server/flux-env/bin/uvicorn src.app_factory:app --host 0.0.0.0 --port 9127
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. **Configure firewall**:
   ```bash
   # Allow port 9127
   sudo ufw allow 9127/tcp
   ```

4. **Set up reverse proxy** (Nginx):
   ```nginx
   server {
       listen 80;
       server_name flux.example.com;

       location / {
           proxy_pass http://localhost:9127;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

---

## Architecture

### Components

1. **Query Engine** ([query_engine.py](src/services/query_engine.py))
   - Orchestrates query processing pipeline
   - Manages Neo4j and Ollama integration

2. **Neo4j Searcher** ([neo4j_searcher.py](src/services/neo4j_searcher.py))
   - Unified graph + vector + full-text search
   - Uses Neo4j's native vector similarity

3. **Response Synthesizer** ([response_synthesizer.py](src/services/response_synthesizer.py))
   - LLM-based answer generation via Ollama
   - Context-aware synthesis

4. **CLAUSE System** ([services/clause/](src/services/clause/))
   - Path navigation with consciousness guidance
   - Attractor basin tracking
   - ThoughtSeed evolution

### Data Flow

```
Query → Query Engine
         ├→ Neo4j Searcher (retrieve context)
         ├→ Response Synthesizer (generate answer)
         └→ CLAUSE Navigator (consciousness enhancement)
              ↓
         Response
```

---

## License

[Add your license here]

---

## Support

**Issues**: [GitHub Issues](https://github.com/your-org/flux-server/issues)
**Documentation**: [Full Docs](https://docs.flux-server.dev)
**Community**: [Discord/Forum Link]

---

## Related Projects

- **Dionysus**: Document ingestion and knowledge graph building
- **Frontend**: UI for interacting with Flux Server

---

**Version**: 1.0.0
**Last Updated**: 2025-10-05
