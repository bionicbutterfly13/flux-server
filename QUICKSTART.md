# Flux Server - Quick Start Guide

## 🎉 What's Been Accomplished

Flux Server has been successfully extracted as a **completely separate standalone project** from Dionysus!

### ✅ Working Right Now
- **Minimal server operational** on port 9127
- **Query endpoint** ready at `/api/query`
- **Health checks** working
- **Neo4j integration** configured
- **Git repository** initialized with 3 commits
- **Documentation** complete (README, STATUS, this file)

## 🚀 Running the Server (3 Simple Steps)

### 1. Start the Server
```bash
cd /Volumes/Asylum/dev/flux-server
source flux-env/bin/activate
uvicorn src.app_factory_minimal:app --host 127.0.0.1 --port 9127 --reload
```

### 2. Test It's Working
```bash
# Health check
curl http://127.0.0.1:9127/health

# Expected response:
# {"status":"ok","service":"flux-server-minimal","version":"1.0.0-minimal"}
```

### 3. View API Documentation
Open in browser: http://127.0.0.1:9127/docs

## 📁 Project Structure
```
flux-server/
├── src/
│   ├── api/
│   │   └── routes/
│   │       ├── query.py      # Working ✅
│   │       └── clause.py     # Future
│   ├── services/
│   │   ├── query_engine.py
│   │   ├── neo4j_searcher.py
│   │   ├── response_synthesizer.py
│   │   ├── ollama_integration.py
│   │   └── clause/           # Full CLAUSE directory copied
│   ├── models/
│   │   ├── query.py
│   │   ├── response.py
│   │   ├── attractor_basin.py
│   │   └── clause/           # All CLAUSE models
│   ├── config/
│   │   ├── settings.py
│   │   ├── neo4j_config.py
│   │   └── redis_config.py
│   ├── app_factory.py        # Full version (pending)
│   └── app_factory_minimal.py  # Working NOW ✅
├── flux-env/                 # Virtual environment
├── requirements.txt
├── .env                      # Your configuration
├── .env.example              # Template
├── README.md                 # Full documentation
├── STATUS.md                 # Detailed status
└── QUICKSTART.md             # This file
```

## 🔧 Configuration

Your `.env` file is already set up with:
- **Neo4j**: bolt://localhost:7687 (shared with Dionysus)
- **Redis**: redis://localhost:6379 (DB 0 for Flux, DB 1 for Dionysus)
- **Ollama**: http://localhost:11434 (for response synthesis)
- **Port**: 9127

## 🎯 Next Steps

### Immediate: Push to GitHub
```bash
# You need to create the repository first:
# 1. Go to https://github.com/bionicbutterfly13
# 2. Create new repository: flux-server
# 3. Then run:

git push -u origin main
```

### After GitHub Push
1. **Test query functionality** with real Neo4j data
2. **Document AutoSchemaKG integration** between Dionysus (write) and Flux (read)
3. **Update Dionysus** to run on port 9128
4. **Resolve CLAUSE model naming** for full CLAUSE integration

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Your System                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Dionysus (Port 9128)          Flux Server (Port 9127)      │
│  ┌────────────────┐           ┌─────────────────┐          │
│  │ Document Upload│           │ Query Processing│          │
│  │ LangGraph      │           │ Neo4j Search    │          │
│  │ Consciousness  │           │ Response        │          │
│  │ ThoughtSeeds   │           │ Synthesis       │          │
│  └───────┬────────┘           └────────┬────────┘          │
│          │                              │                   │
│          └──────────┬───────────────────┘                   │
│                     ↓                                        │
│           ┌──────────────────┐                              │
│           │   Neo4j Graph    │                              │
│           │ (Knowledge Base) │                              │
│           └──────────────────┘                              │
│                                                               │
│  Dionysus: WRITES knowledge                                 │
│  Flux: READS knowledge                                       │
│  Clear separation, shared database                           │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Current Status

- **Server**: ✅ Running on port 9127
- **Query Endpoint**: ✅ Available
- **CLAUSE Routes**: ⏸️ Pending (model naming issues)
- **Git**: ✅ Initialized, ready to push
- **Docs**: ✅ Complete

## 🐛 Known Issues

### CLAUSE Integration Pending
The full CLAUSE system (Phase 2 multi-agent) has naming mismatches between models and services:
- Models define `PathNavigationRequest/Response`
- Services expect `SubgraphRequest/Response`

**Solution**: Using minimal server now, CLAUSE integration incremental later.

## 📝 Quick Reference

### Start Server
```bash
cd /Volumes/Asylum/dev/flux-server
source flux-env/bin/activate
uvicorn src.app_factory_minimal:app --host 127.0.0.1 --port 9127 --reload
```

### Test Endpoints
```bash
curl http://127.0.0.1:9127/health         # Health
curl http://127.0.0.1:9127/               # Info
```

### Git Commands
```bash
git status                                 # Check status
git log --oneline                         # View commits
git push -u origin main                   # Push (after repo created)
```

## 🎓 Understanding the Separation

**Before**: One monolithic Dionysus backend doing both ingestion AND querying
**After**: Two clear services:

1. **Dionysus** (Port 9128): Perceptual processing
   - Receives documents
   - Processes with LangGraph
   - Generates consciousness insights
   - Writes to Neo4j
   - You are "fully aware" when using it

2. **Flux Server** (Port 9127): Knowledge retrieval
   - Queries Neo4j
   - Synthesizes responses
   - CLAUSE navigation (future)
   - Separate, focused responsibility

This is the architectural clarity you requested!

---

**Last Updated**: 2025-10-05
**Version**: 1.0.0-minimal
**Status**: Operational ✅
