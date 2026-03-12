# Medical RAG System for Plastic Surgery

A comprehensive multi-agent RAG (Retrieval-Augmented Generation) system with conversational memory, designed for querying plastic surgery research papers across multiple subspecialties.

## 🌟 Key Features

- **Multi-Agent Architecture**: Dedicated specialist agents for each plastic surgery subspecialty
- **Conversational Memory**: Natural follow-up questions without repeating context
- **Medical Accuracy**: Mandatory citations, evidence grading, hallucination prevention
- **Intelligent Routing**: Automatic query classification to appropriate specialists
- **Paper Caching**: Faster responses through intelligent caching (50% speed improvement)
- **Multi-Specialist Consultation**: Complex queries get answers from multiple experts
- **User-Configurable**: Provide your own PubMed keywords for each subspecialty

## 🏗️ System Architecture

```
User Query
    ↓
Conversation Manager (Context + Memory)
    ↓
Query Classifier & Router
    ↓
Specialist Agents (Parallel Consultation)
├── Craniofacial Agent
├── Hand Surgery Agent
├── Aesthetic Agent
├── Reconstructive Agent
├── Burn Agent
├── Breast Agent
└── [More Agents...]
    ↓
Coordinator Agent (Synthesis)
    ↓
Response with Citations + Evidence Quality
```

## 🎯 Use Case Example

```
User: Start conversation
→ Session ID: abc-123

Turn 1: "What are DIEP flap complications?"
→ [Breast + Reconstructive Agents consulted]
→ 15 papers retrieved, full answer with citations
→ Processing time: 3.2s

Turn 2: "What about in obese patients?"
→ System detects follow-up, rewrites query
→ Rewritten: "What are DIEP flap complications in obese patients?"
→ Reuses 5 cached papers + retrieves 10 new papers
→ Processing time: 1.8s (faster due to cache!)

Turn 3: "Compare that to TRAM flap"
→ Rewrites: "Compare DIEP vs TRAM complications in obese patients"
→ Provides side-by-side comparison with evidence
→ All claims cited with PMIDs
```

## 📋 Subspecialty Agents

Each agent has its own knowledge base and vector store:

- **Craniofacial Surgery**: Cleft lip/palate, craniosynostosis, facial deformities
- **Hand Surgery**: Hand reconstruction, tendon repair, nerve injuries
- **Aesthetic Surgery**: Rhinoplasty, face lift, facial rejuvenation
- **Reconstructive Surgery**: Free flaps, tissue transfer, complex reconstruction
- **Burn Surgery**: Burn care, scar management, contracture release
- **Breast Surgery**: Breast reconstruction, DIEP/TRAM flaps, oncoplastic surgery
- **Microsurgery**: Microvascular techniques, replantation
- **Pediatric Plastic Surgery**: Congenital anomalies, pediatric reconstruction
- **Maxillofacial Surgery**: Jaw surgery, facial trauma
- **Body Contouring**: Abdominoplasty, body lifts

*More agents can be added based on your needs*

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- PubMed API key (free from NCBI)
- Anthropic API key (for Claude) or OpenAI API key

### Installation

```bash
# Clone repository
cd ~/GitHub/medical-rag-system

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys

# Configure agents with your PubMed keywords
cp config/agents.yaml.template config/agents.yaml
# Edit agents.yaml and add your keywords for each subspecialty

# Start infrastructure (Qdrant, Redis, PostgreSQL)
docker-compose up -d

# Initialize first agent (test)
python scripts/initialize_agent.py --agent-id craniofacial

# Test the agent
python scripts/test_conversation.py

# Initialize all agents
python scripts/initialize_all_agents.py

# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Access API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📚 Documentation

- **[DESIGN_DOCUMENT.md](./DESIGN_DOCUMENT.md)**: Complete comprehensive design (130+ pages)
  - System architecture
  - Data acquisition from PubMed
  - Document processing & chunking
  - Vector store setup
  - Multi-agent system
  - Conversational memory
  - Retrieval & generation
  - Implementation roadmap
  - Technology stack
  - Cost estimates

## 🔧 Technology Stack

### Core Technologies

- **Data**: PubMed E-utilities, Biopython
- **Processing**: GROBID (PDF extraction), scispaCy (medical NER)
- **Embeddings**: PubMedBERT (medical domain model)
- **Vector Store**: Qdrant
- **LLM**: Claude 3.5 Sonnet (Anthropic)
- **Memory**: Redis (session storage)
- **API**: FastAPI + Uvicorn
- **Infrastructure**: Docker

### Key Libraries

```
anthropic>=0.18.0
qdrant-client>=1.7.0
sentence-transformers>=2.3.0
scispacy>=0.5.3
redis>=5.0.0
fastapi>=0.109.0
```

## 💡 Key Features Explained

### 1. Conversational Memory

The system remembers context across conversation turns:

```python
# Turn 1
User: "What is DIEP flap surgery?"
System: [Answers with full context]

# Turn 2 - System understands context
User: "What are the complications?"
System: Internally rewrites to "What are the complications of DIEP flap surgery?"
```

### 2. Multi-Agent Consultation

Complex queries consult multiple specialists:

```python
Query: "How do you treat facial burns in children?"

System Routes to:
├── Burn Surgery Agent → Acute burn management
├── Pediatric Surgery Agent → Pediatric considerations
└── Reconstructive Agent → Long-term reconstruction

Coordinator: Synthesizes all perspectives into unified answer
```

### 3. Paper Caching

Cached papers are reused across conversation turns for efficiency:

```
Turn 1: Retrieves 10 papers (3s)
Turn 2: Reuses 5 cached papers + 5 new papers (1.5s - faster!)
Turn 3: Reuses 7 cached papers + 3 new papers (1.2s)
```

### 4. Medical Accuracy

- **Citations**: Every claim includes PMID and author
- **Evidence Grading**: Level I-V evidence classification
- **Hallucination Prevention**: Multi-layer verification
- **Conservative Language**: Appropriate uncertainty expression

## 📊 API Usage

### Start Conversation

```bash
curl -X POST http://localhost:8000/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "doctor123"}'

# Returns: {"session_id": "abc-123"}
```

### Query

```bash
curl -X POST http://localhost:8000/conversation/abc-123/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are DIEP flap complications?"}'
```

### Get Conversation Summary

```bash
curl http://localhost:8000/conversation/abc-123/summary
```

### List Available Agents

```bash
curl http://localhost:8000/agents
```

## 🗂️ Project Structure

```
medical-rag-system/
├── config/
│   ├── system.yaml              # System configuration
│   ├── agents.yaml              # Agent configuration (user edits)
│   └── agents.yaml.template     # Template
├── data/
│   ├── agents/                  # Agent-specific data
│   │   ├── craniofacial/
│   │   ├── hand_surgery/
│   │   └── [other agents]/
│   └── evaluation/              # Test queries
├── src/
│   ├── agents/                  # Specialist agent code
│   ├── acquisition/             # PubMed fetching
│   ├── processing/              # PDF extraction, chunking
│   ├── vectorstore/             # Qdrant management
│   ├── retrieval/               # Hybrid search, reranking
│   ├── generation/              # LLM generation
│   ├── conversation/            # Memory management
│   ├── routing/                 # Query classification
│   └── api/                     # FastAPI endpoints
├── scripts/                     # Utility scripts
├── tests/                       # Unit and integration tests
├── docker/                      # Docker configuration
├── requirements.txt
├── DESIGN_DOCUMENT.md           # Comprehensive design (this doc)
└── README.md                    # This file
```

## 🎯 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)
- Infrastructure setup
- First pilot agent
- Basic RAG pipeline

### Phase 2: Multi-Agent System (Weeks 4-6)
- Initialize all agents
- Routing and coordination
- Multi-agent consultation

### Phase 3: Conversational Memory (Weeks 7-8)
- Session management
- Query rewriting
- Paper caching

### Phase 4: Production (Weeks 9-12)
- API development
- Evaluation and testing
- Deployment

## 💰 Cost Estimates

### With API-Based LLM (Claude 3.5 Sonnet)
- **10,000 queries/month**: ~$6-7K/month (with caching)
- **First query**: ~$0.90
- **Follow-up queries**: ~$0.45 (50% savings from caching)

### With Self-Hosted LLM
- **GPU Server**: $500-2,000/month
- **Total**: ~$600-2,400/month (significant savings at scale)

### Storage & Infrastructure
- Vector DB: $50-200/month (or $0 self-hosted)
- Redis: $20-50/month
- Storage: $10-50/month

## 🔒 Medical Accuracy & Safety

### Built-in Safeguards

1. **Citation Enforcement**: All claims require PMID citations
2. **Evidence Grading**: Clear indication of evidence strength
3. **Hallucination Detection**: Multi-layer fact verification
4. **Conservative Generation**: Low temperature, hedging language
5. **Expert Review**: Designed for expert medical professional use
6. **Disclaimer**: Clear educational purpose statement

### Important Note

⚠️ **This system is for educational and research purposes only. Clinical decisions should be made by qualified healthcare professionals considering individual patient circumstances.**

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test specific component
pytest tests/test_agents.py
pytest tests/test_conversation.py
pytest tests/test_retrieval.py

# Test with evaluation dataset
python scripts/evaluate_agents.py --test-set data/evaluation/test_queries.json
```

## 📈 Monitoring

### System Dashboard

```bash
# Access Grafana dashboard
http://localhost:3000

# Key metrics:
- Query volume
- Response times
- Cache hit rates
- Agent usage statistics
- Error rates
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

[Your License Choice - e.g., MIT, Apache 2.0]

## 👥 Authors

- **Your Name** - Initial work and design

## 🙏 Acknowledgments

- PubMed/NCBI for research paper access
- Microsoft for PubMedBERT
- Anthropic for Claude API
- Open source community for supporting libraries

## 📞 Support

- **Documentation**: See [DESIGN_DOCUMENT.md](./DESIGN_DOCUMENT.md)
- **Issues**: GitHub Issues
- **Email**: your-email@example.com

## 🔄 Version History

- **v1.0.0** (2026-03-12): Initial comprehensive design
  - Multi-agent architecture
  - Conversational memory
  - User-configurable keywords
  - Complete documentation

## 🚦 Status

🟡 **In Development** - Design complete, implementation in progress

---

**Built with ❤️ for advancing medical research accessibility**
