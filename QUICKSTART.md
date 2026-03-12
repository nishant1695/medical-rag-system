# Quick Start Guide - Data Pipeline

## Get Started in 5 Minutes

### Step 1: Initial Setup

```bash
# Clone/navigate to repository
cd /Users/nishantsingh/GitHub/medical-rag-system

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your credentials
nano .env

# Required:
PUBMED_EMAIL=your_email@example.com
PUBMED_API_KEY=your_api_key_here  # Optional but recommended
```

**Get PubMed API Key (Free):**
1. Go to: https://www.ncbi.nlm.nih.gov/account/
2. Sign in or create account
3. Go to Settings → API Key Management
4. Create new API key
5. Copy to .env file

### Step 3: Validate Setup

```bash
# Test everything is working
python tests/test_data_pipeline.py
```

**Expected output:**
```
✅ ALL TESTS PASSED!
You're ready to fetch papers!
```

### Step 4: Fetch Your First Papers

```bash
# Fetch 50 papers for breast surgery
python scripts/01_fetch_papers.py --subspecialty breast --max-papers 50
```

**Expected output:**
```
🔍 Searching PubMed...
✓ Found 100 papers
📥 Fetching paper details...
[Progress bar]
✓ Retrieved 50 papers
💾 Saving papers...
✓ Saved 50 new papers
Skipped 0 duplicates

✅ Fetch complete!
Papers saved to: data/raw_papers/breast/
```

### Step 5: Verify Data

```bash
# Check what was downloaded
ls data/raw_papers/breast/

# View a sample paper
cat data/raw_papers/breast/pmid-12345678.json | head -20
```

---

## Troubleshooting

### Error: "PUBMED_EMAIL not set"
**Fix:** Edit `.env` and set `PUBMED_EMAIL=your_email@example.com`

### Error: "No papers found"
**Possible causes:**
1. Network connection issues
2. PubMed API temporarily down
3. Query too specific (no results)

**Fix:** Check your internet connection, try again in a few minutes

### Error: "Rate limit exceeded"
**Fix:** Add PUBMED_API_KEY to .env for higher rate limits (10 req/s vs 3 req/s)

### Error: "Import error"
**Fix:** Make sure you activated virtual environment: `source venv/bin/activate`

---

## What's Next?

After fetching papers:

1. **Process papers** (extract text, metadata)
   ```bash
   python scripts/02_process_papers.py --subspecialty breast
   ```

2. **Chunk papers** (structure-aware chunking)
   ```bash
   python scripts/03_chunk_papers.py --subspecialty breast
   ```

3. **Create vector store** (embed and index)
   ```bash
   python scripts/04_create_vectorstore.py --subspecialty breast
   ```

---

## Data Pipeline Status

After Step 4, you'll have:
- ✅ 50 papers downloaded with metadata
- ✅ Deduplication index created
- ✅ Papers organized by subspecialty
- ✅ Ready for processing stage

**Location:** `data/raw_papers/breast/`

**Next:** Process and chunk these papers
