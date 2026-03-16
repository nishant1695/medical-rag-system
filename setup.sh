#!/bin/bash
# Medical RAG System Setup Script

set -e
echo "🏥 Medical RAG System Setup"
echo ""

# Check Python
echo "1️⃣  Checking Python..."
python3 --version
echo ""

# Create venv
echo "2️⃣  Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo ""

# Install dependencies
echo "3️⃣  Installing dependencies..."
cd backend
pip install --upgrade pip
pip install -r requirements.txt
cd ..
echo ""

# Download models
echo "4️⃣  Downloading models..."
python -m spacy download en_core_sci_md
echo ""

# Setup .env
echo "5️⃣  Setting up .env..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env - EDIT WITH YOUR API KEYS!"
fi
echo ""

# Create directories
mkdir -p data/papers data/processed data/docling

# Start Docker
echo "6️⃣  Starting Docker services..."
docker-compose up -d
sleep 10
echo ""

# Init database
echo "7️⃣  Initializing database..."
python scripts/init_db.py
echo ""

echo "🎉 Setup Complete!"
echo "Run: cd backend && uvicorn app.main:app --reload"
