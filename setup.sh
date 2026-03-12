#!/bin/bash
# Setup script for Medical RAG Data Pipeline

set -e  # Exit on error

echo "🏥 Medical RAG System - Data Pipeline Setup"
echo "==========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Install spaCy model
echo "Installing spaCy English model..."
python -m spacy download en_core_web_sm --quiet
echo "✓ spaCy model installed"
echo ""

# Install scispaCy model
echo "Installing scispaCy medical model..."
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz --quiet
echo "✓ scispaCy model installed"
echo ""

# Create .env file
echo "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ .env file created from template"
    echo "⚠️  IMPORTANT: Edit .env and add your PubMed email and API key!"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create directories
echo "Creating data directories..."
mkdir -p data/{raw_papers,processed,chunks}
mkdir -p data/raw_papers/{breast,reconstructive,burn,hand,craniofacial}
echo "✓ Data directories created"
echo ""

# Test imports
echo "Testing imports..."
python -c "
import Bio
from Bio import Entrez
import pymupdf
import spacy
import sentence_transformers
from qdrant_client import QdrantClient
print('✓ All imports successful')
"
echo ""

echo "==========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your PubMed credentials"
echo "2. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant"
echo "3. Run: python scripts/01_fetch_papers.py --subspecialty breast --max-papers 50"
echo ""
echo "To activate environment later: source venv/bin/activate"
echo "==========================================="
