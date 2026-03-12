"""
Test script to validate PubMed fetcher setup
Tests API connection and configuration before running full fetch
"""

import os
from dotenv import load_dotenv
from Bio import Entrez
import yaml

def test_environment():
    """Test environment configuration"""
    print("🧪 Testing Environment Configuration")
    print("=" * 60)

    # Load .env
    load_dotenv()

    # Check email
    email = os.getenv("PUBMED_EMAIL")
    if not email or email == "your_email@example.com":
        print("❌ PUBMED_EMAIL not set in .env")
        print("   Action: Edit .env and set PUBMED_EMAIL=your_email@example.com")
        return False
    else:
        print(f"✓ PUBMED_EMAIL: {email}")

    # Check API key (optional but recommended)
    api_key = os.getenv("PUBMED_API_KEY")
    if not api_key:
        print("⚠️  PUBMED_API_KEY not set (optional)")
        print("   Rate limit: 3 requests/second without key")
        print("   Rate limit: 10 requests/second with key")
    else:
        print(f"✓ PUBMED_API_KEY: {api_key[:10]}...")

    print("")
    return True


def test_pubmed_connection():
    """Test PubMed API connection"""
    print("🧪 Testing PubMed API Connection")
    print("=" * 60)

    load_dotenv()
    Entrez.email = os.getenv("PUBMED_EMAIL")
    Entrez.api_key = os.getenv("PUBMED_API_KEY", "")

    try:
        # Simple test search
        print("Searching for 'breast reconstruction'...")
        handle = Entrez.esearch(
            db="pubmed",
            term="breast reconstruction",
            retmax=5
        )
        record = Entrez.read(handle)
        handle.close()

        pmids = record['IdList']
        print(f"✓ Successfully retrieved {len(pmids)} PMIDs")
        print(f"  Sample PMIDs: {pmids[:3]}")
        print("")
        return True

    except Exception as e:
        print(f"❌ Error connecting to PubMed: {e}")
        print("   Check your internet connection")
        print("   Verify email is valid")
        return False


def test_configuration():
    """Test subspecialty configuration"""
    print("🧪 Testing Subspecialty Configuration")
    print("=" * 60)

    try:
        with open('config/subspecialties.yaml', 'r') as f:
            config = yaml.safe_load(f)

        subspecialties = config.get('subspecialties', {})
        print(f"✓ Found {len(subspecialties)} subspecialties:")

        for sub_id, sub_config in subspecialties.items():
            print(f"\n  {sub_id}:")
            print(f"    Name: {sub_config['name']}")
            print(f"    Keywords: {len(sub_config['pubmed_keywords'])} keywords")
            print(f"    MeSH terms: {len(sub_config.get('mesh_terms', []))} terms")
            print(f"    Max papers: {sub_config.get('max_papers', 'N/A')}")

        print("")
        return True

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False


def test_fetch_single_paper():
    """Test fetching a single known paper"""
    print("🧪 Testing Paper Fetch (Single Paper)")
    print("=" * 60)

    load_dotenv()
    Entrez.email = os.getenv("PUBMED_EMAIL")
    Entrez.api_key = os.getenv("PUBMED_API_KEY", "")

    # Known PMID for testing
    test_pmid = "32732678"  # Example paper

    try:
        print(f"Fetching PMID {test_pmid}...")
        handle = Entrez.efetch(
            db="pubmed",
            id=test_pmid,
            rettype="medline",
            retmode="xml"
        )
        records = Entrez.read(handle)
        handle.close()

        if records['PubmedArticle']:
            record = records['PubmedArticle'][0]
            article = record['MedlineCitation']['Article']

            print("✓ Successfully fetched paper")
            print(f"  Title: {article.get('ArticleTitle', '')[:80]}...")
            print(f"  Journal: {article['Journal']['Title']}")

            # Check abstract
            if 'Abstract' in article:
                abstract_text = str(article['Abstract']['AbstractText'][0])
                print(f"  Abstract length: {len(abstract_text)} characters")
            else:
                print("  ⚠️  No abstract available")

            print("")
            return True
        else:
            print("❌ No record found")
            return False

    except Exception as e:
        print(f"❌ Error fetching paper: {e}")
        return False


def test_deduplication():
    """Test deduplication system"""
    print("🧪 Testing Deduplication System")
    print("=" * 60)

    from scripts.fetch_papers import PaperDeduplicator

    dedup = PaperDeduplicator(index_file="data/test_dedup_index.json")

    # Test papers
    paper1 = {
        'pmid': '12345678',
        'title': 'Test Paper on DIEP Flap',
        'doi': '10.1097/TEST.0001',
        'year': '2023'
    }

    paper2 = {
        'pmid': '12345678',  # Same PMID
        'title': 'Test Paper on DIEP Flap',
        'doi': '10.1097/TEST.0001',
        'year': '2023'
    }

    paper3 = {
        'pmid': '87654321',  # Different PMID
        'title': 'Different Paper',
        'doi': '10.1097/TEST.0002',
        'year': '2023'
    }

    # Test 1: Get canonical ID
    id1 = dedup.get_canonical_id(paper1)
    print(f"✓ Paper 1 canonical ID: {id1}")

    # Test 2: Check duplicate
    is_dup = dedup.is_duplicate(paper2)
    print(f"✓ Paper 2 is duplicate: {is_dup}")
    assert is_dup == True, "Should detect duplicate"

    # Test 3: New paper
    id3 = dedup.get_canonical_id(paper3)
    print(f"✓ Paper 3 canonical ID: {id3}")
    assert id3 != id1, "Should have different canonical ID"

    # Cleanup
    import os
    if os.path.exists("data/test_dedup_index.json"):
        os.remove("data/test_dedup_index.json")

    print("")
    return True


def main():
    print("\n" + "=" * 60)
    print("🧪 MEDICAL RAG - DATA PIPELINE VALIDATION")
    print("=" * 60)
    print("")

    all_passed = True

    # Test 1: Environment
    if not test_environment():
        all_passed = False
        print("⚠️  Fix environment issues before proceeding\n")
        return

    # Test 2: Configuration
    if not test_configuration():
        all_passed = False
        print("⚠️  Fix configuration issues before proceeding\n")
        return

    # Test 3: PubMed connection
    if not test_pubmed_connection():
        all_passed = False
        print("⚠️  Cannot connect to PubMed. Check network and credentials\n")
        return

    # Test 4: Single paper fetch
    if not test_fetch_single_paper():
        all_passed = False
        print("⚠️  Paper fetch failed\n")

    # Test 5: Deduplication
    if not test_deduplication():
        all_passed = False
        print("⚠️  Deduplication test failed\n")

    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou're ready to fetch papers!")
        print("\nRun:")
        print("  python scripts/01_fetch_papers.py --subspecialty breast --max-papers 50")
        print("")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nFix the issues above before proceeding.")
        print("")


if __name__ == '__main__':
    main()
