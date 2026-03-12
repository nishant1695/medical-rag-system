"""
Stage 1: Fetch Papers from PubMed with Deduplication
Usage: python scripts/01_fetch_papers.py --subspecialty breast --max-papers 50
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from Bio import Entrez, Medline
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment
load_dotenv()

# Configure Entrez
Entrez.email = os.getenv("PUBMED_EMAIL", "your_email@example.com")
Entrez.api_key = os.getenv("PUBMED_API_KEY", "")

if not Entrez.email or Entrez.email == "your_email@example.com":
    print("⚠️  WARNING: PUBMED_EMAIL not set in .env file!")
    print("   PubMed requires a valid email address.")
    print("   Edit .env and set PUBMED_EMAIL=your_email@example.com")
    exit(1)

class PaperDeduplicator:
    """Handles paper deduplication using multiple strategies"""

    def __init__(self, index_file: str = "data/deduplication_index.json"):
        self.index_file = index_file
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load deduplication index"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {
            "pmid_to_canonical": {},
            "doi_to_canonical": {},
            "title_hash_to_canonical": {}
        }

    def _save_index(self):
        """Save deduplication index"""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def get_canonical_id(self, paper: Dict) -> str:
        """Get or create canonical ID for paper"""
        pmid = paper.get('pmid')
        doi = paper.get('doi')
        title = paper.get('title', '')

        # Strategy 1: Check PMID
        if pmid and pmid in self.index['pmid_to_canonical']:
            return self.index['pmid_to_canonical'][pmid]

        # Strategy 2: Check DOI
        if doi:
            normalized_doi = doi.lower().strip()
            if normalized_doi in self.index['doi_to_canonical']:
                return self.index['doi_to_canonical'][normalized_doi]

        # Strategy 3: Check title hash
        title_hash = self._hash_title(title)
        if title_hash in self.index['title_hash_to_canonical']:
            return self.index['title_hash_to_canonical'][title_hash]

        # New paper - create canonical ID
        canonical_id = f"pmid-{pmid}" if pmid else f"hash-{title_hash}"

        # Register in index
        if pmid:
            self.index['pmid_to_canonical'][pmid] = canonical_id
        if doi:
            self.index['doi_to_canonical'][doi.lower().strip()] = canonical_id
        self.index['title_hash_to_canonical'][title_hash] = canonical_id

        self._save_index()
        return canonical_id

    def _hash_title(self, title: str) -> str:
        """Create normalized hash of title"""
        # Remove punctuation, lowercase, trim
        import re
        normalized = re.sub(r'[^\w\s]', '', title.lower()).strip()
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def is_duplicate(self, paper: Dict) -> bool:
        """Check if paper is already indexed"""
        pmid = paper.get('pmid')
        doi = paper.get('doi')
        title = paper.get('title', '')

        if pmid and pmid in self.index['pmid_to_canonical']:
            return True

        if doi:
            normalized_doi = doi.lower().strip()
            if normalized_doi in self.index['doi_to_canonical']:
                return True

        title_hash = self._hash_title(title)
        if title_hash in self.index['title_hash_to_canonical']:
            return True

        return False


class PubMedFetcher:
    """Fetches papers from PubMed with rate limiting and error handling"""

    def __init__(self, subspecialty_config: Dict):
        self.config = subspecialty_config
        self.deduplicator = PaperDeduplicator()

    def build_query(self) -> str:
        """Build PubMed query from configuration"""
        keywords = self.config.get('pubmed_keywords', [])
        mesh_terms = self.config.get('mesh_terms', [])

        # Combine keywords with OR
        keyword_query = ' OR '.join([f'"{kw}"' for kw in keywords])

        # Add MeSH terms
        if mesh_terms:
            mesh_query = ' OR '.join(mesh_terms)
            full_query = f"({keyword_query}) OR ({mesh_query})"
        else:
            full_query = keyword_query

        # Add plastic surgery context
        full_query = f"({full_query}) AND (plastic surgery[MeSH] OR reconstructive surgery[MeSH])"

        return full_query

    def search_pubmed(self, max_results: int = 50) -> List[str]:
        """Search PubMed and return list of PMIDs"""
        query = self.build_query()

        print(f"🔍 Searching PubMed...")
        print(f"   Query: {query[:100]}...")

        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results * 2,  # Fetch more to account for duplicates
                sort="relevance",
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()

            pmids = record['IdList']
            print(f"✓ Found {len(pmids)} papers")
            return pmids

        except Exception as e:
            print(f"❌ Error searching PubMed: {e}")
            return []

    def fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed information for list of PMIDs"""
        papers = []

        print(f"📥 Fetching paper details...")

        # Fetch in batches of 50
        batch_size = 50
        for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching"):
            batch_pmids = pmids[i:i+batch_size]

            try:
                # Fetch batch
                handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_pmids,
                    rettype="medline",
                    retmode="xml"
                )
                records = Entrez.read(handle)
                handle.close()

                # Parse each record
                for record in records['PubmedArticle']:
                    try:
                        paper = self._parse_record(record)
                        if paper:
                            papers.append(paper)
                    except Exception as e:
                        print(f"⚠️  Error parsing record: {e}")
                        continue

                # Rate limiting
                time.sleep(0.34)  # ~3 requests per second

            except Exception as e:
                print(f"⚠️  Error fetching batch: {e}")
                continue

        return papers

    def _parse_record(self, record) -> Optional[Dict]:
        """Parse PubMed XML record into paper dictionary"""
        try:
            medline = record['MedlineCitation']
            article = medline['Article']

            # Extract PMID
            pmid = str(medline['PMID'])

            # Extract title
            title = article.get('ArticleTitle', '')
            if isinstance(title, list):
                title = ' '.join([str(t) for t in title])

            # Extract abstract
            abstract = ''
            if 'Abstract' in article:
                abstract_sections = article['Abstract'].get('AbstractText', [])
                if isinstance(abstract_sections, list):
                    abstract_parts = []
                    for section in abstract_sections:
                        if isinstance(section, str):
                            abstract_parts.append(section)
                        else:
                            # Has label
                            label = section.attributes.get('Label', '')
                            text = str(section)
                            if label:
                                abstract_parts.append(f"{label}: {text}")
                            else:
                                abstract_parts.append(text)
                    abstract = '\n\n'.join(abstract_parts)
                else:
                    abstract = str(abstract_sections)

            # Extract authors
            authors = []
            if 'AuthorList' in article:
                for author in article['AuthorList']:
                    if 'LastName' in author and 'ForeName' in author:
                        authors.append(f"{author['LastName']} {author['ForeName']}")
                    elif 'CollectiveName' in author:
                        authors.append(author['CollectiveName'])

            # Extract journal
            journal = article['Journal']['Title']

            # Extract publication date
            pub_date = article['Journal']['JournalIssue'].get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '01')
            day = pub_date.get('Day', '01')

            # Try to build ISO date
            try:
                if year and month and day:
                    # Convert month name to number if needed
                    month_map = {
                        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                    }
                    if month in month_map:
                        month = month_map[month]
                    publication_date = f"{year}-{month.zfill(2)}-{str(day).zfill(2)}"
                else:
                    publication_date = year
            except:
                publication_date = year

            # Extract DOI
            doi = None
            if 'ELocationID' in article:
                for eloc in article['ELocationID']:
                    if eloc.attributes.get('EIdType') == 'doi':
                        doi = str(eloc)

            # Extract MeSH terms
            mesh_terms = []
            if 'MeshHeadingList' in medline:
                for mesh in medline['MeshHeadingList']:
                    descriptor = mesh['DescriptorName']
                    mesh_terms.append(str(descriptor))

            # Extract keywords
            keywords = []
            if 'KeywordList' in medline:
                for keyword_list in medline['KeywordList']:
                    keywords.extend([str(k) for k in keyword_list])

            # Extract publication types
            pub_types = []
            if 'PublicationTypeList' in article:
                pub_types = [str(pt) for pt in article['PublicationTypeList']]

            paper = {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'publication_date': publication_date,
                'year': year,
                'doi': doi,
                'mesh_terms': mesh_terms,
                'keywords': keywords,
                'publication_types': pub_types,
                'fetched_at': datetime.now().isoformat()
            }

            return paper

        except Exception as e:
            print(f"⚠️  Error parsing record: {e}")
            return None

    def save_papers(self, papers: List[Dict], subspecialty: str, output_dir: str):
        """Save papers to files with deduplication"""
        subspecialty_dir = Path(output_dir) / subspecialty
        subspecialty_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        duplicate_count = 0

        print(f"\n💾 Saving papers...")

        for paper in tqdm(papers, desc="Saving"):
            # Check for duplicate
            if self.deduplicator.is_duplicate(paper):
                duplicate_count += 1
                continue

            # Get canonical ID
            canonical_id = self.deduplicator.get_canonical_id(paper)

            # Add metadata
            paper['canonical_id'] = canonical_id
            paper['subspecialty'] = subspecialty

            # Save paper
            filename = subspecialty_dir / f"{canonical_id}.json"
            with open(filename, 'w') as f:
                json.dump(paper, f, indent=2)

            saved_count += 1

        print(f"\n✓ Saved {saved_count} new papers")
        print(f"  Skipped {duplicate_count} duplicates")

        # Save summary
        summary = {
            'subspecialty': subspecialty,
            'total_fetched': len(papers),
            'new_papers': saved_count,
            'duplicates': duplicate_count,
            'timestamp': datetime.now().isoformat()
        }

        summary_file = subspecialty_dir / 'fetch_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch papers from PubMed for a subspecialty'
    )
    parser.add_argument(
        '--subspecialty',
        type=str,
        required=True,
        choices=['breast', 'reconstructive', 'burn', 'hand', 'craniofacial'],
        help='Subspecialty to fetch papers for'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        default=50,
        help='Maximum number of papers to fetch (default: 50)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw_papers',
        help='Output directory for papers'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🏥 Medical RAG - PubMed Paper Fetcher")
    print("=" * 60)
    print(f"Subspecialty: {args.subspecialty}")
    print(f"Max papers: {args.max_papers}")
    print("")

    # Load subspecialty configuration
    with open('config/subspecialties.yaml', 'r') as f:
        config = yaml.safe_load(f)

    subspecialty_config = config['subspecialties'][args.subspecialty]

    print(f"Configuration for {subspecialty_config['name']}:")
    print(f"  Keywords: {', '.join(subspecialty_config['pubmed_keywords'][:3])}...")
    print("")

    # Create fetcher
    fetcher = PubMedFetcher(subspecialty_config)

    # Search PubMed
    pmids = fetcher.search_pubmed(max_results=args.max_papers)

    if not pmids:
        print("❌ No papers found. Check your query or network connection.")
        return

    # Fetch paper details
    papers = fetcher.fetch_paper_details(pmids)

    if not papers:
        print("❌ No papers retrieved. Check PubMed API status.")
        return

    print(f"\n✓ Retrieved {len(papers)} papers")

    # Save papers
    fetcher.save_papers(papers, args.subspecialty, args.output_dir)

    print("\n" + "=" * 60)
    print("✅ Fetch complete!")
    print("=" * 60)
    print(f"Papers saved to: {args.output_dir}/{args.subspecialty}/")
    print("")
    print("Next step:")
    print(f"  python scripts/02_process_papers.py --subspecialty {args.subspecialty}")
    print("")


if __name__ == '__main__':
    main()
