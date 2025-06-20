"""
Literature Search Utilities

Provides unified interface for searching multiple academic databases including
PubMed, ArXiv, CrossRef, and web academic sources.
"""

import requests
import xml.etree.ElementTree as ET
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from urllib.parse import quote
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Removed DuckDuckGo imports to avoid rate limiting issues

from core.config import get_config

logger = logging.getLogger(__name__)

class LiteratureSearchError(Exception):
    """Custom exception for literature search errors"""
    pass

class RateLimitError(LiteratureSearchError):
    """Exception for rate limiting"""
    pass

def search_pubmed(query: str, max_results: int = 3, email: str = None) -> List[Dict[str, Any]]:
    """
    Search PubMed for biomedical literature
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        email: Email for PubMed API (uses config if not provided)
        
    Returns:
        List of paper dictionaries
    """
    config = get_config()
    email = email or config.database.pubmed_email
    
    if not email or email == "your_email@example.com":
        logger.warning("PubMed email not configured, skipping PubMed search")
        return []
    
    try:
        logger.info(f"Searching PubMed for: {query}")
        
        # Step 1: Search for PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': email,
            'tool': 'ai_co_scientist'
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=10)
        search_response.raise_for_status()
        
        search_data = search_response.json()
        pmids = search_data.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            logger.warning(f"No PMIDs found for query: {query} - returning empty results, search will continue with other sources")
            return []
        
        # Rate limiting
        time.sleep(config.database.rate_limit_delay)
        
        # Step 2: Fetch article details
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': email,
            'tool': 'ai_co_scientist'
        }
        
        fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
        fetch_response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(fetch_response.text)
        articles = []
        
        for article in root.findall('.//PubmedArticle'):
            try:
                parsed_article = _parse_pubmed_article(article)
                if parsed_article:
                    articles.append(parsed_article)
            except Exception as e:
                logger.warning(f"Failed to parse PubMed article: {e}")
                continue
        
        logger.info(f"Retrieved {len(articles)} articles from PubMed")
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"PubMed API request failed: {e}")
        raise LiteratureSearchError(f"PubMed search failed: {e}")
    except Exception as e:
        logger.error(f"PubMed search failed: {e}")
        return []

def _parse_pubmed_article(article_xml) -> Optional[Dict[str, Any]]:
    """Parse a single PubMed article from XML"""
    try:
        title_elem = article_xml.find('.//ArticleTitle')
        abstract_elem = article_xml.find('.//AbstractText')
        pmid_elem = article_xml.find('.//PMID')
        
        # Extract authors
        authors = []
        for author in article_xml.findall('.//Author')[:3]:  # Limit to 3 authors
            last_name = author.find('.//LastName')
            fore_name = author.find('.//ForeName')
            if last_name is not None and fore_name is not None:
                authors.append(f"{fore_name.text} {last_name.text}")
        
        # Extract journal and publication date
        journal_elem = article_xml.find('.//Journal/Title')
        year_elem = article_xml.find('.//PubDate/Year')
        
        title = title_elem.text if title_elem is not None else "No title"
        abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
        pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
        journal = journal_elem.text if journal_elem is not None else "Unknown journal"
        year = year_elem.text if year_elem is not None else "Unknown year"
        
        # Extract key findings from abstract
        key_findings = _extract_key_findings(abstract)
        
        return {
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'journal': journal,
            'year': year,
            'pmid': pmid,
            'source': 'PubMed',
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
            'key_findings': key_findings,
            'relevance_score': 0.9,
            'citation_count': 0  # PubMed doesn't provide citation counts
        }
        
    except Exception as e:
        logger.warning(f"Failed to parse PubMed article: {e}")
        return None

def search_arxiv(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search ArXiv for preprints
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries
    """
    try:
        logger.info(f"Searching ArXiv for: {query}")
        
        base_url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            logger.warning(f"No papers found in ArXiv for: {query}")
            return []
        
        articles = []
        for entry in entries:
            try:
                parsed_article = _parse_arxiv_entry(entry, ns)
                if parsed_article:
                    articles.append(parsed_article)
            except Exception as e:
                logger.warning(f"Failed to parse ArXiv entry: {e}")
                continue
        
        logger.info(f"Retrieved {len(articles)} preprints from ArXiv")
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"ArXiv API request failed: {e}")
        raise LiteratureSearchError(f"ArXiv search failed: {e}")
    except Exception as e:
        logger.error(f"ArXiv search failed: {e}")
        return []

def _parse_arxiv_entry(entry, ns) -> Optional[Dict[str, Any]]:
    """Parse a single ArXiv entry from XML"""
    try:
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns)[:3]:  # Limit to 3 authors
            name = author.find('atom:name', ns)
            if name is not None:
                authors.append(name.text)
        
        # Extract publication date
        published = entry.find('atom:published', ns).text
        year = published[:4] if published else "Unknown"
        
        # Extract ArXiv URL
        arxiv_url = entry.find('atom:id', ns).text
        
        # Extract key findings
        key_findings = _extract_key_findings(summary)
        
        return {
            'title': title,
            'abstract': summary,
            'authors': authors,
            'journal': 'ArXiv Preprint',
            'year': year,
            'source': 'ArXiv',
            'url': arxiv_url,
            'key_findings': key_findings,
            'relevance_score': 0.75,  # Lower score for preprints
            'citation_count': 0
        }
        
    except Exception as e:
        logger.warning(f"Failed to parse ArXiv entry: {e}")
        return None

def search_crossref(query: str, max_results: int = 3, email: str = None) -> List[Dict[str, Any]]:
    """
    Search CrossRef for academic papers
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        email: Email for better rate limits (uses config if not provided)
        
    Returns:
        List of paper dictionaries
    """
    config = get_config()
    email = email or config.database.crossref_email or config.database.pubmed_email
    
    try:
        logger.info(f"Searching CrossRef for: {query}")
        
        base_url = "https://api.crossref.org/works"
        headers = {
            'User-Agent': f'AI-Co-Scientist/1.0 (mailto:{email})'
        }
        
        params = {
            'query': query,
            'rows': max_results,
            'sort': 'relevance',
            'select': 'title,author,published-print,abstract,DOI,URL,container-title'
        }
        
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        items = data.get('message', {}).get('items', [])
        
        if not items:
            logger.warning(f"No papers found in CrossRef for: {query}")
            return []
        
        articles = []
        for item in items:
            try:
                parsed_article = _parse_crossref_item(item)
                if parsed_article:
                    articles.append(parsed_article)
            except Exception as e:
                logger.warning(f"Failed to parse CrossRef item: {e}")
                continue
        
        logger.info(f"Retrieved {len(articles)} papers from CrossRef")
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"CrossRef API request failed: {e}")
        raise LiteratureSearchError(f"CrossRef search failed: {e}")
    except Exception as e:
        logger.error(f"CrossRef search failed: {e}")
        return []

def _parse_crossref_item(item) -> Optional[Dict[str, Any]]:
    """Parse a single CrossRef item"""
    try:
        # Extract title
        title_list = item.get('title', ['No title'])
        title = title_list[0] if title_list else 'No title'
        
        # Extract authors
        authors = []
        for author in item.get('author', [])[:3]:  # Limit to 3 authors
            given = author.get('given', '')
            family = author.get('family', '')
            if given or family:
                authors.append(f"{given} {family}".strip())
        
        # Extract publication date
        pub_date = item.get('published-print', {})
        year = None
        if pub_date and 'date-parts' in pub_date:
            year = pub_date['date-parts'][0][0] if pub_date['date-parts'][0] else None
        
        # Extract journal
        container = item.get('container-title', [])
        journal = container[0] if container else 'Unknown journal'
        
        return {
            'title': title,
            'abstract': 'Abstract not available from CrossRef',
            'authors': authors,
            'journal': journal,
            'year': str(year) if year else 'Unknown',
            'source': 'CrossRef',
            'url': item.get('URL', ''),
            'key_findings': [],
            'relevance_score': 0.8,
            'citation_count': 0
        }
        
    except Exception as e:
        logger.warning(f"Failed to parse CrossRef item: {e}")
        return None

def search_web_academic(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """
    Web search disabled to avoid rate limiting issues.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Empty list (web search disabled)
    """
    logger.info(f"Web search disabled for query: {query}")
    return []

def _extract_key_findings(text: str) -> List[str]:
    """Extract key findings from abstract text"""
    if not text or text == "No abstract available":
        return []
    
    key_findings = []
    keywords = ['significant', 'demonstrated', 'showed', 'found', 'revealed', 'discovered']
    sentences = text.split('.')
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in keywords):
            if 20 < len(sentence) < 150:  # Reasonable sentence length
                key_findings.append(sentence.strip())
                if len(key_findings) >= 3:  # Limit to 3 key findings
                    break
    
    return key_findings

def multi_source_literature_search(queries: List[str], 
                                  max_results_per_source: int = None,
                                  total_max_results: int = None) -> List[Dict[str, Any]]:
    """
    Search multiple literature sources with intelligent fallbacks
    
    Args:
        queries: List of search queries
        max_results_per_source: Max results per source per query
        total_max_results: Total max results across all sources
        
    Returns:
        List of deduplicated paper dictionaries
    """
    config = get_config()
    max_results_per_source = max_results_per_source or config.database.max_results_per_source
    total_max_results = total_max_results or config.database.total_max_results
    
    if not queries:
        raise ValueError("No search queries provided")
    
    logger.info(f"Starting multi-source search for {len(queries)} queries")
    
    all_results = []
    
    # Define search sources with their functions (web search disabled to avoid rate limits)
    search_sources = [
        ("PubMed", search_pubmed),
        ("ArXiv", search_arxiv),
        ("CrossRef", search_crossref)
    ]
    
    for query in queries:
        logger.info(f"Processing query: {query}")
        query_results = []
        
        for source_name, search_func in search_sources:
            try:
                results = search_func(query, max_results=max_results_per_source)
                if results:
                    query_results.extend(results)
                    logger.debug(f"{source_name}: {len(results)} results")
                else:
                    logger.debug(f"{source_name}: 0 results (continuing with other sources)")
                
                # Stop if we have enough results for this query
                if len(query_results) >= total_max_results:
                    break
                
                # Rate limiting between sources
                time.sleep(config.database.rate_limit_delay)
                
            except RateLimitError:
                logger.warning(f"{source_name} rate limited for query '{query}', continuing")
                continue
            except LiteratureSearchError as e:
                logger.warning(f"{source_name} failed for query '{query}': {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error in {source_name} for query '{query}': {e}")
                continue
        
        all_results.extend(query_results)
    
    # Remove duplicates based on title similarity
    unique_results = _deduplicate_results(all_results)
    
    # Sort by relevance score and citation count, but ensure source diversity
    unique_results.sort(key=lambda x: (x['relevance_score'], x['citation_count']), reverse=True)
    
    # Apply balanced selection to maintain source diversity in limited results
    final_results = _balanced_source_selection(unique_results, total_max_results)
    
    # Count contributing sources
    contributing_sources = set(result['source'] for result in final_results)
    total_sources_searched = len(search_sources)
    
    logger.info(f"Multi-source search completed: {len(final_results)} unique results from {len(all_results)} total")
    logger.info(f"Contributing sources: {len(contributing_sources)}/{total_sources_searched} sources provided results: {', '.join(sorted(contributing_sources))}")
    
    if not final_results:
        raise LiteratureSearchError("Multi-source search returned no results. Try broader search terms.")
    
    return final_results

def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate papers based on title similarity"""
    unique_results = []
    seen_titles = set()
    removed_count_by_source = {}
    
    for result in results:
        title_key = result['title'].lower().strip()
        source = result['source']
        
        # Skip if we've seen a very similar title
        if title_key not in seen_titles and title_key != "no title":
            # Check for similar titles (basic similarity check)
            is_duplicate = False
            for seen_title in seen_titles:
                if _titles_similar(title_key, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(title_key)
                unique_results.append(result)
            else:
                # Track removed duplicates by source
                removed_count_by_source[source] = removed_count_by_source.get(source, 0) + 1
                logger.debug(f"Duplicate removed: [{source}] {title_key[:60]}...")
        else:
            if title_key == "no title":
                logger.debug(f"Skipping result with no title: [{source}]")
            else:
                # Track removed duplicates by source  
                removed_count_by_source[source] = removed_count_by_source.get(source, 0) + 1
                logger.debug(f"Exact duplicate removed: [{source}] {title_key[:60]}...")
    
    # Log deduplication summary
    if removed_count_by_source:
        total_removed = sum(removed_count_by_source.values())
        logger.info(f"Deduplication removed {total_removed} duplicates:")
        for source, count in removed_count_by_source.items():
            logger.info(f"  • {source}: {count} duplicates removed")
    
    return unique_results

def _balanced_source_selection(results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
    """
    Select results with balanced representation from different sources.
    Ensures no source gets completely excluded when limiting results.
    """
    if len(results) <= max_results:
        return results
    
    # Group results by source
    by_source = {}
    for result in results:
        source = result['source']
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(result)
    
    # Calculate fair allocation per source
    num_sources = len(by_source)
    if num_sources == 0:
        return results[:max_results]
    
    base_per_source = max_results // num_sources
    remainder = max_results % num_sources
    
    selected = []
    
    # First pass: allocate base amount per source
    for source, source_results in by_source.items():
        allocation = base_per_source
        # Distribute remainder to first few sources
        if remainder > 0:
            allocation += 1
            remainder -= 1
        
        # Take the top-ranked results from this source
        selected.extend(source_results[:allocation])
    
    # Sort selected results to maintain overall quality ranking
    selected.sort(key=lambda x: (x['relevance_score'], x['citation_count']), reverse=True)
    
    return selected[:max_results]

def _titles_similar(title1: str, title2: str, threshold: float = 0.8) -> bool:
    """Check if two titles are similar (basic implementation)"""
    if not title1 or not title2:
        return False
    
    # Simple word overlap similarity
    words1 = set(title1.lower().split())
    words2 = set(title2.lower().split())
    
    if not words1 or not words2:
        return False
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity >= threshold

async def async_multi_source_search(queries: List[str], 
                                  max_results_per_source: int = None,
                                  total_max_results: int = None) -> List[Dict[str, Any]]:
    """Async version of multi-source literature search"""
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = loop.run_in_executor(
            executor, 
            multi_source_literature_search, 
            queries, 
            max_results_per_source, 
            total_max_results
        )
        return await future

if __name__ == "__main__":
    # Test literature search functions
    try:
        test_query = "machine learning drug discovery"
        
        print("🔍 Testing PubMed search...")
        pubmed_results = search_pubmed(test_query, max_results=2)
        print(f"✅ PubMed: {len(pubmed_results)} results")
        
        print("\n🔍 Testing ArXiv search...")
        arxiv_results = search_arxiv(test_query, max_results=2)
        print(f"✅ ArXiv: {len(arxiv_results)} results")
        
        print("\n🔍 Testing multi-source search...")
        all_results = multi_source_literature_search([test_query], max_results_per_source=1)
        print(f"✅ Multi-source: {len(all_results)} unique results")
        
        for result in all_results[:3]:
            print(f"   - {result['title'][:60]}... [{result['source']}]")
        
    except Exception as e:
        print(f"❌ Literature search test failed: {e}")
        print("\n💡 Make sure to:")
        print("   1. Set PUBMED_EMAIL in your .env file")
        print("   2. Check your internet connection")
        print("   3. Verify API endpoints are accessible")