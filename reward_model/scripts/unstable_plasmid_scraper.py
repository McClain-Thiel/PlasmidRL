#!/usr/bin/env python3
"""
Mine open-access papers for E. coli plasmid instability claims + accession IDs.

What it does:
1) Europe PMC search -> OA PMCID list
2) Fetch OA full-text XML for each PMCID
3) Parse:
   - Sentences containing instability terms
   - Nearby GenBank/ENA/DDBJ accession IDs
   - Addgene plasmid IDs
4) Save a CSV of hits; optionally fetch sequences via NCBI E-utilities to a FASTA.

Usage:
  python mine_instability_accessions.py --query "Escherichia coli AND (plasmid instability OR unstable plasmid OR ITR instability OR toxic insert)" --max-papers 200 --fetch-sequences --email you@example.com --ncbi-api-key <optional>

Notes:
- No manual PDF reading; only APIs/XML.
- Be considerate with rate limits (sleep between requests).
- Outputs:
  - data/plasmid_instability_hits.csv
  - data/sequences.fasta  (if --fetch-sequences)
"""

import argparse
import csv
import os
import re
import sys
import time
import html
import json
from typing import List, Dict, Tuple, Iterable, Optional

import requests

# ----------------------------
# Config / constants
# ----------------------------

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Instability/propagation keywords (expand/adjust as needed)
INSTABILITY_TERMS = [
    r"unstable", r"instabilit(?:y|ies)", r"plasmid loss", r"segregational instability",
    r"rearrang(?:e|ement|ed)", r"deletion", r"recombine(?:d|ation)",
    r"difficult to clone", r"difficult to propagate", r"toxic", r"burden",
    r"low temperature", r"Stbl\d", r"Stbl2", r"Stbl3", r"copy number",
    r"AAV ITR", r"ITR instability"
]
INSTABILITY_REGEX = re.compile(r"|".join(INSTABILITY_TERMS), re.IGNORECASE)

# Accession patterns (INSDC/GenBank/RefSeq/ENA/DDBJ)
ACCESSION_PATTERNS = [
    r"[A-Z]{1}\d{5,8}(?:\.\d+)?",        # e.g., X12345, J01849.1
    r"[A-Z]{2}\d{5,8}(?:\.\d+)?",        # e.g., AF234567, AB123456.2
    r"[A-Z]{3}\d{5,8}(?:\.\d+)?",        # some 3-letter prefixes exist
    r"[A-Z]{2}_[0-9]{6,}(?:\.\d+)?",     # RefSeq like NM_000000.1, NC_000913.3
    r"[A-Z]{2}\d{2}_[0-9]{6,}(?:\.\d+)?" # e.g., LR35_123456.1 (rare, but seen)
]
ACCESSION_REGEX = re.compile(r"\b(?:" + "|".join(ACCESSION_PATTERNS) + r")\b")

# Addgene IDs
ADDGENE_REGEX = re.compile(r"\b(?:Addgene\s*(?:plasmid)?\s*#?|plasmid\s*#)\s*(\d{3,7})\b", re.IGNORECASE)

# Simple sentence splitter (XML contains tags; we split on periods/semicolons with mercy)
SENT_SPLIT = re.compile(r"(?<=[\.\?\!;])\s+")

# ----------------------------
# Helpers
# ----------------------------

def safe_get(url: str, params: dict = None, sleep: float = 0.34, retries: int = 3, timeout: int = 30) -> Optional[requests.Response]:
    """GET with basic backoff + courtesy sleep."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if sleep:
                time.sleep(sleep)
            if resp.status_code == 200:
                return resp
        except requests.RequestException:
            pass
        time.sleep(0.75 * attempt)
    return None

def europepmc_search(query: str, max_papers: int = 200) -> List[Dict]:
    """Search Europe PMC for OA PMCID records matching query."""
    results = []
    cursor = "*"
    page_size = 100
    fetched = 0

    while fetched < max_papers:
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(page_size, max_papers - fetched),
            "cursorMark": cursor
        }
        r = safe_get(EUROPEPMC_SEARCH_URL, params)
        if not r:
            break
        data = r.json()
        hits = data.get("resultList", {}).get("result", [])
        if not hits:
            break

        # Keep only OA with PMCID (we need fullTextXML)
        for h in hits:
            if "pmcid" in h and h.get("isOpenAccess") == "Y":
                results.append(h)
                fetched += 1
                if fetched >= max_papers:
                    break

        cursor = data.get("nextCursorMark")
        if not cursor:
            break

    return results

def fetch_fulltext_xml(pmcid: str) -> Optional[str]:
    url = EUROPEPMC_FULLTEXT_URL.format(pmcid=pmcid)
    r = safe_get(url)
    if not r:
        return None
    # sometimes XML comes escaped or as string; ensure str and unescape any HTML entities
    txt = r.text
    return html.unescape(txt)

def extract_hits_from_xml(xml_text: str) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Return:
      sentence_hits: list of {sentence, accessions[], addgene_ids[]}
      all_accessions: uniq list across doc
      all_addgene: uniq list across doc
    """
    # Remove tags for sentence-level parsing but keep raw for regex just in case
    text = re.sub(r"<[^>]+>", " ", xml_text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = SENT_SPLIT.split(text)
    sentence_hits = []
    seen_accessions = set()
    seen_addgene = set()

    for sent in sentences:
        if INSTABILITY_REGEX.search(sent):
            accs = ACCESSION_REGEX.findall(sent)
            if accs:
                for a in accs:
                    seen_accessions.add(a)
            addgenes = ADDGENE_REGEX.findall(sent)
            if addgenes:
                for g in addgenes:
                    seen_addgene.add(g)
            if accs or addgenes:
                sentence_hits.append({
                    "sentence": sent.strip(),
                    "accessions": list(set(accs)),
                    "addgene_ids": list(set(addgenes)),
                })

    # Also scan whole text (some accessions listed in figure legends/tables near but not in same sentence)
    # Heuristic: capture extra accessions near global mentions of instability.
    if sentence_hits:
        global_accs = ACCESSION_REGEX.findall(text)
        for a in global_accs:
            seen_accessions.add(a)
        global_add = ADDGENE_REGEX.findall(text)
        for g in global_add:
            seen_addgene.add(g)

    return sentence_hits, sorted(seen_accessions), sorted(seen_addgene)

def ncbi_fetch_fasta(accession: str, email: str, api_key: Optional[str] = None, sleep: float = 0.35) -> Optional[str]:
    """Fetch FASTA via E-utilities (nuccore preferred; fallback to nucleotide)."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "nuccore",
        "id": accession,
        "rettype": "fasta",
        "retmode": "text",
        "email": email
    }
    if api_key:
        params["api_key"] = api_key
    r = safe_get(base, params=params, sleep=sleep)
    if r and r.status_code == 200 and r.text and r.text.startswith(">"):
        return r.text
    return None

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=False,
                        default='("Escherichia coli") AND (plasmid instability OR unstable plasmid OR ITR instability OR toxic insert OR "plasmid loss")',
                        help="Europe PMC query string")
    parser.add_argument("--max-papers", type=int, default=200, help="Max OA papers to process")
    parser.add_argument("--outdir", default="data", help="Output directory")
    parser.add_argument("--fetch-sequences", action="store_true", help="Fetch FASTA for found accessions")
    parser.add_argument("--email", default=None, help="Contact email for NCBI E-utilities etiquette")
    parser.add_argument("--ncbi-api-key", default=None, help="NCBI API key (optional, raises rate limits)")
    parser.add_argument("--min-sentences", type=int, default=1, help="Minimum instability sentences w/ IDs to keep paper")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "plasmid_instability_hits.csv")
    fasta_path = os.path.join(args.outdir, "sequences.fasta")

    # 1) Search
    print(f"[+] Searching Europe PMC for OA papers (max {args.max_papers})…")
    results = europepmc_search(args.query, max_papers=args.max_papers)
    print(f"[+] OA PMC hits with PMCID: {len(results)}")

    rows = []
    all_accessions_global = set()
    seq_written = set()

    # 2) Iterate papers
    for i, rec in enumerate(results, 1):
        pmcid = rec.get("pmcid")
        title = rec.get("title", "")
        journal = rec.get("journalTitle", "")
        year = rec.get("pubYear", "")
        doi = rec.get("doi", "")
        url = f"https://europepmc.org/article/PMC/{pmcid}" if pmcid else ""

        xml = fetch_fulltext_xml(pmcid)
        if not xml:
            continue

        sentence_hits, doc_accessions, doc_addgene = extract_hits_from_xml(xml)

        if len(sentence_hits) < args.min_sentences:
            continue

        # Record per-sentence hits
        for hit in sentence_hits:
            rows.append({
                "pmcid": pmcid,
                "title": title,
                "journal": journal,
                "year": year,
                "doi": doi,
                "url": url,
                "sentence": hit["sentence"],
                "accessions": ";".join(hit["accessions"]),
                "addgene_ids": ";".join(hit["addgene_ids"]),
            })

        for a in doc_accessions:
            all_accessions_global.add(a)

        print(f"[{i}/{len(results)}] {pmcid}: {len(sentence_hits)} hits | {len(doc_accessions)} accessions | {len(doc_addgene)} Addgene IDs")

    # 3) Save CSV
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[+] Wrote CSV: {csv_path} ({len(rows)} rows)")
    else:
        print("[!] No hits found that match criteria.")
        sys.exit(0)

    # 4) (Optional) Fetch sequences
    if args.fetch_sequences:
        if not args.email:
            print("[!] --email is recommended for NCBI E-utilities. Proceeding anyway.")
        n_fetched = 0
        with open(fasta_path, "w", encoding="utf-8") as outfa:
            for acc in sorted(all_accessions_global):
                if acc in seq_written:
                    continue
                fasta = ncbi_fetch_fasta(acc, email=args.email or "", api_key=args.ncbi_api_key)
                if fasta:
                    outfa.write(fasta.strip() + "\n")
                    seq_written.add(acc)
                    n_fetched += 1
                    # polite delay already in ncbi_fetch_fasta via safe_get
        print(f"[+] Wrote FASTA: {fasta_path} ({n_fetched} sequences)")

    print("[✓] Done.")

if __name__ == "__main__":
    main()

