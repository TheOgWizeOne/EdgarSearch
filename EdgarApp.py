# app.py — SEC EDGAR Term Finder (Boolean + Proximity + Whole-word + Normalization + Deep links)

import io
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Optional PDF support (page text extraction and page hint)
PDF_OK = False
try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
    PDF_OK = True
except Exception:
    PDF_OK = False

# -----------------------
# SEC client (public data.sec.gov)
# -----------------------

USER_AGENT_DEFAULT = "Your Name your.email@example.com (EDGAR research; Streamlit app)"
BASE = "https://data.sec.gov"

class SecClient:
    def __init__(self, user_agent: str = USER_AGENT_DEFAULT, rps: float = 4.0):
        self.sess = requests.Session()
        self.sess.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.min_interval = 1.0 / max(rps, 1e-6)
        self._last_ts = 0.0

    def _throttle(self):
        now = time.time()
        delta = now - self._last_ts
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last_ts = time.time()

    def _get(self, url: str, **kwargs) -> requests.Response:
        backoff = 1.0
        for _ in range(6):
            self._throttle()
            resp = self.sess.get(url, timeout=30, **kwargs)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            resp.raise_for_status()
        resp.raise_for_status()
        return resp

    @st.cache_data(ttl=3600, show_spinner=False)
    def ticker_table_cached(_self_dummy: object) -> Dict[str, Dict]:
        """
        Returns { 'AAPL': {'cik_str': 320193, 'title': 'Apple Inc.'}, ... }
        """
        url = "https://www.sec.gov/files/company_tickers.json"
        data = requests.get(url, headers={"User-Agent": USER_AGENT_DEFAULT}).json()
        out = {}
        for _, rec in data.items():
            out[str(rec["ticker"]).upper()] = {
                "cik_str": int(rec["cik_str"]),
                "title": rec.get("title", ""),
            }
        return out

    def cik_from_ticker(self, ticker: str, cache: Dict[str, Dict]) -> Optional[int]:
        rec = cache.get(ticker.upper())
        return int(rec["cik_str"]) if rec else None

    def submissions(self, cik: int) -> Dict:
        url = f"{BASE}/submissions/CIK{cik:010d}.json"
        return self._get(url).json()

    def _fetch_index_file(self, name: str) -> Dict:
        url = f"{BASE}/submissions/{name}"
        return self._get(url).json()

    def list_filings(self, cik: int, forms: List[str], start_date: Optional[str], limit_per_company: int) -> List[Dict]:
        sub = self.submissions(cik)
        recs: List[Dict] = []

        def harvest(block: Dict):
            n = len(block.get("accessionNumber", []))
            for i in range(n):
                recs.append({k: block[k][i] for k in block.keys()})

        # recent
        recent = sub.get("filings", {}).get("recent", {})
        harvest(recent)

        # historical
        for f in sub.get("filings", {}).get("files", []):
            idx = self._fetch_index_file(f["name"])
            hist = idx.get("filings", {}).get("recent", {})
            harvest(hist)

        # filters
        forms_up = {f.upper().strip() for f in forms if f.strip()}
        if forms_up:
            recs = [r for r in recs if str(r.get("form", "")).upper().strip() in forms_up]
        if start_date:
            recs = [r for r in recs if r.get("filingDate", "") >= start_date]

        recs.sort(key=lambda r: (r.get("filingDate", ""), r.get("accessionNumber", "")), reverse=True)
        if limit_per_company and limit_per_company > 0:
            recs = recs[:limit_per_company]
        return recs

    @staticmethod
    def _acc_path(accession_number: str) -> str:
        return accession_number.replace("-", "")

    def primary_url(self, cik: int, accession_number: str, primary_document: str) -> str:
        cik_no_zeros = str(int(cik))
        return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{self._acc_path(accession_number)}/{primary_document}"

    def download_primary(self, cik: int, accession_number: str, primary_document: str) -> Tuple[str, bytes, str]:
        """
        Returns (url, content_bytes, media_type_guess) where media_type_guess ∈ {'html','txt','pdf','bin'}.
        """
        url = self.primary_url(cik, accession_number, primary_document)
        r = self._get(url, stream=True)
        content = r.content
        # guess by extension + sniff
        ext = primary_document.lower().rsplit(".", 1)[-1] if "." in primary_document else ""
        if ext in ("htm", "html", "xhtml"):
            mtype = "html"
        elif ext in ("txt",):
            mtype = "txt"
        elif ext in ("pdf",):
            mtype = "pdf"
        else:
            if content[:5].lstrip().lower().startswith(b"<!doc") or b"<html" in content[:400].lower():
                mtype = "html"
            elif content[:4] == b"%PDF":
                mtype = "pdf"
            else:
                mtype = "txt"
        return url, content, mtype

# -----------------------
# Text extraction & deep-link helpers
# -----------------------

def bytes_to_text_html_or_txt(content: bytes, is_html: bool) -> str:
    try:
        txt = content.decode("utf-8", errors="ignore")
    except Exception:
        txt = content.decode("latin-1", errors="ignore")
    if is_html:
        soup = BeautifulSoup(txt, "html.parser")
        for t in soup(["script", "style"]):
            t.extract()
        return soup.get_text(separator=" ")
    return txt

def pdf_bytes_to_text(content: bytes) -> str:
    if not PDF_OK:
        return ""
    with io.BytesIO(content) as bio:
        try:
            return pdf_extract_text(bio) or ""
        except Exception:
            return ""

# Normalization to reduce false positives / Ctrl+F mismatches
def normalize_text_for_search(s: str, normalize: bool = True) -> str:
    if not normalize or not s:
        return s or ""
    # Common culprits
    s = s.replace("\u00a0", " ")  # NBSP -> space
    s = s.replace("\u00ad", "")   # soft hyphen (invisible)
    # Join words split by hyphen at end of line (PDFs)
    s = re.sub(r"-\s*\n\s*", "", s)
    # Collapse odd whitespace runs
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s

def make_text_fragment_from_exact(base_url: str, exact: str) -> str:
    """Best-effort text fragment using only the exact snippet (more robust after normalization)."""
    exact_clean = " ".join((exact or "").split())
    if not exact_clean:
        return base_url
    return f"{base_url}#:~:text={quote(exact_clean, safe='')}"

def make_pdf_search_url(base_url: str, term_for_viewer: str, full_text_raw: str, char_start_guess: int) -> str:
    """Open PDF with search prefilled; add page hint estimated from raw text page breaks (pdfminer)."""
    page = None
    try:
        page = full_text_raw[:char_start_guess].count("\x0c") + 1
    except Exception:
        page = None
    frag = f"#search={quote(term_for_viewer)}"
    if page and page > 0:
        frag = f"#page={page}&search={quote(term_for_viewer)}"
    return f"{base_url}{frag}"

# -----------------------
# Matching primitives (whole-word aware)
# -----------------------

def _build_pattern(phrase: str, whole_words: bool) -> re.Pattern:
    esc = re.escape(phrase)
    if whole_words:
        # Word boundaries that are robust for mixed unicode word chars
        return re.compile(r"(?i)(?<!\w)" + esc + r"(?!\w)")
    return re.compile(r"(?i)" + esc)

def find_matches(text: str, term: str, context: int, whole_words: bool) -> List[Tuple[int, int, str]]:
    pat = _build_pattern(term, whole_words)
    out = []
    for m in pat.finditer(text):
        s, e = m.start(), m.end()
        left = max(0, s - context); right = min(len(text), e + context)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ")
        out.append((s, e, snippet))
    return out

# -----------------------
# Boolean + Proximity engine
# -----------------------

WORD_RE = re.compile(r"\w+", re.UNICODE)

class Span(NamedTuple):
    start: int
    end: int
    snippet: str
    tindex: int  # token index (start token) for proximity

class BoolResult(NamedTuple):
    matched: bool
    span: Optional[Span]  # representative span to jump to

def _normalize_ws(s: str) -> str:
    return " ".join(s.split())

def _build_token_index(text: str) -> List[int]:
    return [m.start() for m in WORD_RE.finditer(text)]

def _char_to_token_index(token_starts: List[int], pos: int) -> int:
    import bisect
    if not token_starts:
        return 0
    i = bisect.bisect_right(token_starts, pos)
    return max(0, i - 1)

def _find_all_spans(text: str, phrase: str, context: int, token_starts: List[int], whole_words: bool) -> List[Span]:
    pat = _build_pattern(phrase, whole_words)
    spans: List[Span] = []
    for m in pat.finditer(text):
        s, e = m.start(), m.end()
        left = max(0, s - context); right = min(len(text), e + context)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ")
        tindex = _char_to_token_index(token_starts, s)
        spans.append(Span(s, e, snippet, tindex))
    return spans

# Tokenizer supports: WORD/PHRASE, AND, OR, (, ), NEAR/n, WITHIN/n
def _tokenize_bool(q: str) -> List[str]:
    tokens: List[str] = []
    i, n = 0, len(q)
    WHSP = set(" \t\r\n")
    while i < n:
        c = q[i]
        if c in WHSP:
            i += 1; continue
        if c in "()":
            tokens.append(c); i += 1; continue
        if c == '"':
            j = i + 1; buf: List[str] = []
            while j < n and q[j] != '"':
                buf.append(q[j]); j += 1
            if j >= n:
                tokens.append('"' + "".join(buf))  # unclosed; accept rest
                i = n
            else:
                tokens.append('"' + "".join(buf) + '"'); i = j + 1
            continue
        tail_up = q[i:].upper()
        if tail_up.startswith("NEAR/") or tail_up.startswith("WITHIN/"):
            k = i
            while k < n and (q[k] not in WHSP) and (q[k] not in "()"):
                k += 1
            tokens.append(q[i:k]); i = k; continue
        j = i
        while j < n and (q[j] not in WHSP) and (q[j] not in '()"'):
            j += 1
        tokens.append(q[i:j]); i = j
    return tokens

# AST: proximity > AND > OR
class Node: 
    def eval(self, spans_map: Dict[str, List[Span]]) -> BoolResult: ...

class PhraseNode(Node):
    def __init__(self, phrase: str): self.phrase = phrase
    def eval(self, spans_map: Dict[str, List[Span]]) -> BoolResult:
        lst = spans_map.get(self.phrase.lower(), [])
        return BoolResult(bool(lst), min(lst, key=lambda s: s.start) if lst else None)

class AndNode(Node):
    def __init__(self, left: Node, right: Node): self.left, self.right = left, right
    def eval(self, spans_map: Dict[str, List[Span]]) -> BoolResult:
        l = self.left.eval(spans_map)
        if not l.matched: return BoolResult(False, None)
        r = self.right.eval(spans_map)
        if not r.matched: return BoolResult(False, None)
        cands = [s for s in (l.span, r.span) if s]
        return BoolResult(True, min(cands, key=lambda s: s.start) if cands else None)

class OrNode(Node):
    def __init__(self, left: Node, right: Node): self.left, self.right = left, right
    def eval(self, spans_map: Dict[str, List[Span]]) -> BoolResult:
        l = self.left.eval(spans_map); r = self.right.eval(spans_map)
        if not l.matched and not r.matched: return BoolResult(False, None)
        cands = [s for s in (l.span, r.span) if s]
        return BoolResult(True, min(cands, key=lambda s: s.start) if cands else None)

class ProxNode(Node):
    def __init__(self, left: Node, right: Node, k: int): self.left, self.right, self.k = left, right, k
    def eval(self, spans_map: Dict[str, List[Span]]) -> BoolResult:
        def gather(n: Node) -> List[Span]:
            if isinstance(n, PhraseNode):
                return spans_map.get(n.phrase.lower(), [])
            br = n.eval(spans_map)
            if not br.matched:
                return []
            return _collect_phrase_spans(n, spans_map)
        L = gather(self.left); R = gather(self.right)
        best: Optional[Span] = None
        for a in L:
            for b in R:
                if abs(a.tindex - b.tindex) <= self.k:
                    cand = a if a.start <= b.start else b
                    if best is None or cand.start < best.start:
                        best = cand
        return BoolResult(best is not None, best)

def _collect_phrase_spans(node: Node, spans_map: Dict[str, List[Span]]) -> List[Span]:
    if isinstance(node, PhraseNode):
        return spans_map.get(node.phrase.lower(), [])
    res: List[Span] = []
    if isinstance(node, (AndNode, OrNode, ProxNode)):
        res.extend(_collect_phrase_spans(node.left, spans_map))
        res.extend(_collect_phrase_spans(node.right, spans_map))
    return res

def _parse_boolean(tokens: List[str]) -> Node:
    pos = 0
    def peek(): return tokens[pos] if pos < len(tokens) else None
    def eat(tok=None):
        nonlocal pos
        t = peek()
        if t is None: return None
        if tok is None or t.upper() == tok or t == tok:
            pos += 1; return t
        return None

    def parse_expr() -> Node:   # OR-level
        node = parse_term()
        while True:
            t = peek()
            if t and t.upper() == "OR":
                eat(); node = OrNode(node, parse_term())
            else:
                break
        return node

    def parse_term() -> Node:   # AND-level
        node = parse_prox()
        while True:
            t = peek()
            if t and t.upper() == "AND":
                eat(); node = AndNode(node, parse_prox())
            else:
                break
        return node

    def parse_prox() -> Node:   # Proximity-level
        node = parse_factor()
        while True:
            t = peek()
            if not t: break
            t_up = t.upper()
            if t_up.startswith("NEAR/") or t_up.startswith("WITHIN/"):
                eat()
                try:
                    k = int(t_up.split("/", 1)[1])
                except Exception:
                    k = 10
                right = parse_factor()
                node = ProxNode(node, right, k)
            else:
                break
        return node

    def parse_factor() -> Node:
        t = peek()
        if t == "(":
            eat("("); node = parse_expr(); eat(")"); return node
        if t and len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            eat(); return PhraseNode(_normalize_ws(t[1:-1]))
        if t:
            eat(); return PhraseNode(_normalize_ws(t))
        return PhraseNode("")

    return parse_expr()

def evaluate_query(text: str, query: str, whole_words: bool) -> BoolResult:
    token_starts = _build_token_index(text)
    toks = _tokenize_bool(query)
    # leaves to index
    phrases: List[str] = []
    for t in toks:
        T = t.upper()
        if t in ("(", ")") or T in ("AND", "OR") or T.startswith("NEAR/") or T.startswith("WITHIN/"):
            continue
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            phrases.append(_normalize_ws(t[1:-1]).lower())
        else:
            phrases.append(_normalize_ws(t).lower())
    unique = sorted(set(p for p in phrases if p))
    spans_map = {p: _find_all_spans(text, p, context=80, token_starts=token_starts, whole_words=whole_words) for p in unique}
    ast = _parse_boolean(toks)
    return ast.eval(spans_map)

# -----------------------
# Excel + manual inputs
# -----------------------

@dataclass
class InputData:
    companies: pd.DataFrame
    terms: List[str]

LIKELY_COMP_SHEETS = ["companies", "company", "tickers", "firms"]
LIKELY_TERM_SHEETS = ["terms", "search", "keywords"]

def load_input_excel(file_like, comp_sheet: Optional[str], term_sheet: Optional[str]) -> InputData:
    xls = pd.read_excel(file_like, sheet_name=None)

    # companies sheet
    comp_df = None
    if comp_sheet and comp_sheet in xls:
        comp_df = xls[comp_sheet]
    else:
        for cand in LIKELY_COMP_SHEETS:
            if cand in xls:
                comp_df = xls[cand]; break
    if comp_df is None:
        for _, df in xls.items():
            cols = {str(c).strip().lower() for c in df.columns if isinstance(c, str)}
            if {"ticker", "cik"} & cols:
                comp_df = df; break
    if comp_df is None:
        raise ValueError("Could not find a companies sheet (needs 'ticker' or 'cik').")
    comp_df = comp_df.copy()
    comp_df.columns = [str(c).strip().lower() for c in comp_df.columns]
    if "ticker" not in comp_df.columns and "cik" not in comp_df.columns:
        raise ValueError("Companies sheet must have 'ticker' or 'cik' column.")
    comp_df = comp_df.dropna(how="all").reset_index(drop=True)

    # terms sheet (optional; ignored in Boolean/Proximity mode)
    terms: List[str] = []
    if term_sheet and term_sheet in xls:
        term_df = xls[term_sheet].copy()
        term_df.columns = [str(c).strip().lower() for c in term_df.columns]
        term_col = "term" if "term" in term_df.columns else ("search_term" if "search_term" in term_df.columns else "keyword" if "keyword" in term_df.columns else None)
        if term_col:
            terms = [str(t).strip() for t in term_df[term_col].dropna().tolist() if str(t).strip()]
    return InputData(companies=comp_df, terms=terms)

def manual_companies_to_df(items: List[str], ticker_cache: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for raw in items:
        s = raw.strip()
        if not s: continue
        if s.isdigit():
            cik = int(s); ticker_guess, title = "", ""
            for t, rec in ticker_cache.items():
                if int(rec["cik_str"]) == cik:
                    ticker_guess = t; title = rec.get("title", ""); break
            rows.append({"ticker": ticker_guess, "cik": cik, "company": title})
        else:
            ticker = s.upper()
            rec = ticker_cache.get(ticker)
            if rec:
                rows.append({"ticker": ticker, "cik": int(rec["cik_str"]), "company": rec.get("title", "")})
    df = pd.DataFrame(rows)
    df = df[df["cik"].notna()].copy()
    if df.empty:
        raise ValueError("No manual companies resolved to CIKs.")
    df["cik"] = df["cik"].astype(int)
    return df.reset_index(drop=True)

# -----------------------
# Search runner
# -----------------------

@dataclass
class SearchConfig:
    forms_csv: str
    start_date: str
    limit_per_company: int
    rps: float
    user_agent: str
    first_match_only: bool

def run_search(
    client: SecClient,
    companies_df: pd.DataFrame,
    terms: List[str],
    cfg: SearchConfig,
    ticker_cache: Dict[str, Dict],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    boolean_or_prox_query: Optional[str] = None,
    whole_words_only: bool = True,
    normalize_text_opt: bool = True,
) -> pd.DataFrame:
    forms = [f.strip() for f in cfg.forms_csv.split(",") if f.strip()]

    cdf = companies_df.copy()
    cdf.columns = [str(c).strip().lower() for c in cdf.columns]
    if "ticker" not in cdf.columns: cdf["ticker"] = ""
    if "company" not in cdf.columns and "title" in cdf.columns: cdf["company"] = cdf["title"]
    if "company" not in cdf.columns: cdf["company"] = ""

    out_rows: List[Dict] = []
    total_steps = max(1, int(len(cdf) * max(1, cfg.limit_per_company)))
    step = 0

    def pb(step, msg):
        if progress_cb:
            progress_cb(step, total_steps, msg)

    for _, row in cdf.iterrows():
        ticker = str(row.get("ticker") or "").strip() or None
        cik = row.get("cik")
        try:
            cik = int(str(cik).split(".")[0]) if pd.notna(cik) else None
        except Exception:
            cik = None
        name = str(row.get("company") or "").strip()
        if not name and ticker and ticker in ticker_cache:
            name = ticker_cache[ticker]["title"]

        if not cik and ticker:
            cik = client.cik_from_ticker(ticker, ticker_cache)
        if not ticker and cik:
            for t, rec in ticker_cache.items():
                if int(rec["cik_str"]) == int(cik):
                    ticker = t; name = name or rec.get("title", ""); break

        if not cik:
            step += 1; pb(step, f"[{ticker or 'UNKNOWN'}] Skipping: cannot resolve CIK."); continue

        pb(step, f"[{ticker or cik}] Listing filings…")
        try:
            filings = client.list_filings(
                cik=cik, forms=forms, start_date=cfg.start_date.strip() or None,
                limit_per_company=cfg.limit_per_company
            )
        except Exception as e:
            step += 1; pb(step, f"[{ticker or cik}] ERROR listing filings: {e}"); continue

        if not filings:
            step += 1; pb(step, f"[{ticker or cik}] No filings found with filters."); continue

        for f in filings:
            acc = f.get("accessionNumber"); prim = f.get("primaryDocument")
            if not acc or not prim: continue

            try:
                url, content, mtype = client.download_primary(cik, acc, prim)
            except Exception as e:
                pb(step, f"[{ticker or cik}] ERROR downloading {acc}/{prim}: {e}"); continue

            # extract raw text
            if mtype in ("html", "txt"):
                raw_text = bytes_to_text_html_or_txt(content, is_html=(mtype == "html"))
            elif mtype == "pdf":
                raw_text = pdf_bytes_to_text(content) if PDF_OK else ""
            else:
                try: raw_text = content.decode("utf-8", errors="ignore")
                except Exception: raw_text = ""

            if not raw_text:
                pb(step, f"[{ticker or cik}] Empty/unreadable text for {acc}/{prim}"); continue

            # normalize for searching
            search_text = normalize_text_for_search(raw_text, normalize_text_opt)

            if boolean_or_prox_query and boolean_or_prox_query.strip():
                br = evaluate_query(search_text, boolean_or_prox_query, whole_words_only)
                if br.matched:
                    if br.span:
                        spos, epos = br.span.start, br.span.end
                        snippet = br.span.snippet
                        exact_for_link = search_text[spos:epos]
                        char_start_guess = spos
                    else:
                        spos, epos = 0, 0
                        snippet = search_text[:160].replace("\n", " ")
                        exact_for_link = boolean_or_prox_query
                        char_start_guess = 0
                    # Build links
                    if mtype in ("html", "txt"):
                        open_at = make_text_fragment_from_exact(url, exact_for_link)
                    elif mtype == "pdf":
                        open_at = make_pdf_search_url(url, exact_for_link, raw_text, char_start_guess)
                    else:
                        open_at = url
                    out_rows.append({
                        "company": name or "", "ticker": ticker or "", "cik": cik,
                        "form": f.get("form", ""), "filingDate": f.get("filingDate", ""),
                        "reportDate": f.get("reportDate", ""), "accessionNumber": acc,
                        "primaryDocument": prim, "doc_url": url,
                        "open_at_match": open_at, "open_doc": url,
                        "term": boolean_or_prox_query, "match_index": 1,
                        "char_start": spos, "char_end": epos, "snippet": snippet,
                    })
            else:
                # simple terms mode
                for term in terms:
                    matches = find_matches(search_text, term, context=80, whole_words=whole_words_only)
                    if not matches: continue
                    for mi, (spos, epos, snippet) in enumerate(matches, start=1):
                        exact_for_link = search_text[spos:epos]
                        if mtype in ("html", "txt"):
                            open_at = make_text_fragment_from_exact(url, exact_for_link)
                        elif mtype == "pdf":
                            open_at = make_pdf_search_url(url, exact_for_link, raw_text, spos)
                        else:
                            open_at = url
                        out_rows.append({
                            "company": name or "", "ticker": ticker or "", "cik": cik,
                            "form": f.get("form", ""), "filingDate": f.get("filingDate", ""),
                            "reportDate": f.get("reportDate", ""), "accessionNumber": acc,
                            "primaryDocument": prim, "doc_url": url,
                            "open_at_match": open_at, "open_doc": url,
                            "term": term, "match_index": mi,
                            "char_start": spos, "char_end": epos, "snippet": snippet,
                        })
                        if cfg.first_match_only: break

        step += 1; pb(step, f"[{ticker or cik}] Done batch.")

    return pd.DataFrame(out_rows)

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="SEC EDGAR Term Finder", page_icon="📄", layout="wide")

st.title("📄 SEC EDGAR Term Finder")
st.caption("Boolean AND/OR, parentheses, quoted phrases, and proximity NEAR/n (client-side over EDGAR documents). "
           "Include a User-Agent with contact info and keep RPS modest.")

with st.sidebar:
    st.subheader("Options")
    forms_csv = st.text_input("Form types (comma-separated)", "10-K,10-Q,8-K")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "")
    limit_per_company = st.number_input("Max filings per company", 1, 100, 20)
    rps = st.number_input("Requests per second (politeness)", 1.0, 10.0, 4.0, 0.5)
    user_agent = st.text_input("User-Agent (include contact email)", USER_AGENT_DEFAULT)
    first_match_only = st.checkbox("Only first match per filing & term (terms mode)", value=False)

    st.markdown("---")
    use_boolean = st.checkbox("Use Boolean / Proximity query", value=True)
    bool_query = ""
    if use_boolean:
        st.caption(
            'Use AND / OR and parentheses. Quote phrases for exact match.\n'
            'Proximity: A NEAR/20 B (alias WITHIN/20) = A and B within 20 words.\n'
            'Precedence: NEAR/WITHIN > AND > OR.'
        )
        bool_query = st.text_area(
            "Boolean / Proximity query",
            height=110,
            value='("artificial intelligence" OR "machine learning") NEAR/15 (incident OR error)'
        )

    st.markdown("---")
    whole_words_only = st.checkbox("Match whole words only", value=True)
    normalize_text_opt = st.checkbox("Normalize spaces & PDF hyphenation", value=True)

    st.caption("HTML/TXT deep links highlight text in Chrome/Edge; PDFs open with a prefilled search (page hint when available). "
               + ("✅ PDF supported." if PDF_OK else "⚠️ PDF search disabled (install pdfminer.six)."))

tab1, tab2 = st.tabs(["Quick Search (no Excel)", "From Excel"])

# Quick Search
with tab1:
    st.subheader("Quick Search")
    companies_csv = st.text_input("Companies (tickers or CIKs, comma-separated)", "AAPL,MSFT")
    if not use_boolean:
        terms_csv = st.text_input("Search terms (comma-separated)", "climate risk,cybersecurity")
    run_quick = st.button("Run Quick Search", type="primary", use_container_width=True)

# Excel
with tab2:
    st.subheader("Excel Upload")
    xfile = st.file_uploader("Upload Excel (.xlsx/.xlsm/.xls)", type=["xlsx", "xlsm", "xls"])
    c_sheet = st.text_input("Companies sheet name (optional)", "companies")
    t_sheet = st.text_input("Terms sheet name (optional; ignored in Boolean/Proximity mode)", "terms")
    run_excel = st.button("Run Excel Search", type="primary", use_container_width=True)

# Shared ticker cache + client
client = SecClient(user_agent=user_agent or USER_AGENT_DEFAULT, rps=float(rps))
try:
    ticker_cache = SecClient.ticker_table_cached(client)
except Exception as e:
    st.error(f"Failed to load SEC ticker table: {e}")
    st.stop()

def ensure_user_agent_ok(ua: str) -> bool:
    return bool(ua and "@" in ua and len(ua) > 8)

def progress_cb_factory(container):
    prog = container.progress(0, text="Starting…")
    status = container.empty()
    def cb(step: int, total: int, message: str):
        pct = int(min(100, max(0, (step / max(1, total)) * 100)))
        prog.progress(pct, text=message)
        status.write(message)
    return cb

def render_results(df: pd.DataFrame):
    if df.empty:
        st.info("No matches found for the current filters/query.")
        return
    cols = {
        "company": "Company",
        "ticker": "Ticker",
        "form": "Form",
        "filingDate": "Filing date",
        "term": "Term / Query",
        "snippet": "Snippet",
        "open_at_match": st.column_config.LinkColumn("Open at match"),
        "open_doc": st.column_config.LinkColumn("Open filing"),
    }
    st.dataframe(
        df[list(cols.keys())],
        use_container_width=True,
        hide_index=True,
        column_config=cols
    )
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="edgar_matches.csv",
        mime="text/csv",
        use_container_width=True
    )

def do_search_with_inputs(companies_df: pd.DataFrame, terms: List[str], label: str):
    if not ensure_user_agent_ok(user_agent):
        st.warning("Please include contact info (e.g., an email) in your User-Agent per SEC guidance.")
    cfg = SearchConfig(
        forms_csv=forms_csv, start_date=start_date,
        limit_per_company=int(limit_per_company), rps=float(rps),
        user_agent=user_agent.strip() or USER_AGENT_DEFAULT,
        first_match_only=bool(first_match_only),
    )
    spot = st.container()
    cb = progress_cb_factory(spot)
    with st.spinner(f"Running {label}…"):
        df = run_search(
            client, companies_df, terms, cfg, ticker_cache,
            progress_cb=cb,
            boolean_or_prox_query=(bool_query.strip() if use_boolean else None),
            whole_words_only=whole_words_only,
            normalize_text_opt=normalize_text_opt,
        )
    st.success(f"Done — {len(df)} match{'es' if len(df)!=1 else ''}.")
    render_results(df)

# Handle Quick Search
if run_quick:
    manual_companies = [x.strip() for x in (companies_csv or "").split(",") if x.strip()]
    if not manual_companies:
        st.error("Enter at least one ticker or CIK.")
    else:
        try:
            companies_df = manual_companies_to_df(manual_companies, ticker_cache)
        except Exception as e:
            st.error(f"Could not resolve manual companies: {e}")
            st.stop()
        if use_boolean:
            if not bool_query.strip():
                st.error("Enter a Boolean/Proximity query.")
            else:
                do_search_with_inputs(companies_df, [], "Quick Search (Boolean/Proximity)")
        else:
            terms = [t.strip() for t in (locals().get("terms_csv", "") or "").split(",") if t.strip()]
            if not terms:
                st.error("Enter at least one search term.")
            else:
                do_search_with_inputs(companies_df, terms, "Quick Search")

# Handle Excel
if run_excel:
    if not xfile:
        st.error("Please upload an Excel file.")
    else:
        try:
            data = load_input_excel(xfile, c_sheet.strip() or None, (t_sheet.strip() or None) if not use_boolean else None)
        except Exception as e:
            st.error(f"Excel error: {e}")
            st.stop()
        do_search_with_inputs(data.companies, data.terms if not use_boolean else [], "Excel Search")
