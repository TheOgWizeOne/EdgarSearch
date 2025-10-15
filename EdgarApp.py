# app.py — SEC EDGAR Term Finder with Regex + AND/Proximity + Deep Links
import io
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from urllib.parse import quote

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------- Optional PDF support --------
PDF_OK = False
try:
    from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
    PDF_OK = True
except Exception:
    PDF_OK = False

# -------- SEC client --------
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

        recent = sub.get("filings", {}).get("recent", {})
        harvest(recent)
        for f in sub.get("filings", {}).get("files", []):
            idx = self._fetch_index_file(f["name"])
            hist = idx.get("filings", {}).get("recent", {})
            harvest(hist)

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

    def download_primary(self, cik: int, accession_number: str, primary_document: str):
        url = self.primary_url(cik, accession_number, primary_document)
        r = self._get(url, stream=True)
        content = r.content
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

# -------- Text extraction & links --------
def html_txt_bytes_to_text(b: bytes, is_html: bool) -> str:
    try:
        txt = b.decode("utf-8", errors="ignore")
    except Exception:
        txt = b.decode("latin-1", errors="ignore")
    if is_html:
        soup = BeautifulSoup(txt, "html.parser")
        for t in soup(["script", "style"]):
            t.extract()
        return soup.get_text(separator=" ")
    return txt

def pdf_bytes_to_text(b: bytes) -> str:
    if not PDF_OK:
        return ""
    with io.BytesIO(b) as bio:
        try:
            return pdf_extract_text(bio) or ""
        except Exception:
            return ""

def make_text_fragment_url(base_url: str, full_text: str, start: int, end: int) -> str:
    exact = " ".join(full_text[start:end].split())
    prefix = " ".join(full_text[max(0, start-30):start].split())
    suffix = " ".join(full_text[end:end+30].split())
    frag = f"#:~:text={quote(prefix, safe='')}-," \
           f"{quote(exact, safe='')}," \
           f"-{quote(suffix, safe='')}"
    return f"{base_url}{frag}"

def make_pdf_search_url(base_url: str, term_for_searchbox: str, full_text: str, start: int) -> str:
    page = None
    try:
        page = full_text[:start].count("\x0c") + 1  # pdfminer page breaks
    except Exception:
        pass
    if page and page > 0:
        return f"{base_url}#page={page}&search={quote(term_for_searchbox)}"
    return f"{base_url}#search={quote(term_for_searchbox)}"

# -------- Matching helpers (Literal / Regex / AND+Proximity) --------
def matches_literal(text: str, term: str, context_chars: int = 80) -> List[Tuple[int, int, str, str]]:
    out = []
    if not text or not term:
        return out
    pat = re.compile(re.escape(term), re.IGNORECASE)
    for m in pat.finditer(text):
        s, e = m.start(), m.end()
        left, right = max(0, s-context_chars), min(len(text), e+context_chars)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ")
        out.append((s, e, snippet, term))
    return out

def matches_regex(text: str, pattern: str, context_chars: int = 80, case_insensitive: bool = True, dotall: bool = True) -> List[Tuple[int, int, str, str]]:
    out = []
    if not text or not pattern:
        return out
    flags = 0
    if case_insensitive: flags |= re.IGNORECASE
    if dotall: flags |= re.DOTALL
    rx = re.compile(pattern, flags)
    for m in rx.finditer(text):
        s, e = m.start(), m.end()
        left, right = max(0, s-context_chars), min(len(text), e+context_chars)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ")
        out.append((s, e, snippet, m.group(0)))
    return out

def tokenize_words_with_spans(text: str) -> Tuple[List[str], List[Tuple[int,int]]]:
    # Simple word tokenizer: sequences of letters/digits/underscore
    spans = []
    words = []
    for m in re.finditer(r"\b\w+\b", text):
        spans.append((m.start(), m.end()))
        words.append(text[m.start():m.end()])
    return words, spans

def find_phrase_occurrences(text_lower: str, phrase_lower: str) -> List[int]:
    """Return start char indices for every occurrence of 'phrase_lower' in 'text_lower'."""
    out = []
    start = 0
    while True:
        i = text_lower.find(phrase_lower, start)
        if i == -1:
            break
        out.append(i)
        start = i + 1
    return out

def char_index_to_word_index(char_index: int, word_spans: List[Tuple[int,int]]) -> int:
    # Binary search for the word whose start is <= char_index < end (or nearest following)
    lo, hi = 0, len(word_spans)-1
    best = 0
    while lo <= hi:
        mid = (lo+hi)//2
        s, e = word_spans[mid]
        if s <= char_index < e:
            return mid
        if char_index < s:
            hi = mid - 1
        else:
            best = mid
            lo = mid + 1
    return best

def matches_and_proximity(text: str, terms: List[str], window_words: int = 40, context_chars: int = 100) -> List[Tuple[int,int,str,str]]:
    """
    Return spans where ALL terms occur within a window of <= window_words.
    We locate a minimal covering window across the token sequence, then project to char spans.
    """
    out = []
    if not text or not terms:
        return out
    terms_l = [t.strip().lower() for t in terms if t.strip()]
    if not terms_l:
        return out

    text_l = text.lower()
    words, spans = tokenize_words_with_spans(text_l)
    if not words:
        return out

    # For each term/phrase, map to list of word positions where the phrase starts
    term_pos_lists: List[List[int]] = []
    for t in terms_l:
        char_starts = find_phrase_occurrences(text_l, t)
        if not char_starts:
            return out  # if any term missing, no AND match
        wpos = [char_index_to_word_index(cs, spans) for cs in char_starts]
        term_pos_lists.append(wpos)

    # Multi-pointer sweep to find minimal windows covering all terms
    # Flatten (pos, term_id) pairs
    pairs = []
    for tid, lst in enumerate(term_pos_lists):
        pairs += [(p, tid) for p in lst]
    pairs.sort()

    from collections import defaultdict, deque
    need = len(term_pos_lists)
    count: Dict[int,int] = defaultdict(int)
    have = 0
    q = deque()  # sliding window over pairs
    best_windows: List[Tuple[int,int]] = []

    for p, tid in pairs:
        q.append((p, tid))
        if count[tid] == 0:
            have += 1
        count[tid] += 1
        # shrink from left while we still cover all terms
        while q and have == need:
            left_p, left_tid = q[0]
            right_p = q[-1][0]
            if right_p - left_p <= max(1, window_words):
                best_windows.append((left_p, right_p))
            # pop left
            count[left_tid] -= 1
            if count[left_tid] == 0:
                have -= 1
            q.popleft()

    # Turn each window into a char span + snippet
    for wp_l, wp_r in best_windows:
        s_char = spans[wp_l][0]
        e_char = spans[min(wp_r, len(spans)-1)][1]
        left, right = max(0, s_char - context_chars), min(len(text), e_char + context_chars)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ")
        label = " AND ".join(terms) + f" (≤{window_words} words)"
        out.append((s_char, e_char, snippet, label))
    return out

# -------- Excel + manual helpers --------
@dataclass
class InputData:
    companies: pd.DataFrame
    terms: List[str]

LIKELY_COMP_SHEETS = ["companies", "company", "tickers", "firms"]
LIKELY_TERM_SHEETS = ["terms", "search", "keywords"]

def load_input_excel(file_like, comp_sheet: Optional[str], term_sheet: Optional[str]) -> InputData:
    xls = pd.read_excel(file_like, sheet_name=None)
    comp_df = None
    if comp_sheet and comp_sheet in xls:
        comp_df = xls[comp_sheet]
    else:
        for cand in LIKELY_COMP_SHEETS:
            if cand in xls: comp_df = xls[cand]; break
    if comp_df is None:
        for _, df in xls.items():
            cols = {str(c).strip().lower() for c in df.columns if isinstance(c, str)}
            if {"ticker", "cik"} & cols: comp_df = df; break
    if comp_df is None:
        raise ValueError("Could not find a companies sheet (needs 'ticker' or 'cik').")
    comp_df = comp_df.copy()
    comp_df.columns = [str(c).strip().lower() for c in comp_df.columns]
    if "ticker" not in comp_df.columns and "cik" not in comp_df.columns:
        raise ValueError("Companies sheet must have 'ticker' or 'cik' column.")
    comp_df = comp_df.dropna(how="all").reset_index(drop=True)

    term_df = None
    if term_sheet and term_sheet in xls:
        term_df = xls[term_sheet]
    else:
        for cand in LIKELY_TERM_SHEETS:
            if cand in xls: term_df = xls[cand]; break
    if term_df is None:
        for _, df in xls.items():
            cols = {str(c).strip().lower() for c in df.columns if isinstance(c, str)}
            if {"term", "search_term", "keyword"} & cols: term_df = df; break
    if term_df is None:
        raise ValueError("Could not find a terms sheet (needs 'term' or 'search_term' or 'keyword').")
    term_df = term_df.copy()
    term_df.columns = [str(c).strip().lower() for c in term_df.columns]
    term_col = "term" if "term" in term_df.columns else ("search_term" if "search_term" in term_df.columns else "keyword")
    terms = [str(t).strip() for t in term_df[term_col].dropna().tolist() if str(t).strip()]
    if not terms:
        raise ValueError("No search terms found in the terms sheet.")
    return InputData(companies=comp_df, terms=terms)

def manual_companies_to_df(items: List[str], ticker_cache: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for raw in items:
        s = raw.strip()
        if not s: continue
        if s.isdigit():
            cik = int(s)
            ticker_guess, title = "", ""
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

# -------- Runner --------
@dataclass
class SearchConfig:
    forms_csv: str
    start_date: str
    limit_per_company: int
    rps: float
    user_agent: str
    first_match_only: bool
    mode: str  # 'literal' | 'regex' | 'andprox'
    regex_pattern: str
    and_terms: List[str]
    prox_window_words: int
    case_insensitive: bool
    regex_dotall: bool

def run_search(
    client: SecClient,
    companies_df: pd.DataFrame,
    # For literal/regex: a single "terms" list; For AND/Prox: use cfg.and_terms
    terms_for_literal_or_regex: List[str],
    cfg: SearchConfig,
    ticker_cache: Dict[str, Dict],
    progress_cb: Optional[Callable[[int, int, str], None]] = None
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
        if progress_cb: progress_cb(step, total_steps, msg)

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
                cik=cik,
                forms=forms,
                start_date=cfg.start_date.strip() or None,
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
                pb(step, f"[{ticker or cik}] ERROR downloading {acc}/{prim}: {e}")
                continue

            # Extract text
            if mtype in ("html", "txt"):
                text = html_txt_bytes_to_text(content, is_html=(mtype == "html"))
            elif mtype == "pdf":
                text = pdf_bytes_to_text(content) if PDF_OK else ""
            else:
                try: text = content.decode("utf-8", errors="ignore")
                except Exception: text = ""

            if not text:
                pb(step, f"[{ticker or cik}] Empty/unreadable text for {acc}/{prim}")
                continue

            # Run chosen matcher
            doc_matches: List[Tuple[int,int,str,str]] = []
            if cfg.mode == "literal":
                for term in terms_for_literal_or_regex:
                    doc_matches += matches_literal(text, term, context_chars=100)
            elif cfg.mode == "regex":
                # Single pattern; if user pasted multiple, we OR them with |
                pattern = cfg.regex_pattern.strip()
                if not pattern and terms_for_literal_or_regex:
                    pattern = "|".join(re.escape(t) for t in terms_for_literal_or_regex)
                doc_matches = matches_regex(
                    text, pattern,
                    context_chars=120,
                    case_insensitive=cfg.case_insensitive,
                    dotall=cfg.regex_dotall
                )
            else:  # AND + Proximity
                doc_matches = matches_and_proximity(
                    text, cfg.and_terms,
                    window_words=max(1, int(cfg.prox_window_words)),
                    context_chars=120
                )

            if not doc_matches:  # continue to next filing
                continue

            # Emit rows (respect first_match_only)
            emitted = 0
            for mi, (spos, epos, snip, label) in enumerate(doc_matches, start=1):
                # Build deep links
                if mtype in ("html", "txt"):
                    open_at = make_text_fragment_url(url, text, spos, epos)
                elif mtype == "pdf":
                    # For PDFs we prefill the search with the best available label (exact/regex match text)
                    open_at = make_pdf_search_url(url, label, text, spos)
                else:
                    open_at = url

                out_rows.append({
                    "company": name or "",
                    "ticker": ticker or "",
                    "cik": cik,
                    "form": f.get("form", ""),
                    "filingDate": f.get("filingDate", ""),
                    "reportDate": f.get("reportDate", ""),
                    "accessionNumber": acc,
                    "primaryDocument": prim,
                    "doc_url": url,
                    "open_at_match": open_at,
                    "open_doc": url,
                    "query_mode": cfg.mode,
                    "term_or_pattern": label if cfg.mode != "literal" else label,  # literal term OR matched text/label
                    "match_index": mi,
                    "char_start": spos,
                    "char_end": epos,
                    "snippet": snip,
                })
                emitted += 1
                if cfg.first_match_only and emitted >= 1:
                    break

        step += 1
        pb(step, f"[{ticker or cik}] Done batch.")

    return pd.DataFrame(out_rows)

# -------- UI --------
st.set_page_config(page_title="SEC EDGAR Term Finder", page_icon="📄", layout="wide")
st.title("📄 SEC EDGAR Term Finder")
st.caption("Search primary documents in EDGAR filings. Include a descriptive User-Agent with contact info and keep rates modest (SEC fair access).")

with st.sidebar:
    st.subheader("Options")
    forms_csv = st.text_input("Form types (comma-separated)", "10-K,10-Q,8-K")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "")
    limit_per_company = st.number_input("Max filings per company", 1, 100, 5)
    rps = st.number_input("Requests per second (politeness)", 1.0, 10.0, 4.0, 0.5)
    user_agent = st.text_input("User-Agent (include contact email)", USER_AGENT_DEFAULT)
    first_match_only = st.checkbox("Only first match per filing & term", value=False)

    st.markdown("---")
    st.subheader("Query mode")
    mode = st.radio("Choose", ["Exact (literal)", "Regex", "Advanced (AND + proximity)"], index=0)

    case_ins = True
    dotall = True
    regex_pattern = ""
    and_terms_list: List[str] = []
    prox_window_words = 40

    if mode == "Regex":
        regex_pattern = st.text_input("Regex pattern (Python `re`)", "")
        case_ins = st.checkbox("Case-insensitive", value=True)
        dotall = st.checkbox("Dot matches newline (DOTALL)", value=True)
        st.caption("Tip: Leave pattern empty and also fill 'Search terms' to OR them automatically.")
    elif mode == "Advanced (AND + proximity)":
        prox_window_words = st.number_input("Proximity window (words)", 1, 400, 40)

tab1, tab2 = st.tabs(["Quick Search (no Excel)", "From Excel"])

with tab1:
    st.subheader("Quick Search")
    companies_csv = st.text_input("Companies (tickers or CIKs, comma-separated)", "AAPL,MSFT")
    if mode == "Exact (literal)" or mode == "Regex":
        terms_csv = st.text_input("Search terms (comma-separated)", "climate risk,cybersecurity")
    else:
        terms_csv = st.text_input("AND terms (comma-separated; all must appear)", "ai incident,model failure")
    run_quick = st.button("Run Quick Search", type="primary", use_container_width=True)

with tab2:
    st.subheader("Excel Upload")
    xfile = st.file_uploader("Upload Excel (.xlsx/.xlsm/.xls)", type=["xlsx", "xlsm", "xls"])
    c_sheet = st.text_input("Companies sheet name (optional)", "companies")
    t_sheet = st.text_input("Terms sheet name (optional)", "terms")
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
        st.info("No matches found for the current filters/terms.")
        return
    cols = {
        "company": "Company",
        "ticker": "Ticker",
        "form": "Form",
        "filingDate": "Filing date",
        "query_mode": "Mode",
        "term_or_pattern": "Matched term/pattern",
        "snippet": "Snippet",
        "open_at_match": st.column_config.LinkColumn("Open at match"),
        "open_doc": st.column_config.LinkColumn("Open filing"),
    }
    st.dataframe(df[list(cols.keys())], use_container_width=True, hide_index=True, column_config=cols)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="edgar_matches.csv",
        mime="text/csv",
        use_container_width=True
    )

def do_search(companies_df: pd.DataFrame, raw_terms_csv: str, trigger: str):
    if not ensure_user_agent_ok(user_agent):
        st.warning("Please include contact info (e.g., an email) in your User-Agent per SEC guidance.")

    # Parse terms field according to mode
    terms_list = [t.strip() for t in (raw_terms_csv or "").split(",") if t.strip()]
    and_terms = terms_list if mode == "Advanced (AND + proximity)" else []
    cfg = SearchConfig(
        forms_csv=forms_csv,
        start_date=start_date,
        limit_per_company=int(limit_per_company),
        rps=float(rps),
        user_agent=user_agent.strip() or USER_AGENT_DEFAULT,
        first_match_only=bool(first_match_only),
        mode=("literal" if mode == "Exact (literal)" else "regex" if mode == "Regex" else "andprox"),
        regex_pattern=regex_pattern,
        and_terms=and_terms,
        prox_window_words=int(prox_window_words),
        case_insensitive=case_ins,
        regex_dotall=dotall,
    )

    # For literal/regex we pass the list; for AND we pass an empty list (handled via cfg.and_terms)
    pass_terms = [] if cfg.mode == "andprox" else terms_list

    spot = st.container()
    cb = progress_cb_factory(spot)
    with st.spinner(f"Running {trigger}…"):
        df = run_search(client, companies_df, pass_terms, cfg, ticker_cache, progress_cb=cb)
    st.success(f"Done — {len(df)} matches.")
    render_results(df)

# Handle actions
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
        if mode == "Advanced (AND + proximity)":
            if not terms_csv.strip():
                st.error("Enter AND terms (comma-separated).")
            else:
                do_search(companies_df, terms_csv, "Quick Search")
        else:
            if not terms_csv.strip() and not regex_pattern.strip():
                st.error("Enter search terms or a regex pattern.")
            else:
                do_search(companies_df, terms_csv, "Quick Search")

if run_excel:
    if not xfile:
        st.error("Please upload an Excel file.")
    else:
        try:
            data = load_input_excel(xfile, c_sheet.strip() or None, t_sheet.strip() or None)
        except Exception as e:
            st.error(f"Excel error: {e}")
            st.stop()
        # Excel provides a list of terms (one per row) -> join for UI path
        excel_terms_csv = ",".join(data.terms)
        do_search(data.companies, excel_terms_csv, "Excel Search")
