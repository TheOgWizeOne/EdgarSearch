# app.py
import io
import re
import csv
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional PDF support
PDF_OK = False
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
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
        Cacheable method (note: takes dummy self for Streamlit cache).
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
        # e.g., "CIK0000320193-index.json"
        url = f"{BASE}/submissions/{name}"
        return self._get(url).json()

    def list_filings(self, cik: int, forms: List[str], start_date: Optional[str], limit_per_company: int) -> List[Dict]:
        """
        Combine recent + historical index. Filter by forms, start_date. Sort newest first. Truncate to limit.
        """
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
        Returns (url, content_bytes, media_type_guess).
        media_type_guess: 'html' | 'txt' | 'pdf' | 'bin'
        """
        url = self.primary_url(cik, accession_number, primary_document)
        r = self._get(url, stream=True)
        content = r.content
        # guess type by extension + sniff
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
# Text extraction & search
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

def find_matches(text: str, term: str, context: int = 80) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start, end, snippet) for case-insensitive literal matches.
    """
    matches = []
    if not text or not term:
        return matches
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    for m in pattern.finditer(text):
        s, e = m.start(), m.end()
        left = max(0, s - context)
        right = min(len(text), e + context)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ")
        matches.append((s, e, snippet))
    return matches

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
                comp_df = xls[cand]
                break
    if comp_df is None:
        for name, df in xls.items():
            cols = {str(c).strip().lower() for c in df.columns if isinstance(c, str)}
            if {"ticker", "cik"} & cols:
                comp_df = df
                break
    if comp_df is None:
        raise ValueError("Could not find a companies sheet (needs 'ticker' or 'cik').")

    comp_df = comp_df.copy()
    comp_df.columns = [str(c).strip().lower() for c in comp_df.columns]
    if "ticker" not in comp_df.columns and "cik" not in comp_df.columns:
        raise ValueError("Companies sheet must have 'ticker' or 'cik' column.")
    comp_df = comp_df.dropna(how="all").reset_index(drop=True)

    # terms sheet
    term_df = None
    if term_sheet and term_sheet in xls:
        term_df = xls[term_sheet]
    else:
        for cand in LIKELY_TERM_SHEETS:
            if cand in xls:
                term_df = xls[cand]
                break
    if term_df is None:
        for name, df in xls.items():
            cols = {str(c).strip().lower() for c in df.columns if isinstance(c, str)}
            if {"term", "search_term", "keyword"} & cols:
                term_df = df
                break
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
    """
    Accepts tickers or CIKs; returns DataFrame with columns: ticker, cik, company.
    """
    rows = []
    for raw in items:
        s = raw.strip()
        if not s:
            continue
        if s.isdigit():
            cik = int(s)
            ticker_guess, title = "", ""
            for t, rec in ticker_cache.items():
                if int(rec["cik_str"]) == cik:
                    ticker_guess = t
                    title = rec.get("title", "")
                    break
            rows.append({"ticker": ticker_guess, "cik": cik, "company": title})
        else:
            ticker = s.upper()
            rec = ticker_cache.get(ticker)
            if rec:
                rows.append({"ticker": ticker, "cik": int(rec["cik_str"]), "company": rec.get("title", "")})
            else:
                rows.append({"ticker": ticker, "cik": None, "company": ""})
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
    progress_cb: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Returns a DataFrame of matches (one row per match).
    progress_cb(step, total, message) can be provided to update UI.
    """
    forms = [f.strip() for f in cfg.forms_csv.split(",") if f.strip()]

    # normalize companies
    cdf = companies_df.copy()
    cdf.columns = [str(c).strip().lower() for c in cdf.columns]
    if "ticker" not in cdf.columns:
        cdf["ticker"] = ""
    if "company" not in cdf.columns and "title" in cdf.columns:
        cdf["company"] = cdf["title"]
    if "company" not in cdf.columns:
        cdf["company"] = ""

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
                    ticker = t
                    name = name or rec.get("title", "")
                    break

        if not cik:
            step += 1
            pb(step, f"[{ticker or 'UNKNOWN'}] Skipping: cannot resolve CIK.")
            continue

        pb(step, f"[{ticker or cik}] Listing filings…")
        try:
            filings = client.list_filings(
                cik=cik,
                forms=forms,
                start_date=cfg.start_date.strip() or None,
                limit_per_company=cfg.limit_per_company
            )
        except Exception as e:
            step += 1
            pb(step, f"[{ticker or cik}] ERROR listing filings: {e}")
            continue

        if not filings:
            step += 1
            pb(step, f"[{ticker or cik}] No filings found with filters.")
            continue

        for f in filings:
            acc = f.get("accessionNumber")
            prim = f.get("primaryDocument")
            if not acc or not prim:
                continue

            try:
                url, content, mtype = client.download_primary(cik, acc, prim)
            except Exception as e:
                pb(step, f"[{ticker or cik}] ERROR downloading {acc}/{prim}: {e}")
                continue

            # extract text
            if mtype in ("html", "txt"):
                text = bytes_to_text_html_or_txt(content, is_html=(mtype == "html"))
            elif mtype == "pdf":
                text = pdf_bytes_to_text(content) if PDF_OK else ""
            else:
                try:
                    text = content.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""

            if not text:
                pb(step, f"[{ticker or cik}] Empty/unreadable text for {acc}/{prim}")
                continue

            for term in terms:
                matches = find_matches(text, term, context=80)
                if not matches:
                    continue
                for mi, (spos, epos, snip) in enumerate(matches, start=1):
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
                        "term": term,
                        "match_index": mi,
                        "char_start": spos,
                        "char_end": epos,
                        "snippet": snip,
                    })
                    if cfg.first_match_only:
                        break

        step += 1
        pb(step, f"[{ticker or cik}] Done batch.")

    return pd.DataFrame(out_rows)

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="SEC EDGAR Term Finder", page_icon="📄", layout="wide")

st.title("📄 SEC EDGAR Term Finder")
st.caption("Search primary documents in recent/historical filings via the public data.sec.gov endpoints. "
           "Use a descriptive User-Agent with contact info and reasonable rate limits.")

with st.sidebar:
    st.subheader("Options")
    forms_csv = st.text_input("Form types (comma-separated)", "10-K,10-Q,8-K")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "")
    limit_per_company = st.number_input("Max filings per company", 1, 100, 5)
    rps = st.number_input("Requests per second (politeness)", 1.0, 10.0, 4.0, 0.5)
    user_agent = st.text_input("User-Agent (include contact email)", USER_AGENT_DEFAULT)
    first_match_only = st.checkbox("Only first match per filing & term", value=False)
    pdf_note = "✅ PDF supported" if PDF_OK else "⚠️ PDF search disabled (install pdfminer.six)"
    st.caption(pdf_note)

tab1, tab2 = st.tabs(["Quick Search (no Excel)", "From Excel"])

# --- Quick Search ---
with tab1:
    st.subheader("Quick Search")
    companies_csv = st.text_input("Companies (tickers or CIKs, comma-separated)", "AAPL,MSFT")
    terms_csv = st.text_input("Search terms (comma-separated)", "climate risk,cybersecurity")

    run_quick = st.button("Run Quick Search", type="primary", use_container_width=True)

# --- From Excel ---
with tab2:
    st.subheader("Excel Upload")
    xfile = st.file_uploader("Upload Excel (.xlsx/.xlsm/.xls)", type=["xlsx", "xlsm", "xls"])
    c_sheet = st.text_input("Companies sheet name (optional)", "companies")
    t_sheet = st.text_input("Terms sheet name (optional)", "terms")
    run_excel = st.button("Run Excel Search", type="primary", use_container_width=True)

# Shared ticker cache + client
client = SecClient(user_agent=user_agent or USER_AGENT_DEFAULT, rps=float(rps))
try:
    ticker_cache = SecClient.ticker_table_cached(client)  # cached
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

def do_search_with_inputs(companies_df: pd.DataFrame, terms: List[str], trigger_label: str):
    if not ensure_user_agent_ok(user_agent):
        st.warning("Please include contact info (e.g., an email) in your User-Agent per SEC guidance.")
    cfg = SearchConfig(
        forms_csv=forms_csv,
        start_date=start_date,
        limit_per_company=int(limit_per_company),
        rps=float(rps),
        user_agent=user_agent.strip() or USER_AGENT_DEFAULT,
        first_match_only=bool(first_match_only),
    )

    spot = st.container()
    cb = progress_cb_factory(spot)
    with st.spinner(f"Running {trigger_label}…"):
        df = run_search(client, companies_df, terms, cfg, ticker_cache, progress_cb=cb)

    st.success(f"Done — {len(df)} matches.")
    if len(df):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="edgar_matches.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No matches found for the current filters/terms.")

# Handle Quick Search action
if run_quick:
    # Parse companies
    manual_companies = [x.strip() for x in (companies_csv or "").split(",") if x.strip()]
    if not manual_companies:
        st.error("Enter at least one ticker or CIK.")
    else:
        try:
            companies_df = manual_companies_to_df(manual_companies, ticker_cache)
        except Exception as e:
            st.error(f"Could not resolve manual companies: {e}")
            st.stop()

        # Parse terms
        terms = [t.strip() for t in (terms_csv or "").split(",") if t.strip()]
        if not terms:
            st.error("Enter at least one search term.")
        else:
            do_search_with_inputs(companies_df, terms, "Quick Search")

# Handle Excel action
if run_excel:
    if not xfile:
        st.error("Please upload an Excel file.")
    else:
        try:
            data = load_input_excel(xfile, c_sheet.strip() or None, t_sheet.strip() or None)
        except Exception as e:
            st.error(f"Excel error: {e}")
            st.stop()
        do_search_with_inputs(data.companies, data.terms, "Excel Search")
