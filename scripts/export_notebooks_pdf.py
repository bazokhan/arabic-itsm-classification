"""
export_notebooks_pdf.py
-----------------------
Export all project notebooks to PDF and merge into a single document.

Requirements (installed automatically if missing):
    pip install "nbconvert[webpdf]" pypdf
    playwright install chromium

Usage:
    python scripts/export_notebooks_pdf.py
    python scripts/export_notebooks_pdf.py --html-only   # HTML only
    python scripts/export_notebooks_pdf.py --webpdf-first  # try nbconvert webpdf first
    python scripts/export_notebooks_pdf.py --merge-only  # merge existing PDFs only
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "results" / "exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOKS = [
    "01_data_inspection.ipynb",
    "02_data_preparation.ipynb",
    "03_baseline_models.ipynb",
    "04_marbert_finetuning.ipynb",
    "05_evaluation_results.ipynb",
]

MERGED_PDF = OUT_DIR / "arabic_itsm_all_notebooks.pdf"


# ── Helpers ──────────────────────────────────────────────────────────────────
def run(cmd: list[str], desc: str) -> bool:
    print(f"\n[+] {desc}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()[:400]}")
        return False
    print(f"    OK")
    return True


def can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def ensure_packages():
    """Install webpdf dependencies and pypdf if not present."""
    try:
        import nbconvert  # noqa: F401
        from nbconvert.exporters import WebPDFExporter  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "nbconvert[webpdf]"],
            "Installing nbconvert[webpdf]")

    # nbconvert[webpdf] usually pulls playwright, but not always in all envs.
    playwright_ok = True
    try:
        import playwright  # noqa: F401
    except ImportError:
        playwright_ok = run(
            [sys.executable, "-m", "pip", "install", "playwright"],
            "Installing playwright Python package",
        )
        if playwright_ok:
            try:
                import playwright  # noqa: F401
            except ImportError:
                playwright_ok = False

    pypdf_ok = can_import("pypdf")
    pypdf2_ok = can_import("PyPDF2")
    if not (pypdf_ok or pypdf2_ok):
        installed = run([sys.executable, "-m", "pip", "install", "pypdf"],
                        "Installing pypdf")
        if not installed:
            # Fallback for environments where pypdf is unavailable but PyPDF2 is.
            run([sys.executable, "-m", "pip", "install", "PyPDF2"],
                "Installing PyPDF2 fallback")
        pypdf_ok = can_import("pypdf")
        pypdf2_ok = can_import("PyPDF2")
        if not (pypdf_ok or pypdf2_ok):
            print("\n[!] Could not import pypdf/PyPDF2 after install attempt. Merge may be skipped.")

    # Install Chromium for playwright (idempotent)
    if playwright_ok:
        run([sys.executable, "-m", "playwright", "install", "chromium"],
            "Installing Playwright Chromium (may take a minute on first run)")
    else:
        print("\n[!] Playwright unavailable. PDF web export may fail; HTML fallback remains available.")

    return playwright_ok


def export_webpdf(nb_path: Path) -> Path | None:
    out_path = OUT_DIR / nb_path.with_suffix(".pdf").name
    ok = run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "webpdf",
            "--output-dir", str(OUT_DIR),
            "--no-input",  # hide code cells — remove this line to keep code
            str(nb_path),
        ],
        f"Converting {nb_path.name} -> PDF",
    )
    return out_path if (ok and out_path.exists()) else None


def export_html(nb_path: Path) -> Path:
    out_path = OUT_DIR / nb_path.with_suffix(".html").name
    run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            "--output-dir", str(OUT_DIR),
            str(nb_path),
        ],
        f"Converting {nb_path.name} -> HTML",
    )
    return out_path


def html_to_pdf_with_playwright(html_path: Path, pdf_path: Path) -> bool:
    """Render HTML to PDF using Playwright if webpdf export fails."""
    script = (
        "from pathlib import Path\n"
        "from playwright.sync_api import sync_playwright\n"
        f"html = Path(r'''{html_path}''').resolve().as_uri()\n"
        f"out = Path(r'''{pdf_path}''')\n"
        "with sync_playwright() as p:\n"
        "    browser = p.chromium.launch()\n"
        "    page = browser.new_page()\n"
        "    page.goto(html, wait_until='networkidle')\n"
        "    page.pdf(path=str(out), format='A4', print_background=True)\n"
        "    browser.close()\n"
    )
    return run([sys.executable, "-c", script], f"Rendering {html_path.name} -> {pdf_path.name} via Playwright")


def merge_pdfs(pdf_paths: list[Path], out: Path):
    # New pypdf versions (e.g., 6.x) use PdfWriter/PdfReader and no PdfMerger.
    try:
        from pypdf import PdfWriter, PdfReader  # type: ignore
        writer = PdfWriter()
        added = 0
        for p in pdf_paths:
            if p and p.exists():
                reader = PdfReader(str(p))
                for page in reader.pages:
                    writer.add_page(page)
                added += 1
                print(f"    + {p.name}")
            else:
                print(f"    - MISSING {p}")
        if added == 0:
            print("\n[!] No valid PDFs to merge.")
            return
        with open(out, "wb") as f:
            writer.write(f)
        print(f"\n[+] Merged PDF saved: {out}")
        return
    except Exception:
        pass

    # Legacy fallback path (older PyPDF2)
    try:
        from PyPDF2 import PdfMerger as LegacyMerger  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfFileMerger as LegacyMerger  # type: ignore
        except Exception:
            print("\n[!] pypdf/PyPDF2 merge backend unavailable — skipping merge.")
            return

    merger = LegacyMerger()
    added = 0
    for p in pdf_paths:
        if p and p.exists():
            merger.append(str(p))
            added += 1
            print(f"    + {p.name}")
        else:
            print(f"    - MISSING {p}")
    if added == 0:
        print("\n[!] No valid PDFs to merge.")
        return
    with open(out, "wb") as f:
        merger.write(f)
    print(f"\n[+] Merged PDF saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export notebooks to PDF/HTML")
    parser.add_argument("--html-only", action="store_true",
                        help="Export to HTML only (no PDF rendering)")
    parser.add_argument("--webpdf-first", action="store_true",
                        help="Try nbconvert webpdf first; fallback to HTML->Playwright PDF")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge existing notebook PDFs in output directory and exit")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging individual PDFs into one")
    parser.add_argument("--with-code", action="store_true",
                        help="Include code cells in output (default: outputs only)")
    args = parser.parse_args()

    print("=" * 60)
    print("Arabic ITSM — Notebook Export")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    if args.merge_only:
        existing = [OUT_DIR / f"{Path(n).stem}.pdf" for n in NOTEBOOKS]
        merge_pdfs(existing, MERGED_PDF)
        print("\nDone.")
        return

    if args.html_only:
        # ── HTML fallback ───────────────────────────────────────────────────
        print("\nMode: HTML export (open in browser -> Ctrl+P -> Save as PDF)")
        for nb_name in NOTEBOOKS:
            nb_path = NB_DIR / nb_name
            if nb_path.exists():
                export_html(nb_path)
            else:
                print(f"  [!] Not found: {nb_path}")
        print(f"\nHTML files written to: {OUT_DIR}")
        print("To print: open each .html in Chrome/Edge -> Ctrl+P -> Destination: Save as PDF")
        return

    # ── PDF via webpdf ──────────────────────────────────────────────────────
    playwright_available = ensure_packages()

    pdf_paths = []
    for nb_name in NOTEBOOKS:
        nb_path = NB_DIR / nb_name
        if not nb_path.exists():
            print(f"  [!] Not found, skipping: {nb_path}")
            continue
        pdf_path = None
        if args.webpdf_first:
            pdf_path = export_webpdf(nb_path)
        if pdf_path is None:
            html_path = export_html(nb_path)
            if playwright_available:
                fallback_pdf = OUT_DIR / nb_path.with_suffix(".pdf").name
                if html_path.exists() and html_to_pdf_with_playwright(html_path, fallback_pdf) and fallback_pdf.exists():
                    pdf_path = fallback_pdf
        pdf_paths.append(pdf_path)

    if not args.no_merge:
        valid = [p for p in pdf_paths if p and p.exists()]
        if valid:
            merge_pdfs(valid, MERGED_PDF)
        else:
            print("\n[!] No PDFs produced — check errors above.")

    print("\nDone.")
    print(f"Individual PDFs : {OUT_DIR}/")
    if not args.no_merge:
        print(f"Merged PDF      : {MERGED_PDF}")


if __name__ == "__main__":
    main()
