"""
export_notebooks_pdf.py
-----------------------
Export all project notebooks to PDF and merge into a single document.

Requirements (installed automatically if missing):
    pip install "nbconvert[webpdf]" pypdf
    playwright install chromium

Usage:
    python scripts/export_notebooks_pdf.py
    python scripts/export_notebooks_pdf.py --html   # fallback: HTML only
    python scripts/export_notebooks_pdf.py --no-merge  # individual PDFs only
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


def ensure_packages():
    """Install webpdf extra and pypdf if not present."""
    try:
        import nbconvert  # noqa: F401
        from nbconvert.exporters import WebPDFExporter  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "nbconvert[webpdf]"],
            "Installing nbconvert[webpdf]")

    try:
        import pypdf  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "pypdf"],
            "Installing pypdf")

    # Install Chromium for playwright (idempotent)
    run([sys.executable, "-m", "playwright", "install", "chromium"],
        "Installing Playwright Chromium (may take a minute on first run)")


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
        f"Converting {nb_path.name} → PDF",
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
        f"Converting {nb_path.name} → HTML",
    )
    return out_path


def merge_pdfs(pdf_paths: list[Path], out: Path):
    try:
        from pypdf import PdfMerger
    except ImportError:
        print("\n[!] pypdf not available — skipping merge. Install with: pip install pypdf")
        return

    merger = PdfMerger()
    for p in pdf_paths:
        if p and p.exists():
            merger.append(str(p))
            print(f"    + {p.name}")
        else:
            print(f"    - MISSING {p}")

    with open(out, "wb") as f:
        merger.write(f)
    print(f"\n[+] Merged PDF saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Export notebooks to PDF/HTML")
    parser.add_argument("--html", action="store_true",
                        help="Export to HTML only (no Chromium needed)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging individual PDFs into one")
    parser.add_argument("--with-code", action="store_true",
                        help="Include code cells in output (default: outputs only)")
    args = parser.parse_args()

    print("=" * 60)
    print("Arabic ITSM — Notebook Export")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    if args.html:
        # ── HTML fallback ───────────────────────────────────────────────────
        print("\nMode: HTML export (open in browser → Ctrl+P → Save as PDF)")
        for nb_name in NOTEBOOKS:
            nb_path = NB_DIR / nb_name
            if nb_path.exists():
                export_html(nb_path)
            else:
                print(f"  [!] Not found: {nb_path}")
        print(f"\nHTML files written to: {OUT_DIR}")
        print("To print: open each .html in Chrome/Edge → Ctrl+P → Destination: Save as PDF")
        return

    # ── PDF via webpdf ──────────────────────────────────────────────────────
    ensure_packages()

    pdf_paths = []
    for nb_name in NOTEBOOKS:
        nb_path = NB_DIR / nb_name
        if not nb_path.exists():
            print(f"  [!] Not found, skipping: {nb_path}")
            continue
        pdf_path = export_webpdf(nb_path)
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
