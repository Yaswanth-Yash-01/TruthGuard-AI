"""
Generate TruthGuard_AI_Documentation.pdf from the HTML source.
Run from inside the truthguard/ directory:

    source venv/bin/activate
    pip install weasyprint
    python3 docs/generate_pdf.py
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "TruthGuard_AI_Documentation.html")
PDF  = os.path.join(HERE, "TruthGuard_AI_Documentation.pdf")


def via_weasyprint():
    from weasyprint import HTML as WP
    print("Generating PDF with weasyprint ...")
    WP(filename=HTML).write_pdf(PDF)
    print(f"Saved → {PDF}")


def via_chrome():
    """Headless Chrome / Chromium fallback."""
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome", "chromium", "chromium-browser",
    ]
    for exe in candidates:
        try:
            subprocess.run(
                [exe, "--headless", "--disable-gpu",
                 f"--print-to-pdf={PDF}", HTML],
                check=True, capture_output=True,
            )
            print(f"Saved → {PDF}")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def main():
    if not os.path.exists(HTML):
        print(f"ERROR: {HTML} not found.")
        sys.exit(1)

    # Try weasyprint first
    try:
        via_weasyprint()
        return
    except ImportError:
        print("weasyprint not installed — trying headless Chrome ...")
    except Exception as e:
        print(f"weasyprint failed ({e}) — trying headless Chrome ...")

    # Try headless Chrome
    if via_chrome():
        return

    # Final fallback: open in browser for manual print
    print("\nCould not auto-generate PDF.")
    print("Open the documentation in your browser and use File → Print → Save as PDF:")
    print(f"  open \"{HTML}\"")
    print("\nOr install weasyprint:")
    print("  pip install weasyprint")


if __name__ == "__main__":
    main()
