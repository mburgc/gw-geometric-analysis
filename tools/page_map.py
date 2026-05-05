import re
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    root = Path(__file__).resolve().parent
    tex = (root / "main.tex").read_text(encoding="utf-8", errors="ignore")

    pat = re.compile(r"\\(section|subsection|subsubsection)\{([^}]*)\}")
    titles: list[tuple[str, str]] = []
    for m in pat.finditer(tex):
        level, title = m.group(1), m.group(2)
        # strip common latex macros inside titles for matching
        title = re.sub(r"\\[a-zA-Z]+\b", "", title).strip()
        title = re.sub(r"\s+", " ", title)
        titles.append((level, title))

    priority = [t for lvl, t in titles if lvl in ("section", "subsection")]
    seen: set[str] = set()
    priority2: list[str] = []
    for t in priority:
        if t and t not in seen:
            seen.add(t)
            priority2.append(t)

    reader = PdfReader(str(root / "main.pdf"))
    lines: list[str] = []
    for i, p in enumerate(reader.pages, start=1):
        txt = (p.extract_text() or "").replace("\u00a0", " ")

        found = None
        for t in priority2:
            if t in txt:
                found = t
                break

        first = None
        for ln in (x.strip() for x in txt.splitlines()):
            if ln:
                first = ln
                break

        lines.append(f"{i:02d}\t{found or '-'}\t{first or '-'}")

    out = root / "page_map.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} pages)")


if __name__ == "__main__":
    main()
