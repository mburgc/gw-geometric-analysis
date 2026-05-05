from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

WORKSPACE = Path(__file__).resolve().parents[1]

LATEX_FILE = WORKSPACE / "main.tex"
OUT_MD = WORKSPACE / "paper_correlation_map.md"

SCRIPT_FILES = [
    WORKSPACE / "pipelineGW.py",
    WORKSPACE / "3DFullVisualization.py",
    WORKSPACE / "waveforms.py",
    WORKSPACE / "ProjectedEighenValues.py",
    WORKSPACE / "page_map.py",
    WORKSPACE / "tools" / "pdf_page_headings.py",
]

FIG_ENV_RE = re.compile(r"\\begin\{figure\}")
END_FIG_ENV_RE = re.compile(r"\\end\{figure\}")
INCLUDE_RE = re.compile(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}|\\includegraphics\{([^}]+)\}")
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
CAPTION_RE = re.compile(r"\\caption\{")

SECTION_RE = re.compile(r"^\\(section|subsection|subsubsection|paragraph)\{(.+?)\}")

MATH_BEGIN_RE = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}")
MATH_END_RE = re.compile(r"\\end\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}")


@dataclass
class Block:
    kind: str  # "figure" or "math"
    env: str
    start_line: int
    end_line: int
    heading: str
    label: Optional[str]
    files: list[str]
    caption: Optional[str]
    body: str


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def normalize_heading(level: str, title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip()
    return f"{level}: {title}" if level else title


def extract_caption(lines: list[str], start_idx: int) -> Optional[str]:
    r"""Extract a \caption{...} content starting on start_idx line where '\caption{' begins.

    Very small brace-matching parser; returns one-line normalized caption.
    """
    text = "\n".join(lines[start_idx: start_idx + 20])
    pos = text.find("\\caption{")
    if pos == -1:
        return None
    i = pos + len("\\caption{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    caption = "".join(out)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption if caption else None


@dataclass(frozen=True)
class CodeHit:
    line_no: int
    line: str
    score: int


@dataclass
class ScriptIndex:
    rel_path: str
    lines: list[str]

    def find_hits(self, patterns: list[str], *, is_regex: bool = False) -> list[CodeHit]:
        hits: list[CodeHit] = []
        for i, raw in enumerate(self.lines, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            matched = False
            if is_regex:
                matched = any(re.search(p, line) for p in patterns)
            else:
                matched = any(p in line for p in patterns)
            if not matched:
                continue

            stripped = line.lstrip()
            score = 0
            if stripped.startswith("#"):
                score -= 4
            if stripped.startswith("import ") or stripped.startswith("from "):
                score -= 5
            if "def " in line:
                score += 4
            if "=" in line and "==" not in line:
                score += 2
            if "(" in line and ")" in line:
                score += 1

            # Prefer lines that look like the core implementation rather than plotting/printing.
            strong_tokens = [
                "Cnet",
                "Znet",
                "P_perp",
                "C_proj",
                "eta_proj",
                "Dmt",
                "reconstruct_k_hat",
                "geo_to_ecef",
                "np.linalg.lstsq",
            ]
            if any(tok in line for tok in strong_tokens):
                score += 3

            hits.append(CodeHit(line_no=i, line=stripped, score=score))

        hits.sort(key=lambda h: (-h.score, h.line_no))
        return hits


def build_script_indices() -> list[ScriptIndex]:
    indices: list[ScriptIndex] = []
    for script in SCRIPT_FILES:
        if not script.exists():
            continue
        indices.append(
            ScriptIndex(rel_path=str(script.relative_to(WORKSPACE)).replace("\\", "/"), lines=read_lines(script))
        )
    return indices


def find_script_refs(
    indices: list[ScriptIndex],
    patterns: list[str],
    *,
    is_regex: bool = False,
    max_hits_per_file: int = 4,
) -> dict[str, list[CodeHit]]:
    refs: dict[str, list[CodeHit]] = {}
    for idx in indices:
        hits = idx.find_hits(patterns, is_regex=is_regex)
        if hits:
            refs[idx.rel_path] = hits[:max_hits_per_file]
    return refs


def classify_equation(body: str, heading: str = "") -> list[str]:
    tags: list[str] = []
    checks = [
        (r"C_{\\mathrm\{net\}}|\\mathbf\{C\}_\{\\mathrm\{net\}\}", "C_net"),
        (r"C_{\\mathrm\{det\}}|\\mathbf\{C\}_\{\\mathrm\{det\}\}", "C_det"),
        (r"C_{\\mathrm\{proj\}}|\\mathbf\{C\}_\{\\mathrm\{proj\}\}", "C_proj"),
        (r"P_\\perp|\\mathbf\{P\}_\\perp", "P_perp"),
        (r"eta_{\\mathrm\{proj\}}|\\eta_\{\\mathrm\{proj\}\}", "eta_proj"),
        (r"D_d\(|Dmt|\\bar\{D\}", "markers"),
        (r"\\Delta t|delay|\\Delta\s*phi|\\delta\s*\\phi", "phase_delay"),
        (r"\\hat\{\\mathbf\{k\}\}|\\mathbf\{k\}\^\{\(p\)\}", "direction_fit"),
        (r"STFT|X_d\(|\\mathbf\{Z\}_", "stft"),
        (r"whiten|S_n\(|\\tilde\{h\}_w", "whitening"),
    ]
    for pat, tag in checks:
        if re.search(pat, body):
            tags.append(tag)

    # Heading-based fallbacks (helps map explanatory equations to pipeline stages).
    h = (heading or "").lower()
    if "whiten" in h or "raw strain" in h or "narrowband" in h:
        if "whitening" not in tags:
            tags.append("whitening")
    if "time--frequency" in h or "time-frequency" in h or "stft" in h or "spectrogram" in h:
        if "stft" not in tags:
            tags.append("stft")
    if "projection" in h or "instrumental" in h:
        if "P_perp" not in tags:
            tags.append("P_perp")
        if "C_proj" not in tags:
            tags.append("C_proj")
    if "eigenvalue" in h or "eigenspectrum" in h or "eigenvalue" in body.lower():
        if "eta_proj" not in tags:
            tags.append("eta_proj")
        if "C_proj" not in tags:
            tags.append("C_proj")
    if "3d" in h or "three-dimensional" in h:
        if "direction_fit" not in tags:
            tags.append("direction_fit")
    if "reconstruction" in h and ("3d" in h or "spatial" in h):
        if "direction_fit" not in tags:
            tags.append("direction_fit")
    return tags


def tags_to_script_patterns(tags: list[str]) -> list[str]:
    patterns: list[str] = []
    if "P_perp" in tags or "C_proj" in tags or "eta_proj" in tags:
        patterns += ["P_perp", "C_proj", "eta_proj", "off_windows", "C_instr", "Znet_proj", "Cnet"]
    if "C_net" in tags:
        patterns += ["Znet = np.vstack", "Cnet =", "Znet_on", "Znet_off", "np.vstack"]
    if "markers" in tags or "phase_delay" in tags:
        patterns += ["Dmt_", "Dmt_phase", "np.unwrap", "f_center", "mode_profile", "phase_diff"]
    if "direction_fit" in tags:
        patterns += ["reconstruct_k_hat", "ECEF", "geo_to_ecef", "np.linalg.lstsq", "least"]
    if "stft" in tags:
        patterns += ["stft(", "STFT", "Z_h_narrow", "Z_l_narrow", "Z_v_narrow"]
    if "whitening" in tags:
        patterns += [".whiten(", ".bandpass(", "whiten(", "bandpass("]
    if "waveform_compare" in tags:
        patterns += [
            "Waveform comparison",
            "h_geom",
            "h_strain",
            "h_strain_interp",
            "corr_coef",
            "overall waveform correlation",
            "residual_env",
        ]
    return list(dict.fromkeys(patterns))


def format_file_line_link(rel_path: str, line_no: int) -> str:
    rel_path = rel_path.replace("\\", "/")
    return f"[{rel_path}#L{line_no}]({rel_path}#L{line_no})"


def summarize_refs(refs: dict[str, list[CodeHit]]) -> str:
    parts: list[str] = []
    for rel_path in sorted(refs.keys()):
        top = refs[rel_path][0]
        parts.append(format_file_line_link(rel_path, top.line_no))
    return "; ".join(parts)


def extract_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    heading = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        m = SECTION_RE.match(line.strip())
        if m:
            heading = normalize_heading(m.group(1), m.group(2))

        if FIG_ENV_RE.search(line):
            start = i + 1
            j = i
            while j < len(lines) and not END_FIG_ENV_RE.search(lines[j]):
                j += 1
            end = min(j + 1, len(lines))
            fig_lines = lines[i:end]
            body = "\n".join(fig_lines)

            files: list[str] = []
            for fl in fig_lines:
                im = INCLUDE_RE.search(fl)
                if im:
                    files.append(im.group(1) or im.group(2) or "")
            files = [f for f in files if f]

            label = None
            for fl in fig_lines:
                lm = LABEL_RE.search(fl)
                if lm:
                    label = lm.group(1)
                    break

            caption = None
            for idx, fl in enumerate(fig_lines):
                if CAPTION_RE.search(fl):
                    caption = extract_caption(fig_lines, idx)
                    break

            blocks.append(
                Block(
                    kind="figure",
                    env="figure",
                    start_line=i + 1,
                    end_line=end,
                    heading=heading,
                    label=label,
                    files=files,
                    caption=caption,
                    body=body,
                )
            )
            i = end
            continue

        mm = MATH_BEGIN_RE.search(line)
        if mm:
            env = mm.group(1)
            start = i + 1
            j = i + 1
            while j < len(lines):
                if MATH_END_RE.search(lines[j]):
                    break
                j += 1
            end = min(j + 1, len(lines))
            math_lines = lines[i:end]
            body = "\n".join(math_lines)

            label = None
            for ml in math_lines:
                lm = LABEL_RE.search(ml)
                if lm:
                    label = lm.group(1)
                    break

            blocks.append(
                Block(
                    kind="math",
                    env=env,
                    start_line=start,
                    end_line=end,
                    heading=heading,
                    label=label,
                    files=[],
                    caption=None,
                    body=body,
                )
            )
            i = end
            continue

        i += 1

    return blocks


def rel_link(path: str) -> str:
    # Normalize to workspace-relative.
    if path.startswith("./"):
        path = path[2:]
    p = (WORKSPACE / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        return str(p.relative_to(WORKSPACE)).replace('\\', '/')
    except Exception:
        return str(Path(path)).replace('\\', '/')


def main() -> None:
    if not LATEX_FILE.exists():
        raise SystemExit(f"Missing {LATEX_FILE}")

    lines = read_lines(LATEX_FILE)
    blocks = extract_blocks(lines)

    script_indices = build_script_indices()

    figures = [b for b in blocks if b.kind == "figure"]
    maths = [b for b in blocks if b.kind == "math"]

    # Collect workspace image files referenced.
    referenced_images: dict[str, str] = {}  # latex -> resolved
    for fig in figures:
        for f in fig.files:
            referenced_images[f] = rel_link(f)

    # Find all images in workspace (recursive; common types)
    img_exts = {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"}
    skip_dirs = {".venv", "__pycache__", ".git"}
    workspace_images: list[Path] = []
    for p in WORKSPACE.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix.lower() in img_exts:
            workspace_images.append(p.relative_to(WORKSPACE))
    workspace_images.sort(key=lambda p: str(p).lower())

    out: list[str] = []
    out.append("# Paper ↔ Pipeline Correlation Map")
    out.append("")
    out.append(f"Generated from: [{rel_link('main.tex')}]({rel_link('main.tex')})")
    out.append("")
    out.append("This file maps **(a)** every figure included in the LaTeX, and **(b)** every displayed math block (equation/align/etc.),")
    out.append("to: (1) the relevant paper location (line ranges), (2) the producing/related analysis scripts, and (3) any related workspace figure assets.")
    out.append("")
    out.append("Notes:")
    out.append("- ‘Equations’ here means *displayed* math environments (equation/align/gather/multline/eqnarray). Inline $...$ is not enumerated.")
    out.append("- Script links are based on direct keyword matches to the implementation in this workspace.")
    out.append("")

    out.append("## Figures included in the paper")
    out.append("")
    out.append(f"Count: **{len(figures)}**")
    out.append("")
    out.append("| # | Paper location | Label | Included file(s) | Caption (abridged) | Script correlation |")
    out.append("|---:|---|---|---|---|---|")

    for idx, fig in enumerate(figures, start=1):
        paper_loc = f"[main.tex](main.tex#L{fig.start_line}-L{fig.end_line})"
        label = fig.label or ""
        inc = ", ".join(f"[{rel_link(f)}]({rel_link(f)})" for f in fig.files) if fig.files else "LaTeX/TikZ (no external file)"
        cap = (fig.caption or "").strip()
        cap_short = cap if len(cap) <= 120 else cap[:117] + "..."

        # Tag-based mapping for figures (by caption + included filenames).
        tags: list[str] = []
        text_for_tags = (fig.caption or "") + " " + " ".join(fig.files)
        if any(s in text_for_tags.lower() for s in ["wavefront", "three-dimensional", "detector_layout", "ecef"]):
            tags += ["direction_fit", "markers"]
        if any(s in text_for_tags.lower() for s in ["eigenvalue", "spectrum", "projected"]):
            tags += ["C_proj", "eta_proj"]
        if any(s in text_for_tags.lower() for s in ["strain", "waveform", "waaveforms", "reconstructed"]):
            tags += ["waveform_compare"]
        if any(s in text_for_tags.lower() for s in ["spectrogram", "stft"]):
            tags += ["stft"]

        patterns = tags_to_script_patterns(tags)
        refs = find_script_refs(script_indices, patterns) if patterns else {}
        script_summary = summarize_refs(refs)
        out.append(
            f"| {idx} | {paper_loc} | {label} | {inc} | {cap_short} | {script_summary} |"
        )

    out.append("")
    out.append("### Figure details")
    out.append("")
    for idx, fig in enumerate(figures, start=1):
        out.append(f"#### Figure {idx}: {fig.label or '(no label)'}")
        out.append(f"- Paper: [main.tex](main.tex#L{fig.start_line}-L{fig.end_line})")
        out.append(f"- Context: {fig.heading or ''}")
        if fig.caption:
            out.append(f"- Caption: {fig.caption}")
        if fig.files:
            out.append("- Assets:")
            for f in fig.files:
                out.append(f"  - [{rel_link(f)}]({rel_link(f)})")
        else:
            out.append("- Assets: LaTeX/TikZ (no external files)")

        tags: list[str] = []
        text_for_tags = (fig.caption or "") + " " + " ".join(fig.files)
        if any(s in text_for_tags.lower() for s in ["wavefront", "three-dimensional", "detector_layout", "ecef", "3d"]):
            tags += ["direction_fit", "markers"]
        if any(s in text_for_tags.lower() for s in ["eigenvalue", "eigenspectrum", "projected", "instrumental", "projection"]):
            tags += ["C_proj", "eta_proj", "P_perp"]
        if any(s in text_for_tags.lower() for s in ["strain", "waveform", "reconstructed"]):
            tags += ["waveform_compare"]
        if any(s in text_for_tags.lower() for s in ["spectrogram", "time--frequency", "stft"]):
            tags += ["stft"]
        patterns = tags_to_script_patterns(tags)
        refs = find_script_refs(script_indices, patterns) if patterns else {}
        if refs:
            out.append("- Related code:")
            for rel_path in sorted(refs.keys()):
                out.append(f"  - {rel_path}")
                for hit in refs[rel_path]:
                    out.append(f"    - {format_file_line_link(rel_path, hit.line_no)}: {hit.line}")
        else:
            out.append("- Related code: (no direct keyword match; static asset or LaTeX-only figure)")
        out.append("")

    out.append("")
    out.append("### Workspace image assets")
    out.append("")
    out.append("Images present in the workspace (not necessarily all used by the paper):")
    referenced_set = {rel_link(k) for k in referenced_images.keys()}
    referenced = [p for p in workspace_images if p.as_posix() in referenced_set]
    unreferenced = [p for p in workspace_images if p not in referenced]

    out.append("")
    out.append(f"Referenced by LaTeX: **{len(referenced)}**")
    for p in referenced:
        rp = p.as_posix()
        out.append(f"- [{rp}]({rp})")

    out.append("")
    out.append(f"Not referenced by LaTeX: **{len(unreferenced)}**")
    for p in unreferenced:
        rp = p.as_posix()
        out.append(f"- [{rp}]({rp})")

    out.append("")
    out.append("## Displayed equations / math blocks")
    out.append("")
    out.append(f"Count: **{len(maths)}**")
    out.append("")
    out.append("| # | Paper location | Environment | Label | Section context | Tag(s) | Script correlation |")
    out.append("|---:|---|---|---|---|---|---|")

    for idx, eq in enumerate(maths, start=1):
        paper_loc = f"[main.tex](main.tex#L{eq.start_line}-L{eq.end_line})"
        label = eq.label or ""
        tags = classify_equation(eq.body, eq.heading)
        patterns = tags_to_script_patterns(tags)
        refs = find_script_refs(script_indices, patterns) if patterns else {}
        script_summary = summarize_refs(refs)
        out.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    paper_loc,
                    eq.env,
                    label,
                    eq.heading,
                    ", ".join(tags),
                    script_summary,
                ]
            )
            + " |"
        )

    out.append("")
    out.append("### Equation details")
    out.append("")
    for idx, eq in enumerate(maths, start=1):
        out.append(f"#### Equation {idx}: {eq.label or '(no label)'}")
        out.append(f"- Paper: [main.tex](main.tex#L{eq.start_line}-L{eq.end_line})")
        out.append(f"- Context: {eq.heading or ''}")
        tags = classify_equation(eq.body, eq.heading)
        out.append(f"- Tags: {', '.join(tags) if tags else '(none)'}")
        patterns = tags_to_script_patterns(tags)
        refs = find_script_refs(script_indices, patterns) if patterns else {}
        if refs:
            out.append("- Related code:")
            for rel_path in sorted(refs.keys()):
                out.append(f"  - {rel_path}")
                for hit in refs[rel_path]:
                    out.append(f"    - {format_file_line_link(rel_path, hit.line_no)}: {hit.line}")
        else:
            out.append("- Related code: (no direct keyword match)")
        out.append("")

    out.append("")
    out.append("## Script inventory (what generates what)")
    out.append("")
    out.append("- pipelineGW.py: STFT construction, active-window selection, stacked network operator, off-source instrumental projection, projected eigenvalues/\u03b7, per-detector Dmt markers and phase-derived delays.")
    out.append("- 3DFullVisualization.py: converts phase-derived delays + amplitude ratios into an ECEF least-squares propagation direction and 3D geometry figures.")
    out.append("- waveforms.py: loads per-phase .npz products and compares a reconstructed time series to public strain; produces correlation/RMS summaries and the waveform comparison plot asset.")
    out.append("- ProjectedEighenValues.py: plots projected eigenvalues (as currently hard-coded in that script).")
    out.append("")

    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
