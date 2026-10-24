"""Build a source-only submission ZIP without reading local secrets or databases.

Run from any directory: python tools/build_submission_archive.py
The output includes a SHA-256 manifest. Existing archives are never overwritten.
No application modules are imported and no network connections are made.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import shlex
import struct
import zipfile
import zlib
from xml.etree import ElementTree
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = (
    "access_control.py", "access_guard.py",
    "anketa.py", "bot.py", "candidate_grounding.py", "checklist.py", "city_candidates.py", "city_coverage.py", "city_estimates.py", "city_planner.py", "city_quality.py", "city_response.py", "city_schedule.py", "db.py", "destination_name_hints.py", "destination_scope.py",
    "formatting.py", "generation_feedback.py", "grounded_response.py", "grounded_schedule.py", "memory.py", "pdf_export.py", "privacy.py", "places.py",
    "planner.py", "pricing.py", "program_coverage.py", "prompts.py", "regional_routing.py", "regional_skeleton.py", "reminders.py", "rules.py",
    "schemas.py", "source_planning.py", "visit_facts.py", "verified_map_places.py", "web_app.py", "web_auth.py", "web_service.py", "weather.py",
    "yandex_ai.py", "city_visit_estimates.py", "city_rollout.py", "city_context.py", "city_edit.py", "city_edit_service.py",
    "city_deterministic.py", "city_visit_options.py", "city_preferences.py", "city_evidence.py", "generation_jobs.py", "generation_quality.py", "cover_prompts.py", "cover_subjects.py",
    "yandex_art.py", "trip_covers.py", "trip_deletion.py",
)
SUPPORT_FILES = (
    ".dockerignore", ".env.example", ".gitignore", "Dockerfile", "compose.yaml",
    "requirements.txt", "requirements-dev.txt", "README.md", "setup_web.py",
    "setup_pinecone.py", "data/verified_places.json", "data/destination_name_hints.json", "data/visit_facts.json",
    "data/verified_map_places.json",
    "tools/build_submission_archive.py", "tools/run_offline_checks.py",
    "tools/diagnose_generation.py",
    "tools/check_city_acceptance.py",
    "tools/check_generation_matrix.py",
    "tools/replay_grounding.py",
    "tools/replay_source_plan.py", "tools/preview_source_plan.py", "test_web_ui.js",
)
COVER_ASSETS = (
    "web/static/images/travel-hero.png",
    "web/static/images/sochi.png",
    "web/static/images/baikal.png",
    "web/static/images/nha-trang.png",
    "web/static/images/moscow.png",
)
# These reviewed generated assets retain their original provenance, unchanged.
# A replacement caBX-bearing cover requires explicit review and a new digest.
APPROVED_COVER_SHA256 = {
    "web/static/images/travel-hero.png": "5a218747172ba9dac47e8ff5bb4635195a27f17bee8f22ca07fed9f09bd16ea7",
    "web/static/images/sochi.png": "d59713333d392124597e3dad591e9454f24a8f17218072aa6284087b59958ba1",
    "web/static/images/baikal.png": "fc440b6a0233e5ab7d31c31722543ae7f5fb3db7d69277f340c92a231aeec51b",
    "web/static/images/nha-trang.png": "198e5798d9cd9143a076cf92a4a827f4350965d6781fbc9b61f9fba4a67eb46f",
    "web/static/images/moscow.png": "4926866f7c4f877d3fe3a540d8b4172691cc290a4cba92234e8f410a09f7a724",
}
WEB_FILES = (
    "web/index.html", "web/static/app.js", "web/static/styles.css",
    "web/static/favicon.svg", "web/static/vendor/leaflet.js",
    "web/static/vendor/leaflet.css", "web/static/vendor/leaflet.LICENSE",
) + COVER_ASSETS
SUBMISSION_FILES = (
    "docs/submission/specification.docx",
    "docs/submission/additional-features.docx",
    "docs/submission/architecture.docx",
    "docs/submission/prompt-library.docx",
    "docs/submission/full-prompts.txt",
)
TOKEN_PATTERNS = (
    re.compile(r"\b\d{7,12}:[A-Za-z0-9_-]{30,}\b"),  # Telegram bot token
    re.compile(r"\b(?:sk-proj-|sk-ant-|pcsk_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAQVN[A-Za-z0-9_-]{30,}\b"),  # Yandex Cloud API key
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    re.compile(r"(?:[?&#]token=|[?&]start=invite_)[A-Za-z0-9_-]{24,}"),
)
SENSITIVE_ENV_KEYS = frozenset({
    "YANDEX_API_KEY", "YANDEX_ART_API_KEY", "PINECONE_API_KEY",
    "TELEGRAM_BOT_TOKEN", "WEB_SESSION_SECRET",
})


def _placeholder(value: str) -> bool:
    value = value.strip().strip("\"'")
    return not value or value.startswith(("ваш_", "токен_", "change-me", "YOUR_"))


def _check_literal_python_credentials(text: str, name: str) -> None:
    """Catch direct hard-coded settings without executing code or test fixtures.

    This complements known token formats, not a universal secret detector.
    Dynamic expressions and unknown credential formats still require review.
    """
    tree = ast.parse(text, filename=name)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            key = target.id.upper()
            value = node.value
            if key in SENSITIVE_ENV_KEYS and isinstance(value, ast.Constant):
                if isinstance(value.value, str) and not _placeholder(value.value):
                    raise ValueError(f"Hard-coded credential setting in source: {name}: {key}")
            if key == "TARGET_OWNER_ID" and isinstance(value, ast.Constant) and value.value is not None:
                raise ValueError(f"Personal owner must not be embedded in source: {name}")


def source_files() -> list[Path]:
    names = set(RUNTIME_MODULES + SUPPORT_FILES + WEB_FILES + SUBMISSION_FILES)
    names.update(path.name for pattern in ("test_*.py", "try_*.py") for path in ROOT.glob(pattern))
    names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob("*.md"))
    paths = [ROOT / name for name in sorted(names)]
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or unsafe source file: {path.relative_to(ROOT)}")
    return paths


def check_docker_sources() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    logical_lines = dockerfile.replace("\\\n", " ").splitlines()
    copied = set()
    for line in logical_lines:
        if line.strip().upper().startswith("COPY "):
            copied.update(shlex.split(line.strip())[1:-1])
    missing = set(RUNTIME_MODULES) - copied
    if missing:
        raise ValueError("Dockerfile omits runtime modules: " + ", ".join(sorted(missing)))

    allowed = {
        line.strip()[1:] for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("!")
    }
    excluded = {name for name in copied if name not in allowed}
    if excluded:
        raise ValueError("COPY source not allowed by .dockerignore: " + ", ".join(sorted(excluded)))

    local_modules = {path.stem for path in ROOT.glob("*.py")}
    bundled_modules = {Path(name).stem for name in RUNTIME_MODULES}
    for name in RUNTIME_MODULES:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8-sig"), filename=name)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        missing_imports = (imports & local_modules) - bundled_modules
        if missing_imports:
            raise ValueError(f"Runtime import omitted from image: {name}: {sorted(missing_imports)}")


def _checked_png(content: bytes, name: str) -> bytes:
    """Check only explicitly selected public covers, never scan output images.

    Bounding and chunk checks preserve a small deterministic source package.
    This is not an image decoder or a claim about visual/geographic accuracy.
    Text/EXIF metadata and appended payloads are excluded from the package.
    Original bounded caBX provenance is preserved for exact reviewed asset hashes;
    this validator does not claim to verify its C2PA signature or authenticity.
    """
    def invalid() -> None:
        raise ValueError(f"Invalid or unsupported public PNG asset: {name}")

    if len(content) > 20 * 1024 * 1024 or not content.startswith(b"\x89PNG\r\n\x1a\n"):
        invalid()
    offset = 8
    chunks = []
    allowed_chunks = {
        b"IHDR", b"PLTE", b"IDAT", b"IEND", b"cHRM", b"gAMA", b"iCCP", b"sBIT",
        b"sRGB", b"bKGD", b"hIST", b"tRNS", b"pHYs", b"sPLT", b"tIME",
    }
    reviewed_provenance = (name in COVER_ASSETS and
                           hashlib.sha256(content).hexdigest() == APPROVED_COVER_SHA256.get(name))
    while offset < len(content):
        if offset + 12 > len(content):
            invalid()
        length, kind = struct.unpack(">I4s", content[offset:offset + 8])
        end = offset + 12 + length
        if kind == b"caBX":
            if not reviewed_provenance or length > 256 * 1024 or kind in chunks:
                invalid()
        elif kind not in allowed_chunks:
            invalid()
        if end > len(content):
            invalid()
        payload = content[offset + 8:offset + 8 + length]
        checksum = struct.unpack(">I", content[end - 4:end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != checksum:
            invalid()
        if not chunks:
            if kind != b"IHDR" or length != 13:
                invalid()
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            valid_depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
            if (not 1 <= width <= 4096 or not 1 <= height <= 4096
                    or depth not in valid_depths.get(color, set())
                    or compression != 0 or filtering != 0 or interlace not in {0, 1}):
                invalid()
        elif kind == b"IHDR":
            invalid()
        chunks.append(kind)
        offset = end
        if kind == b"IEND":
            if length or end != len(content) or b"IDAT" not in chunks:
                invalid()
            break
    if not chunks or chunks[-1] != b"IEND":
        invalid()
    return content


def _checked_docx(content: bytes, name: str) -> bytes:
    """Validate the exact approved reports without changing their Word bytes.

    Only bounded Open XML plus validated PNG media are permitted.
    No macros, scripts, embedded/OLE objects or external relationships are
    accepted. This structural check does not replace reviewing document text.
    """
    def invalid() -> None:
        raise ValueError(f"Invalid or unsafe public DOCX document: {name}")

    if len(content) > 16 * 1024 * 1024 or not content.startswith(b"PK\x03\x04"):
        invalid()
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if not 3 <= len(entries) <= 512 or sum(item.file_size for item in entries) > 48 * 1024 * 1024:
                invalid()
            names = set()
            xml_parts = {}
            for entry in entries:
                part = entry.filename
                lower = part.lower()
                if (not part or part != entry.orig_filename or lower in names or part.startswith(("/", "\\"))
                        or "\\" in part or ":" in part or any(p in {"", ".", ".."} for p in part.split("/"))
                        or entry.flag_bits & 1 or entry.is_dir()
                        or (entry.external_attr >> 16) & 0o170000 == 0o120000
                        or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                        or entry.file_size > 8 * 1024 * 1024):
                    invalid()
                names.add(lower)
                if lower.startswith("word/embeddings/") or "vba" in lower:
                    invalid()
                payload = archive.read(entry)
                if lower.endswith((".xml", ".rels")):
                    text = payload.decode("utf-8-sig")
                    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
                        invalid()
                    xml = ElementTree.fromstring(text)
                    # Check raw attributes and joined decoded text, so XML
                    # entities or split Word runs do not conceal known tokens.
                    joined = "".join(xml.itertext())
                    attribute_values = "\n".join(value for node in xml.iter() for value in node.attrib.values())
                    if any(pattern.search(value) for pattern in TOKEN_PATTERNS for value in (text, joined, attribute_values)):
                        raise ValueError(f"Possible credential in public DOCX; archive not written: {name}")
                    if any(term in text.lower() for term in ("macroenabled", "vbaproject", "/oleobject", "/attachedtemplate")):
                        invalid()
                    if any(node.tag.rsplit("}", 1)[-1] == "instrText" and
                           re.search(r"\b(?:DDEAUTO|DDE|INCLUDETEXT|INCLUDEPICTURE|LINK)\b", node.text or "", re.IGNORECASE)
                           for node in xml.iter()):
                        invalid()
                    if lower.endswith(".rels") and any(
                            node.attrib.get("TargetMode", "").lower() == "external" for node in xml.iter()):
                        invalid()
                    xml_parts[part] = xml
                elif lower.startswith("word/media/") and lower.endswith(".png"):
                    _checked_png(payload, name + ":" + part)
                else:
                    invalid()
            types = xml_parts.get("[Content_Types].xml")
            relations = xml_parts.get("_rels/.rels")
            document = xml_parts.get("word/document.xml")
            if types is None or relations is None or document is None:
                invalid()
            if (types.tag != "{http://schemas.openxmlformats.org/package/2006/content-types}Types"
                    or relations.tag != "{http://schemas.openxmlformats.org/package/2006/relationships}Relationships"
                    or document.tag != "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}document"):
                invalid()
            if not any(node.attrib.get("PartName") == "/word/document.xml" and node.attrib.get("ContentType") ==
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
                       for node in types):
                invalid()
            if not any(node.attrib.get("Type") == "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
                       and node.attrib.get("Target") in {"word/document.xml", "/word/document.xml"}
                       for node in relations):
                invalid()
    except (zipfile.BadZipFile, UnicodeError, ElementTree.ParseError, RuntimeError, NotImplementedError):
        invalid()
    return content


def checked_content(path: Path) -> bytes:
    content = path.read_bytes()
    name = path.relative_to(ROOT).as_posix()
    if name in COVER_ASSETS:
        return _checked_png(content, name)
    if name in SUBMISSION_FILES and path.suffix == ".docx":
        return _checked_docx(content, name)
    text = content.decode("utf-8-sig")
    if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
        raise ValueError(f"Possible credential in source; archive not written: {path.relative_to(ROOT)}")
    if path.suffix == ".py":
        _check_literal_python_credentials(text, name)
    if path.name == ".env.example":
        for line in text.splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            if key in SENSITIVE_ENV_KEYS and not _placeholder(value):
                raise ValueError(f"Non-placeholder credential in .env.example: {key}")
            if key == "ACCESS_OWNER_IDS" and value.strip().strip("\"'"):
                raise ValueError("Personal owners must not be included in .env.example")
    return content


def build(output: Path) -> dict:
    check_docker_sources()
    files = source_files()
    # Read and validate only the explicit source set before creating an archive.
    contents = [(path.relative_to(ROOT).as_posix(), checked_content(path)) for path in files]
    manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in contents)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in contents + [("MANIFEST.sha256", manifest.encode("utf-8"))]:
            entry = zipfile.ZipInfo("family-trip-planner/" + name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        for name, data in contents:
            if archive.read("family-trip-planner/" + name) != data:
                raise ValueError(f"ZIP content mismatch: {name}")
    return {
        "archive": str(output.resolve()),
        "source_files": len(contents),
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "checks": ["runtime imports", "Docker COPY allowlist", "credential patterns", "public PNG structure",
                   "explicit public DOCX structure and content", "ZIP content and CRC"],
        "excluded": [".env", "databases", "output", "tmp", ".git", ".venv", "personal screenshots"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "submission" /
                        f"family-trip-planner-source-{datetime.now():%Y%m%d-%H%M%S}.zip")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
