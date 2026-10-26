"""Read-only allowlist checks: no image build, archive, secrets or app imports."""

from pathlib import Path
import ast
import hashlib
import io
import re
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib
import zipfile

from tools import build_submission_archive as packaging


class SubmissionPackagingTests(unittest.TestCase):
    def test_docker_source_output_resolves_to_owned_shared_data(self):
        dockerfile = (packaging.ROOT / "Dockerfile").read_text(encoding="utf-8")
        logical = dockerfile.replace("\\\n", " ")
        self.assertIn("mkdir -p /app /data/output", logical)
        self.assertIn("ln -s /data/output /app/output", logical)
        self.assertIn("chown planner:planner /data /data/output", logical)
        self.assertIn("WORKDIR /data\nUSER planner", dockerfile)
        self.assertLess(logical.index("ln -s /data/output /app/output"), logical.index("USER planner"))

    def test_both_containers_keep_read_only_root_and_shared_data_volume(self):
        compose = (packaging.ROOT / "compose.yaml").read_text(encoding="utf-8")
        services = dict(re.findall(r"^  (web|bot):\n(.*?)(?=^  [A-Za-z_][\w-]*:|\Z)",
                                   compose, flags=re.MULTILINE | re.DOTALL))
        self.assertEqual(set(services), {"web", "bot"})
        for service, block in services.items():
            with self.subTest(service=service):
                self.assertRegex(block, r"(?m)^    read_only: true$")
                self.assertRegex(block, r"(?m)^      - planner-data:/data$")
                self.assertNotRegex(block, r"(?m)^    privileged: true$")
                self.assertIn("      - no-new-privileges:true", block)

    def test_source_relative_runtime_output_is_covered_by_docker_symlink(self):
        # Parse source without importing application modules or creating clocks.
        source_output = []
        for name in packaging.RUNTIME_MODULES:
            tree = ast.parse((packaging.ROOT / name).read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                    continue
                descendants = tuple(ast.walk(node.value))
                if (any(isinstance(part, ast.Name) and part.id == "__file__" for part in descendants)
                        and any(isinstance(part, ast.Constant) and part.value == "output"
                                for part in descendants)):
                    source_output.append((name, ast.unparse(node.value)))
        self.assertEqual(source_output, [
            ("places.py", "Path(__file__).resolve().parent / 'output' / 'nominatim_clock.db'"),
        ])

    def check_fixture(self, text, name="README.md"):
        with tempfile.TemporaryDirectory() as directory, patch.object(packaging, "ROOT", Path(directory)):
            path = Path(directory) / name
            path.write_text(text, encoding="utf-8", newline="\n")
            return packaging.checked_content(path)

    def test_known_credential_formats_and_private_links_are_rejected(self):
        # Build unmistakably synthetic fixtures in pieces: source packaging must
        # reject real token-shaped literals, including those inside test files.
        samples = [
            "123456789:" + "A" * 35,
            "AQVN" + "A" * 36,
            "pcsk_" + "B" * 40,
            "sk-proj-" + "C" * 40,
            "sk-ant-" + "D" * 40,
            "ghp_" + "E" * 36,
            "github_pat_" + "F" * 60,
            "AKIA" + "G" * 16,
            "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
            "-----BEGIN " + "ENCRYPTED PRIVATE KEY-----",
            "https://example.invalid/?token=" + "H" * 48,
            "https://example.invalid/?start=invite_" + "J" * 40,
        ]
        for index, sample in enumerate(samples):
            with self.subTest(index=index), self.assertRaises(ValueError) as caught:
                self.check_fixture(sample)
            self.assertNotIn(sample, str(caught.exception))

    def test_public_template_rejects_non_placeholder_sensitive_values(self):
        for key in packaging.SENSITIVE_ENV_KEYS:
            for value in ("synthetic-value-123", '"synthetic-value-123"'):
                with self.subTest(key=key, quoted=value.startswith('"')):
                    with self.assertRaises(ValueError) as caught:
                        self.check_fixture(key + "=" + value + "\n", ".env.example")
                    self.assertNotIn(value, str(caught.exception))

    def test_public_template_allows_blank_values_and_explicit_placeholders(self):
        for value in ("", '""', "ваш_ключ", "токен_из_настройки", "YOUR_API_KEY", "change-me"):
            text = "YANDEX_API_KEY=" + value + "\n"
            self.assertEqual(self.check_fixture(text, ".env.example"), text.encode())
        self.check_fixture("ACCESS_OWNER_IDS=\n", ".env.example")

    def test_public_template_rejects_a_personal_owner_list(self):
        with self.assertRaises(ValueError):
            self.check_fixture("ACCESS_OWNER_IDS=10001\n", ".env.example")

    def test_python_direct_settings_are_checked_without_execution(self):
        for key in packaging.SENSITIVE_ENV_KEYS:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_fixture(key + " = 'synthetic-value-123'\n", "example.py")
            self.check_fixture(key + " = None\n", "example.py")
        with self.assertRaises(ValueError):
            self.check_fixture("TARGET_OWNER_ID = 10001\n", "repair_trip3.py")
        self.check_fixture("TARGET_OWNER_ID: int | None = None\n", "repair_trip3.py")
        self.check_fixture("TEST_OWNER_ID = 10001\n", "try_repair_trip3.py")

    def test_every_selected_source_passes_content_safety_checks(self):
        for path in packaging.source_files():
            with self.subTest(path=path.relative_to(packaging.ROOT).as_posix()):
                packaging.checked_content(path)

    def test_ignore_rules_cover_common_private_files_but_keep_public_template(self):
        rules = set((packaging.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        required = {".env", ".env.*", "!.env.example", "*.db", "*.db-wal", "*.db-shm",
                    "*.sqlite", "*.sqlite3", "*.pem", "*.key", "id_ed25519",
                    "id_rsa", "output/", "tmp/", ".venv/", ".vscode/", "*.log"}
        self.assertTrue(required.issubset(rules))

    def docx(self, *, extra=None, document_xml=None):
        parts = {
            "[Content_Types].xml": (
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
                'officedocument.wordprocessingml.document.main+xml"/></Types>'),
            "_rels/.rels": (
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                'relationships/officeDocument" Target="word/document.xml"/></Relationships>'),
            "word/document.xml": document_xml or (
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body><w:p><w:r><w:t>Public report</w:t></w:r></w:p></w:body></w:document>'),
        }
        parts.update(extra or {})
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in parts.items():
                # Assign after construction so Windows does not normalize the
                # deliberately unsafe names used by the negative tests.
                entry = zipfile.ZipInfo()
                entry.filename = name
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, value)
        return output.getvalue()

    def test_final_reports_have_an_explicit_allowlist_and_bytes_are_preserved(self):
        self.assertEqual(set(packaging.SUBMISSION_FILES), {
            "docs/submission/specification.docx", "docs/submission/additional-features.docx",
            "docs/submission/architecture.docx", "docs/submission/prompt-library.docx",
            "docs/submission/full-prompts.txt",
        })
        names = {path.relative_to(packaging.ROOT).as_posix() for path in packaging.source_files()}
        self.assertTrue(set(packaging.SUBMISSION_FILES).issubset(names))
        self.assertIn("requirements-dev.txt", names)
        for name in packaging.SUBMISSION_FILES:
            with self.subTest(name=name):
                path = packaging.ROOT / name
                self.assertEqual(packaging.checked_content(path), path.read_bytes())

    def test_docx_rejects_macros_objects_scripts_unsafe_paths_and_duplicate_parts(self):
        for path in ("word/vbaProject.bin", "word/embeddings/object.bin", "word/script.ps1",
                     "../private.xml", "/absolute.xml", "word\\unsafe.xml", "WORD/document.xml"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                packaging._checked_docx(self.docx(extra={path: "<part/>"}), "report.docx")
        safe = self.docx(extra={"word/media/diagram.png": self.png()})
        self.assertEqual(packaging._checked_docx(safe, "report.docx"), safe)

    def test_docx_checks_decoded_xml_attributes_and_text_across_word_runs(self):
        prefix = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p>'
        suffix = '</w:p></w:body></w:document>'
        bodies = [
            '<w:r><w:t>AQVN' + '&#65;' * 36 + '</w:t></w:r>',
            '<w:r><w:t>AQVN' + 'A' * 12 + '</w:t></w:r><w:r><w:t>' + 'A' * 24 + '</w:t></w:r>',
            '<w:r data="AQVN' + '&#65;' * 36 + '"><w:t>Text</w:t></w:r>',
        ]
        for index, body in enumerate(bodies):
            with self.subTest(index=index), self.assertRaises(ValueError):
                packaging._checked_docx(self.docx(document_xml=prefix + body + suffix), "report.docx")

    def test_docx_rejects_external_relationships_dtd_and_active_fields(self):
        external = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                    '<Relationship Id="rId1" Target="https://example.invalid/" TargetMode="External"/>'
                    '</Relationships>')
        with self.assertRaises(ValueError):
            packaging._checked_docx(self.docx(extra={"word/_rels/document.xml.rels": external}), "report.docx")
        for xml in ('<!DOCTYPE example [<!ENTITY a "x">]><example/>', '<broken>'):
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                packaging._checked_docx(self.docx(extra={"word/extra.xml": xml}), "report.docx")
        field = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                 '<w:body><w:p><w:r><w:instrText>DDEAUTO synthetic</w:instrText></w:r></w:p></w:body></w:document>')
        with self.assertRaises(ValueError):
            packaging._checked_docx(self.docx(document_xml=field), "report.docx")

    def test_docx_requires_expected_structure_and_has_a_part_size_bound(self):
        invalid = [b"not-a-zip", self.docx(extra={"word/document.xml": "<wrong/>"}),
                   self.docx(extra={"word/large.xml": "x" * (8 * 1024 * 1024 + 1)})]
        for index, payload in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                packaging._checked_docx(payload, "report.docx")

    @staticmethod
    def png_chunk(kind, payload=b""):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    def png(self, width=1, height=1, extra=b""):
        return (b"\x89PNG\r\n\x1a\n"
                + self.png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + extra + self.png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
                + self.png_chunk(b"IEND"))

    def test_offline_acceptance_runner_and_its_scripts_are_in_source_package(self):
        from tools.run_offline_checks import SMOKE_SCRIPTS

        names = {path.relative_to(packaging.ROOT).as_posix() for path in packaging.source_files()}
        self.assertIn("tools/run_offline_checks.py", packaging.SUPPORT_FILES)
        self.assertIn("tools/run_offline_checks.py", names)
        self.assertEqual(len(SMOKE_SCRIPTS), 11)
        self.assertTrue(set(SMOKE_SCRIPTS).issubset(names))
        self.assertTrue({"test_web_ui.js", "test_region_request.py"}.issubset(names))

    def test_local_planning_modules_and_imports_are_in_runtime_image(self):
        required = {"regional_discovery.py", "regional_skeleton.py", "regional_routing.py", "visit_facts.py", "source_planning.py", "verified_map_places.py", "city_response.py", "city_candidates.py", "city_planner.py", "city_quality.py", "city_schedule.py"}
        self.assertTrue(required.issubset(packaging.RUNTIME_MODULES))
        packaging.check_docker_sources()

    def test_public_visit_catalog_is_explicitly_in_every_allowlist(self):
        dockerfile = (packaging.ROOT / "Dockerfile").read_text(encoding="utf-8")
        allowed = (packaging.ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        for name in ("data/visit_facts.json", "data/verified_map_places.json"):
            with self.subTest(name=name):
                self.assertIn(name, packaging.SUPPORT_FILES)
                self.assertIn(f"COPY {name} /app/{name}", dockerfile)
                self.assertIn(f"!{name}", allowed)
                self.assertTrue((packaging.ROOT / name).is_file())

    def test_invitation_access_modules_are_in_source_and_runtime_image(self):
        names = {path.relative_to(packaging.ROOT).as_posix() for path in packaging.source_files()}
        allowed = (packaging.ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        for name in ("access_control.py", "access_guard.py"):
            with self.subTest(name=name):
                self.assertIn(name, packaging.RUNTIME_MODULES)
                self.assertIn(name, names)
                self.assertIn("!" + name, allowed)
        packaging.check_docker_sources()

    def test_single_trip_deletion_is_in_source_and_runtime_image(self):
        name = "trip_deletion.py"
        self.assertIn(name, packaging.RUNTIME_MODULES)
        allowed = (packaging.ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("!" + name, allowed)
        self.assertIn(packaging.ROOT / name, packaging.source_files())
        packaging.check_docker_sources()

    def test_public_environment_template_defaults_to_closed_invitation_access(self):
        # Only the public template is read. Never inspect the operator's .env.
        settings = {}
        for line in (packaging.ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip()
        self.assertEqual(settings["ACCESS_MODE"], "invite")
        self.assertEqual(settings["ACCESS_OWNER_IDS"], "")
        self.assertEqual(settings["ACCESS_GLOBAL_ROUTE_DAILY_LIMIT"], "12")
        self.assertEqual(settings["DAILY_LIMIT"], "3")
        self.assertEqual(settings["IMAGE_GENERATION_DAILY_LIMIT"], "10")
        self.assertEqual(settings["IMAGE_GENERATION_ENABLED"], "0")

    def test_selected_source_paths_exclude_runtime_data_and_secrets(self):
        names = {path.relative_to(packaging.ROOT).as_posix() for path in packaging.source_files()}
        self.assertTrue({"regional_skeleton.py", "regional_routing.py", "visit_facts.py", "source_planning.py",
                         "data/visit_facts.json", "verified_map_places.py", "data/verified_map_places.json",
                         "tools/replay_source_plan.py", "test_web_ui.js"}.issubset(names))
        self.assertNotIn(".env", names)
        for name in names:
            self.assertFalse(name.startswith(("output/", "tmp/", ".git/", ".venv/")), name)
            self.assertFalse(name.lower().endswith((".db", ".sqlite", ".sqlite3", ".zip")), name)

    def test_cover_prompt_and_exact_public_assets_are_packaged(self):
        self.assertIn("cover_prompts.py", packaging.RUNTIME_MODULES)
        self.assertEqual(set(packaging.COVER_ASSETS), {
            "web/static/images/travel-hero.png", "web/static/images/sochi.png",
            "web/static/images/baikal.png", "web/static/images/nha-trang.png",
            "web/static/images/moscow.png",
        })
        allowed = (packaging.ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertNotIn("!web/static/images/*", allowed)
        self.assertNotIn("!web/static/images/*.png", allowed)
        for name in packaging.COVER_ASSETS:
            with self.subTest(name=name):
                self.assertIn(name, packaging.WEB_FILES)
                self.assertIn("!" + name, allowed)

    def test_explicit_png_is_preserved_as_bytes_without_utf8_decode(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(packaging, "ROOT", Path(directory)):
            path = Path(directory) / packaging.COVER_ASSETS[0]
            path.parent.mkdir(parents=True)
            path.write_bytes(self.png())
            self.assertEqual(packaging.checked_content(path), self.png())

    def test_selected_public_png_assets_pass_binary_validation(self):
        for name in packaging.COVER_ASSETS:
            with self.subTest(name=name):
                content = packaging.checked_content(packaging.ROOT / name)
                self.assertTrue(content.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(hashlib.sha256(content).hexdigest(), packaging.APPROVED_COVER_SHA256[name])

    def test_reviewed_bounded_generated_provenance_is_preserved_unchanged(self):
        name = packaging.COVER_ASSETS[0]
        content = self.png(extra=self.png_chunk(b"caBX", b"test provenance fixture"))
        with patch.dict(packaging.APPROVED_COVER_SHA256, {name: hashlib.sha256(content).hexdigest()}):
            self.assertEqual(packaging._checked_png(content, name), content)

    def test_provenance_requires_exact_reviewed_asset_name_and_hash(self):
        name = packaging.COVER_ASSETS[0]
        content = self.png(extra=self.png_chunk(b"caBX", b"test provenance fixture"))
        with self.assertRaises(ValueError):
            packaging._checked_png(content, name)
        with patch.dict(packaging.APPROVED_COVER_SHA256, {name: hashlib.sha256(content).hexdigest()}):
            with self.assertRaises(ValueError):
                packaging._checked_png(content, "web/static/images/not-approved.png")
            changed = self.png(extra=self.png_chunk(b"caBX", b"different provenance fixture"))
            with self.assertRaises(ValueError):
                packaging._checked_png(changed, name)

    def test_even_reviewed_provenance_has_length_and_duplicate_limits(self):
        name = packaging.COVER_ASSETS[0]
        for content in (
            self.png(extra=self.png_chunk(b"caBX", b"x" * (256 * 1024 + 1))),
            self.png(extra=self.png_chunk(b"caBX", b"a") + self.png_chunk(b"caBX", b"b")),
        ):
            with patch.dict(packaging.APPROVED_COVER_SHA256, {name: hashlib.sha256(content).hexdigest()}):
                with self.assertRaises(ValueError):
                    packaging._checked_png(content, name)

    def test_unknown_png_does_not_bypass_source_text_checks(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(packaging, "ROOT", Path(directory)):
            path = Path(directory) / "not-approved.png"
            path.write_bytes(self.png())
            with self.assertRaises(UnicodeDecodeError):
                packaging.checked_content(path)

    def test_png_rejects_wrong_type_oversize_dimensions_metadata_and_trailing_data(self):
        malformed = (
            b"not a png", self.png(width=0), self.png(width=4097), self.png(height=4097),
            self.png(extra=self.png_chunk(b"tEXt", b"private metadata")),
            self.png(extra=self.png_chunk(b"zTXt", b"compressed metadata")),
            self.png(extra=self.png_chunk(b"iTXt", b"international metadata")),
            self.png(extra=self.png_chunk(b"eXIf", b"private location")),
            self.png() + b"appended payload", self.png()[:-1],
            self.png()[:29] + b"0000" + self.png()[33:],
            self.png(extra=self.png_chunk(b"FAKE")),
        )
        for index, content in enumerate(malformed):
            with self.subTest(index=index), self.assertRaises(ValueError):
                packaging._checked_png(content, packaging.COVER_ASSETS[0])

    def test_png_requires_header_image_data_and_end_marker(self):
        header = b"\x89PNG\r\n\x1a\n"
        ihdr = self.png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        for content in (header, header + ihdr, header + ihdr + self.png_chunk(b"IEND"),
                        header + self.png_chunk(b"IDAT") + ihdr + self.png_chunk(b"IEND")):
            with self.subTest(size=len(content)), self.assertRaises(ValueError):
                packaging._checked_png(content, packaging.COVER_ASSETS[0])


if __name__ == "__main__":
    unittest.main()
