"""Read-only local UI preview of a successfully validated source replay.

No account, database, AI, credentials or application POST endpoint is used.
The real current UI assets and detail serializer render one labelled example.
Bind only to 127.0.0.1; stop with Ctrl+C. This is not a new live generation.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def preview_data(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "output") or not path.is_file() or path.is_symlink():
        raise ValueError("Use a local successful report inside output/.")
    # Serializer imports transitively reach memory.load_dotenv. Preview must
    # not read application credentials, including on a cold process startup.
    # Keep this before every application import, not merely around DB calls.
    with patch("dotenv.load_dotenv", return_value=False):
        from schemas import Itinerary, TripRequest
        from rules import check_program_depth, check_rules
        import web_service
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("route_problems") != [] or report.get("validation_problems") != []
            or report.get("skeleton", {}).get("status") not in ("ready", "limited_program")):
        raise ValueError("The source replay must have passed both route and final validation.")
    request = TripRequest.model_validate(report["request"])
    itinerary = Itinerary.model_validate(report["itinerary"])
    credits = {day["day"]: day["depth_credit_min"] for day in report["skeleton"]["days"]}
    problems = check_rules(itinerary, request, allow_limited_program=True)
    problems += check_program_depth(itinerary, request, credits, allow_limited_program=True)
    if problems:
        raise ValueError("The saved replay no longer passes local rules: " + "; ".join(problems))
    trip = {"id": 1, "request": request.model_dump(), "itinerary": itinerary.model_dump(),
            "destination": request.destination, "start_date": request.start_date, "end_date": request.end_date,
            "created_at": datetime.now(timezone.utc).isoformat(), "revision": 1}
    with ExitStack() as stack:
        # Explicit fixture boundary: serialization only. Any unexpected DB
        # access is a failure, not an accidental read/write of the user's data.
        stack.enter_context(patch.object(web_service.db, "connect", side_effect=AssertionError("No DB in preview")))
        stack.enter_context(patch.object(web_service, "_validated_trip", return_value=(trip, request, itinerary)))
        stack.enter_context(patch.object(web_service.db, "get_checklist", return_value=[]))
        stack.enter_context(patch.object(web_service.db, "get_reminder_statuses", return_value=[]))
        detail = web_service.get_trip_detail(0, 1)
        summary = web_service._summary_from_trip(trip)
    detail["reminders_text"] = "Контрольный просмотр: напоминания не создаются и не отправляются."
    return trip, detail, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--export-pdf", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("Use an unprivileged local port.")
    trip, detail, summary = preview_data(args.report)
    from pdf_export import build_trip_pdf
    pdf = build_trip_pdf(trip, [])
    if args.export_pdf:
        pdf_path = args.export_pdf.resolve()
        if not pdf_path.is_relative_to(ROOT / "output") or pdf_path.exists():
            raise ValueError("Choose a fresh PDF path inside output/.")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with pdf_path.open("xb") as stream:
            stream.write(pdf)
    web = ROOT / "web"
    banner = ('<aside style="padding:16px;background:#fff1cc;color:#382c10;font:16px sans-serif">'
              '<strong>Контрольный просмотр</strong> — пример из сохранённого проверенного набора. '
              'Не новая AI-генерация. Ваши поездки и квота не изменены; доступно только чтение.</aside>')
    preview_style = ('<style>#newTripButton,#editDayButton,#logoutButton,.trip-header-actions button'
                     '{display:none!important}</style>')
    html = (web / "index.html").read_text(encoding="utf-8").replace("</head>", preview_style + "</head>")
    html = html.replace("<body>", "<body>" + banner).encode("utf-8")
    payloads = {
        "/api/me": {"name": "Контрольный просмотр"},
        "/api/trips": {"trips": [summary]},
        "/api/trips/1": detail,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path in ("/", "/index.html"):
                return self.send(html, "text/html; charset=utf-8")
            if path in payloads:
                return self.send(json.dumps(payloads[path], ensure_ascii=False).encode("utf-8"),
                                 "application/json; charset=utf-8")
            if path == "/api/trips/1/pdf":
                return self.send(pdf, "application/pdf")
            if path.startswith("/static/"):
                asset = (web / path.lstrip("/")).resolve()
                if asset.is_relative_to(web / "static") and asset.is_file() and not asset.is_symlink():
                    return self.send(asset.read_bytes(), mimetypes.guess_type(asset)[0] or "application/octet-stream")
            self.send(b"Not found", "text/plain", 404)

        def do_POST(self):
            self.send(json.dumps({"detail": "Это контрольный просмотр: изменения отключены."},
                                 ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", 405)

        do_PUT = do_PATCH = do_DELETE = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Read-only preview: http://127.0.0.1:{args.port}/#trip/1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
