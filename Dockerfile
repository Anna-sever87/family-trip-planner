FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core fonts-wqy-zenhei \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home --home-dir /home/planner planner \
    && mkdir -p /app /data/output \
    && ln -s /data/output /app/output \
    && chown planner:planner /data /data/output

# Source-relative and working-directory output paths share the writable volume.
# In particular, bot and web must use the same persistent Nominatim request clock.

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY access_control.py access_guard.py anketa.py bot.py candidate_grounding.py checklist.py city_candidates.py city_coverage.py city_estimates.py city_planner.py city_quality.py city_response.py city_schedule.py db.py destination_name_hints.py destination_scope.py formatting.py generation_feedback.py grounded_response.py grounded_schedule.py memory.py pdf_export.py privacy.py \
     places.py place_labels.py planner.py pricing.py program_coverage.py prompts.py regional_discovery.py regional_schedule_witness.py regional_routing.py regional_skeleton.py reminders.py rules.py schemas.py source_planning.py visit_facts.py \
     verified_map_places.py web_app.py web_auth.py web_service.py weather.py yandex_ai.py city_visit_estimates.py city_rollout.py city_context.py city_edit.py city_edit_service.py city_deterministic.py city_visit_options.py city_preferences.py city_evidence.py generation_jobs.py generation_quality.py cover_prompts.py cover_subjects.py yandex_art.py trip_covers.py trip_deletion.py /app/
COPY web/ /app/web/
COPY data/verified_places.json /app/data/verified_places.json
COPY data/destination_name_hints.json /app/data/destination_name_hints.json
COPY data/visit_facts.json /app/data/visit_facts.json
COPY data/verified_map_places.json /app/data/verified_map_places.json

WORKDIR /data
USER planner
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
