"""Local programme permutations: isolated SQLite, no AI, memory or HTTP."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import json
import sqlite3
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

with patch('dotenv.load_dotenv', return_value=False):
    import db
    import trip_covers
    import web_app
    import web_auth
    import web_service

from fastapi.testclient import TestClient
from city_candidates import CityCandidate
from city_context import attach_city_presentation, build_city_context, load_city_context
import city_evidence
import city_visit_options
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def fixture(days=5):
    start = date(2099, 6, 1)
    request = TripRequest(destination='Тестгород, Россия', start_date=start.isoformat(),
                          end_date=(start + timedelta(days=days - 1)).isoformat(),
                          budget=100000, travel_cost=2000, lodging_cost=3000, adults=2,
                          start_time='09:00', end_time='18:00', pace='спокойный')
    center = {'name': 'Тестгород', 'lat': 55., 'lon': 83., 'country_code': 'ru', 'country': 'Россия'}
    candidates, programme = [], []
    for index in range(days):
        title = f'Городской музей истории {index + 1}'
        query = f'{title}, Тестгород, Россия'
        lat = 55. + index * .0001
        record = {'name': query, 'lat': lat, 'lon': 83., 'osm_type': 'node', 'osm_id': 100 + index,
                  'provider': 'osm', 'category': 'tourism', 'type': 'museum',
                  'address': {'city': 'Тестгород', 'country': 'Россия', 'country_code': 'ru'}}
        candidates.append(CityCandidate(f'osm-node-{100 + index}', title, query, 'Тестгород', record))
        programme.append(ItineraryDay(day=index + 1, date=(start + timedelta(days=index)).isoformat(),
            theme=f'Программа {index + 1}', rainy_alternative='', items=[
                ItineraryItem(time='10:00', title=title, place_query=query, area='Тестгород',
                              description='Осмотр экспозиции.', duration_min=90, cost=0., indoor=True,
                              verified=True, lat=lat, lon=83., official_name=query, provider='osm'),
                ItineraryItem(time='12:00', title='Обед', kind='meal', description='Резерв на обед.', duration_min=60, cost=5000.),
                ItineraryItem(time='14:00', title='Отдых', kind='rest', description='Пауза.', duration_min=30, cost=0.),
            ]))
    itinerary = Itinerary(title='Синтетическая поездка', summary='Сохранить текст', days=programme,
                          city_schedule=True, tips=['Сохранить совет']).model_dump()
    for index, day in enumerate(itinerary['days']):
        day['private_extension'] = {'retain': index}
    itinerary['private_extension'] = {'retain': 'whole itinerary'}
    return {'request': {**request.model_dump(), 'city_map_context': build_city_context(request, candidates, center),
                        'weather_text': 'Сохранённая погода', 'weather': {'source': 'history', 'days': []},
                        'generation_coverage': {'quality_status': 'partial', 'selection_mode': 'memory_ranking',
                                                'memory_ranking_status': 'applied'}, 'private_extension': [1, 2, 3]},
            'itinerary': itinerary}


class DaySwapTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='day-swap-')))
        self.stack.enter_context(patch.object(db, 'DB_PATH', str(self.directory / 'synthetic.db')))
        self.stack.enter_context(patch.dict('os.environ', {
            'ACCESS_MODE': 'open', 'WEB_COOKIE_SECURE': '0', 'WEB_BASE_URL': 'http://testserver',
            'WEB_SESSION_SECRET': 'synthetic-swap-session-1234567890', 'DAILY_LIMIT': '3',
        }))
        db.init_db(); trip_covers.init_schema()
        for owner in (1, 2):
            db.save_user(owner, 'Synthetic', 2, [], 'RUB')
        self.account = db.get_account_key(1)
        self.fixture = fixture()
        self.trip = db.save_trip(1, self.fixture['request'], self.fixture['itinerary'])
        self.other = db.save_trip(2, self.fixture['request'], self.fixture['itinerary'])
        db.save_checklist(self.trip, [('Synthetic', 'Preserve checklist')])
        self.forbidden = [self.stack.enter_context(patch.object(module, name,
            side_effect=AssertionError('No quota, memory, AI, image or map calls')))
            for module, name in ((db, 'claim_ai_request'), (web_service, 'generate_itinerary'),
                                (web_service, 'edit_itinerary'), (web_service, 'resolve_city_for_planning'),
                                (web_service, '_enqueue_trip_cover_safely'),
                                (web_service.vector_memory, 'recall_family_memories'),
                                (web_service.city_context, 'prepare_stored_city_context'))]
        self.stack.enter_context(patch('places.geocode', side_effect=AssertionError('No map lookup')))
        self.stack.enter_context(patch('httpx.get', side_effect=AssertionError('No HTTP')))

    def snapshot(self):
        with db.connect() as conn:
            return {row[0]: [tuple(item) for item in conn.execute(f'SELECT * FROM "{row[0]}" ORDER BY 1')]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")}

    def raw(self, trip=None):
        with db.connect() as conn:
            return dict(conn.execute('SELECT * FROM trips WHERE id=?', (self.trip if trip is None else trip,)).fetchone())

    def swap(self, a=1, b=5, revision=1, trip=None, account=None):
        return web_service.swap_trip_days(1, self.trip if trip is None else trip, a, b, revision,
                                          expected_account_key=self.account if account is None else account)

    def save(self, data):
        return db.save_trip(1, data['request'], data['itinerary'])

    def client(self, owner=1):
        client = TestClient(web_app.app)
        self.addCleanup(client.close)
        if owner is not None:
            token, csrf, _ = web_auth.create_session(owner)
            client.cookies.set(web_auth.SESSION_COOKIE, token)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            client.headers['X-CSRF-Token'] = csrf
        return client

    def test_first_and_fifth_programmes_swap_calendar_and_all_other_data_are_preserved(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO trip_covers(trip_id,account_key,cache_key,status,file_name,created_at,subject_label) "
                         "VALUES(?,?,?,'ready',?,?,?)", (self.trip, self.account, 'existing', 'synthetic.png', db._now(), 'Музей'))
            conn.execute("INSERT INTO reminder_deliveries(trip_id,kind,status,claimed_at,sent_at) "
                         "VALUES(?,'day1','sent','now','now')", (self.trip,))
        before, raw = self.snapshot(), self.raw()
        result = self.swap()
        self.assertEqual((result['id'], result['revision'], result['changed'], result['day_a'], result['day_b']),
                         (self.trip, 2, True, 1, 5))
        after, updated = self.snapshot(), self.raw()
        self.assertEqual({k: v for k, v in before.items() if k != 'trips'},
                         {k: v for k, v in after.items() if k != 'trips'})
        self.assertEqual({k: v for k, v in raw.items() if k not in {'itinerary_json', 'revision'}},
                         {k: v for k, v in updated.items() if k not in {'itinerary_json', 'revision'}})
        old, new = json.loads(raw['itinerary_json']), json.loads(updated['itinerary_json'])
        self.assertEqual({k: v for k, v in old.items() if k != 'days'}, {k: v for k, v in new.items() if k != 'days'})
        self.assertEqual([(d['day'], d['date']) for d in old['days']], [(d['day'], d['date']) for d in new['days']])
        for left, right in ((0, 4), (1, 1), (2, 2), (3, 3), (4, 0)):
            self.assertEqual({k: v for k, v in old['days'][right].items() if k not in {'day', 'date'}},
                             {k: v for k, v in new['days'][left].items() if k not in {'day', 'date'}})
        for method in self.forbidden:
            method.assert_not_called()

    def test_swap_back_preserves_exact_original_programme_and_advances_revision(self):
        before = db.get_trip(self.trip, 1)['itinerary']
        self.swap(); self.swap(revision=2)
        self.assertEqual(db.get_trip(self.trip, 1)['itinerary'], before)
        self.assertEqual(self.raw()['revision'], 3)

    def test_any_two_nonadjacent_days_work_and_pdf_uses_current_row(self):
        self.swap(2, 4)
        detail = web_service.get_trip_detail(1, self.trip)
        self.assertEqual(detail['itinerary']['days'][1]['theme'], 'Программа 4')
        self.assertEqual(detail['itinerary']['days'][3]['theme'], 'Программа 2')
        self.assertEqual(detail['budget']['planned_total'], 30000)
        with patch.object(web_service, 'build_trip_pdf', return_value=b'synthetic-pdf') as build:
            self.assertEqual(web_service.get_trip_pdf(1, self.trip), b'synthetic-pdf')
        self.assertEqual(build.call_args.args[0]['itinerary']['days'][1]['theme'], 'Программа 4')
        self.assertEqual(build.call_args.args[1], db.get_checklist(self.trip))

    def test_day_range_types_and_same_day_reject_without_mutation(self):
        for a, b in ((0, 2), (1, 6), (-1, 2), (1, 1), (True, 2), ('1', 2), (1, 1.5)):
            before = self.snapshot()
            with self.subTest(a=a, b=b), self.assertRaises(web_service.DaySwapError):
                self.swap(a, b)
            self.assertEqual(self.snapshot(), before)

    def test_owner_account_stale_revision_and_deleting_account_fail_closed(self):
        for args, error in (({'trip': self.other}, web_service.TripNotFoundError),
                            ({'account': 'replaced-account'}, web_service.TripNotFoundError),
                            ({'revision': 2}, web_service.EditConflictError)):
            before = self.snapshot()
            with self.subTest(args=args), self.assertRaises(error):
                self.swap(**args)
            self.assertEqual(self.snapshot(), before)
        with db.connect() as conn:
            conn.execute("UPDATE users SET deletion_token='deleting' WHERE tg_id=1")
        before = self.snapshot()
        with self.assertRaises(web_service.TripNotFoundError):
            self.swap()
        self.assertEqual(self.snapshot(), before)

    def test_concurrent_same_revision_requests_commit_only_once(self):
        def attempt():
            try:
                return self.swap()['revision']
            except web_service.EditConflictError:
                return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        self.assertCountEqual(results, [2, 'conflict'])
        self.assertEqual(self.raw()['revision'], 2)

    def test_final_cas_failure_and_sql_failure_roll_back_without_touching_data(self):
        original_connect = db.connect
        for mode in ('cas', 'sql'):
            @contextmanager
            def connection():
                with original_connect() as conn:
                    class Proxy:
                        def execute(self, sql, values=()):
                            if sql.startswith('UPDATE trips SET itinerary_json='):
                                if mode == 'cas':
                                    return SimpleNamespace(rowcount=0)
                                conn.execute(sql, values)
                                raise sqlite3.OperationalError('Synthetic failure after the UPDATE')
                            return conn.execute(sql, values)
                    yield Proxy()
            before = self.snapshot()
            expected = web_service.EditConflictError if mode == 'cas' else sqlite3.OperationalError
            with self.subTest(mode=mode), patch.object(db, 'connect', side_effect=connection), self.assertRaises(expected):
                self.swap()
            self.assertEqual(self.snapshot(), before)

    def test_each_busy_ledger_blocks_swap_without_changing_any_table(self):
        inserts = (
            ('ai_requests', "INSERT INTO ai_requests(claim_id,tg_id,usage_date,kind,source,status,claimed_at) VALUES('busy',1,'2099-06-01','plan','web','running','now')", ()),
            ('generation_jobs', "INSERT INTO generation_jobs(job_id,tg_id,account_key,idempotency_key,request_digest,status,created_at) VALUES(?,1,?,?,'digest','queued','now')", (str(uuid4()), self.account, str(uuid4()))),
            ('trip_covers', "INSERT INTO trip_covers(trip_id,account_key,cache_key,status,created_at) VALUES(?,?,'cache','running','now')", (self.trip, self.account)),
            ('reminder_deliveries', "INSERT INTO reminder_deliveries(trip_id,kind,status,claimed_at) VALUES(?,'day1','sending','now')", (self.trip,)),
        )
        for table, sql, params in inserts:
            with self.subTest(table=table):
                with db.connect() as conn:
                    conn.execute(sql, params)
                before = self.snapshot()
                with self.assertRaises(web_service.EditBusyError):
                    self.swap()
                self.assertEqual(self.snapshot(), before)
                with db.connect() as conn:
                    conn.execute(f'DELETE FROM "{table}"')

    def test_source_date_or_hours_conflict_rejects_original_is_kept(self):
        original = city_visit_options.visit_option
        for mode in ('date', 'time'):
            def option(candidate, request, **kwargs):
                value = original(candidate, request, **kwargs)
                if candidate.candidate_id != 'osm-node-100':
                    return value
                dates = (request.start_date,) if mode == 'date' else tuple(day['date'] for day in self.fixture['itinerary']['days'])
                windows = tuple({'date': when, 'start': '09:00', 'end': '18:00' if when == request.start_date else '10:30'} for when in dates)
                return replace(value, allowed_dates=dates, opening_windows=windows)
            before = self.snapshot()
            with self.subTest(mode=mode), patch.object(city_visit_options, 'visit_option', side_effect=option):
                web_service._swapped_city_plan(self.fixture, 2, 3)
                with self.assertRaisesRegex(web_service.DaySwapError, 'новой дате'):
                    self.swap()
            self.assertEqual(self.snapshot(), before)

    def test_managed_source_window_tips_follow_new_dates_without_rewriting_other_text_or_evidence(self):
        data = deepcopy(self.fixture)
        request = TripRequest.model_validate(data['request'])
        candidates, _ = load_city_context(data['request']['city_map_context'], request)
        original_option = city_visit_options.visit_option
        dates = tuple(day['date'] for day in data['itinerary']['days'])
        def option(candidate, request, **kwargs):
            value = original_option(candidate, request, **kwargs)
            return replace(value, allowed_dates=dates, opening_windows=tuple(
                {'date': when, 'start': '09:00', 'end': '18:00'} for when in dates))
        with patch.object(city_visit_options, 'visit_option', side_effect=option), \
                patch.object(city_evidence, 'visit_option', side_effect=option):
            itinerary = Itinerary.model_validate(data['itinerary'])
            user_tips = ['Личный совет до сведений', 'Сведения о посещении [not-owned]. Личный текст', 'Личный совет после сведений']
            data['itinerary']['tips'] = [user_tips[0], *city_evidence.city_managed_tips(itinerary, candidates, request), *user_tips[1:]]
            itinerary = Itinerary.model_validate(data['itinerary'])
            data['request']['city_map_context'] = attach_city_presentation(data['request']['city_map_context'], itinerary, candidates, request)
            trip = self.save(data)
            before = db.get_trip(trip, 1)
            self.swap(trip=trip)
            after = db.get_trip(trip, 1)
            updated = Itinerary.model_validate(after['itinerary'])
            owned = city_evidence.city_managed_tips(updated, candidates, request)
            self.assertEqual([value for value in after['itinerary']['tips'] if value not in owned], user_tips)
            first_visit = next(value for value in owned if '[osm-node-100]' in value)
            self.assertIn('Учтённые окна на 2099-06-05:', first_visit)
            self.assertNotIn('Учтённые окна на 2099-06-01:', first_visit)
            old_context, new_context = before['request']['city_map_context'], after['request']['city_map_context']
            self.assertNotEqual(old_context['presentation'], new_context['presentation'])
            self.assertEqual({k: v for k, v in old_context.items() if k != 'presentation'},
                             {k: v for k, v in new_context.items() if k != 'presentation'})
            self.assertEqual({k: v for k, v in before['request'].items() if k != 'city_map_context'},
                             {k: v for k, v in after['request'].items() if k != 'city_map_context'})
            self.assertEqual(after['itinerary']['summary'], before['itinerary']['summary'])
            for slot, source in ((0, 4), (4, 0)):
                self.assertEqual(after['itinerary']['days'][slot]['items'], before['itinerary']['days'][source]['items'])

    def test_saved_forecast_stays_on_calendar_dates_and_blocks_new_outdoor_rain_day(self):
        data = deepcopy(self.fixture)
        data['request']['weather'] = {'source': 'forecast', 'days': [{'date': '2099-06-01', 'rain_mm': 2.}]}
        data['itinerary']['days'][4]['items'][0]['indoor'] = False
        trip = self.save(data)
        before = self.snapshot()
        with self.assertRaisesRegex(web_service.DaySwapError, 'погодных'):
            self.swap(trip=trip)
        self.assertEqual(self.snapshot(), before)

    def test_full_validation_does_not_accept_preexisting_too_short_adult_medium_programme(self):
        data = deepcopy(self.fixture)
        old_request = TripRequest.model_validate(data['request'])
        candidates, center = load_city_context(data['request']['city_map_context'], old_request)
        data['request']['pace'] = 'средний'
        updated_request = TripRequest.model_validate(data['request'])
        data['request']['city_map_context'] = build_city_context(updated_request, candidates, center)
        trip = self.save(data)
        before = self.snapshot()
        with self.assertRaisesRegex(web_service.DaySwapError, 'не проходит проверку'):
            self.swap(trip=trip)
        self.assertEqual(self.snapshot(), before)

    def test_malformed_saved_weather_rejects_instead_of_ignoring_real_forecast(self):
        for observations in (None, ['not a record'], [{'date': '2099-06-01', 'rain_mm': '2'}],
                             [{'date': 'wrong date', 'rain_mm': 2.}], [{'date': '2099-06-01', 'rain_mm': True}]):
            data = deepcopy(self.fixture)
            data['request']['weather'] = {'source': 'forecast', 'days': observations}
            trip = self.save(data)
            before = self.snapshot()
            with self.subTest(observations=observations), self.assertRaises(web_service.DaySwapError):
                self.swap(trip=trip)
            self.assertEqual(self.snapshot(), before)

    def test_unrelated_users_busy_request_does_not_lock_owner_and_exported_file_is_untouched(self):
        document = self.directory / f'trip_{self.trip}.pdf'
        document.write_bytes(b'An already downloaded document is not the live PDF cache.')
        with db.connect() as conn:
            conn.execute("INSERT INTO ai_requests(claim_id,tg_id,usage_date,kind,source,status,claimed_at) "
                         "VALUES('other-busy',2,'2099-06-01','plan','web','running','now')")
        before = document.read_bytes()
        self.swap()
        self.assertEqual(document.read_bytes(), before)

    def test_invalid_place_calendar_or_manifest_cannot_be_accepted_as_old_evidence(self):
        for kind in ('foreign_context', 'coordinates', 'date', 'number', 'length', 'manifest', 'source_mix'):
            data = deepcopy(self.fixture)
            if kind == 'foreign_context':
                data['request']['city_map_context']['resolved_city']['country_code'] = 'cn'
            elif kind == 'coordinates':
                data['itinerary']['days'][0]['items'][0]['lat'] += .1
            elif kind == 'date':
                data['itinerary']['days'][0]['date'] = '2099-06-02'
            elif kind == 'number':
                data['itinerary']['days'][0]['day'] = 2
            elif kind == 'length':
                data['itinerary']['days'].pop()
            elif kind == 'manifest':
                data['request']['city_map_context']['presentation'] = {'version': 1, 'managed_tip_sha256': ['a' * 64]}
            else:
                data['itinerary']['source_schedule'] = True
            trip = self.save(data)
            before = self.snapshot()
            with self.subTest(kind=kind), self.assertRaises(web_service.DaySwapError):
                self.swap(trip=trip)
            self.assertEqual(self.snapshot(), before)

    def test_capability_reasons_for_single_day_region_and_missing_legacy_context(self):
        self.assertTrue(web_service.get_trip_detail(1, self.trip)['day_swap_available'])
        cases = [('one', fixture(1)), ('region', deepcopy(self.fixture)), ('legacy', deepcopy(self.fixture))]
        cases[1][1]['request']['destination_kind'] = 'region'
        cases[1][1]['itinerary']['city_schedule'] = False
        cases[1][1]['itinerary']['source_schedule'] = True
        cases[2][1]['request'].pop('city_map_context')
        for name, data in cases:
            trip = self.save(data)
            with self.subTest(case=name):
                capability = web_service.get_trip_detail(1, trip)
                self.assertFalse(capability['day_swap_available'])
                self.assertTrue(capability['day_swap_unavailable_reason'])
                if name == 'region':
                    self.assertIn('пересчитать переезды и смену жилья', capability['day_swap_unavailable_reason'])
                before = self.snapshot()
                with self.assertRaises(web_service.DaySwapError):
                    self.swap(trip=trip)
                self.assertEqual(self.snapshot(), before)

    def test_free_and_program_days_swap_without_changing_report_or_totals(self):
        data = deepcopy(self.fixture)
        last = data['itinerary']['days'][4]
        last['day_mode'] = 'free'; last['items'] = last['items'][1:]
        data['itinerary']['program_report'] = {'schema_version': 1, 'reason': 'limited_verified_program',
            'coverage_basis': 'map', 'requested_days': 5, 'program_days': 4, 'free_days': 1,
            'eligible_unique_visits': 5, 'selected_unique_visits': 4}
        trip = self.save(data)
        self.swap(trip=trip)
        updated = db.get_trip(trip, 1)
        self.assertEqual(updated['itinerary']['program_report'], data['itinerary']['program_report'])
        self.assertEqual(updated['request'], data['request'])
        self.assertEqual([day['day_mode'] for day in updated['itinerary']['days']], ['free', 'program', 'program', 'program', 'program'])
        self.assertTrue(web_service.get_trip_detail(1, trip)['day_swap_available'])

    def test_http_requires_session_csrf_owner_valid_body_and_revision(self):
        url = f'/api/trips/{self.trip}/days/swap'
        payload = {'day_a': 1, 'day_b': 5, 'expected_revision': 1}
        self.assertEqual(self.client(None).patch(url, json=payload).status_code, 401)
        self.assertEqual(self.client(2).patch(url, json=payload).status_code, 404)
        client = self.client()
        csrf = client.headers.pop('X-CSRF-Token')
        self.assertEqual(client.patch(url, json=payload).status_code, 403)
        client.headers['X-CSRF-Token'] = csrf
        with db.connect() as conn:
            conn.execute("INSERT INTO reminder_deliveries(trip_id,kind,status,claimed_at) VALUES(?,'day1','sending','now')",
                         (self.trip,))
        self.assertEqual(client.patch(url, json=payload).status_code, 409)
        with db.connect() as conn:
            conn.execute('DELETE FROM reminder_deliveries WHERE trip_id=?', (self.trip,))
        for changes in ({'day_a': True}, {'day_a': '1'}, {'day_b': 15}, {'day_b': 1},
                        {'expected_revision': 0}, {'expected_revision': True}, {'extra': 'forbidden'}):
            before = self.raw()
            response = client.patch(url, json={**payload, **changes})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.raw(), before)
        response = client.patch(url, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['revision'], 2)
        self.assertEqual(client.patch(url, json=payload).status_code, 409)
        self.assertEqual(client.get(f'/api/trips/{self.trip}').json()['itinerary']['days'][0]['theme'], 'Программа 5')


if __name__ == '__main__':
    unittest.main()
