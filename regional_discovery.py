"""Bounded map-only regional reserve, never proof of hours or transport.

Only administrative regions use this reserve. The original full polygon,
country, provider identity, actual address and radius remain independent gates.
No AI, database writes or invented visitor facts. Nominatim calls share its
existing cross-process clock; each reserve call is charged to the caller's
existing HTTP ceiling, with no retries.
"""
from __future__ import annotations

import math

import httpx

import places
from destination_scope import _distance_from_bounds, _valid_geometry

ADMIN_TYPES = frozenset({'administrative', 'state', 'region', 'county'})
MAX_RESERVE_CALLS = 8
MAX_TOWN_RESULTS = 40
MAX_BYTES = 1_000_000


def _issue(result, code):
    counts = getattr(result, 'regional_discovery_issues', None)
    if not isinstance(counts, dict):
        counts = {}
        result.regional_discovery_issues = counts
    counts[code] = counts.get(code, 0) + 1


def administrative_scope(scope):
    return (scope.area_type in ADMIN_TYPES and _valid_geometry(scope.geometry)
            and len(places._country_identity(scope.country)) == 1
            and len(places._country_identity(scope.country)[0]) == 2)


def inside_admin(record, scope):
    """Country + original polygon + explicit mapped administrative label."""
    if not administrative_scope(scope) or not isinstance(record, dict):
        return False
    address = record.get('address')
    if not isinstance(address, dict):
        return False
    if places._country_identity(str(address.get('country_code', ''))) != places._country_identity(scope.country):
        return False
    if address.get('country') and places._country_identity(str(address['country'])) != places._country_identity(scope.country):
        return False
    name = places._context_identity(scope.name.split(',', 1)[0])
    labels = {places._context_identity(str(address.get(field, ''))) for field in ('state', 'region', 'county')}
    if not name or name not in labels:
        return False
    try:
        if any(isinstance(record.get(key), bool) for key in ('lat', 'lon')):
            return False
        lat, lon = float(record['lat']), float(record['lon'])
        return (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180
                and _distance_from_bounds(lat, lon, scope) == 0)
    except (TypeError, ValueError, KeyError, OverflowError):
        return False


def _reserve(result):
    if result.reserved_http_attempts_upper_bound >= result.http_attempt_limit:
        return False
    if result.regional_discovery_calls >= MAX_RESERVE_CALLS:
        return False
    result.reserved_http_attempts_upper_bound += 1
    result.regional_discovery_calls += 1
    return True


def _admin_centre(region_id, scope, result, near_bases):
    """Read one observed region relation, then the exact referenced centre ID.

    OSM element GET is read-only (API v0.6); this is not a geographic bulk
    download and neither endpoint receives personal trip/profile information.
    """
    from city_candidates import _from_provider, _provider_title, _MAP_LOOKUP_URL
    try:
        response = httpx.get(f'https://api.openstreetmap.org/api/0.6/relation/{region_id}.json',
                             headers={'User-Agent':places.USER_AGENT}, timeout=15)
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            return None
        payload = response.json()
        rows = payload.get('elements') if isinstance(payload,dict) else None
        if not isinstance(rows,list) or len(rows) != 1:
            return None
        relation = rows[0]
        if (not isinstance(relation,dict) or relation.get('type') != 'relation'
                or type(relation.get('id')) is not int or relation['id'] != region_id
                or relation.get('tags',{}).get('boundary') != 'administrative'):
            return None
        members = relation.get('members')
        if not isinstance(members,list) or len(members) > 10000:
            return None
        centres = sorted({member['ref'] for member in members if isinstance(member,dict)
                          and member.get('role') == 'admin_centre' and member.get('type') == 'node'
                          and type(member.get('ref')) is int and member['ref'] > 0})
        if len(centres) != 1 or not _reserve(result):
            return None
        with places._wait_for_turn():
            response = httpx.get(_MAP_LOOKUP_URL, params={'osm_ids':f'N{centres[0]}',
                'format':'json','addressdetails':1,'namedetails':1,'accept-language':'ru,en'},
                headers={'User-Agent':places.USER_AGENT}, timeout=15)
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            return None
        rows = response.json()
        if not isinstance(rows,list) or len(rows) != 1:
            return None
        record = _from_provider(rows[0]); title = _provider_title(record)
        if (places.map_record_identity(record) != ('osm','node',centres[0])
                or record.get('category') not in {'place','boundary'}
                or not ({record.get('type'),record.get('addresstype')} & {'city','town'})
                or not title or not places._place_matches_query(f'{title}, {scope.country}',record)
                or not inside_admin(record,scope)
                or near_bases and not any(places.distance_km(base['lat'],base['lon'],float(record['lat']),float(record['lon'])) <= 100
                                          for base in near_bases)):
            return None
        return {**record,'base_query':f'{title}, {scope.country}'}
    except (httpx.HTTPError,places.GeocodeUnavailable,ValueError,TypeError,KeyError,OverflowError,AttributeError):
        _issue(result,'admin_centre_unavailable')
        return None


def discover_town(scope, result, *, near_bases=()):
    """One exact admin-centre chain, or bounded town-list + two ID lookups.

    All branches share the eight-call reserve and the original region's
    remaining HTTP ceiling. No invented town name or unverified coordinates.
    """
    from city_candidates import _from_provider, _provider_title, _SHOPPING_DISCOVERY_URL, _MAP_LOOKUP_URL
    if not administrative_scope(scope) or not _reserve(result):
        return None
    south, north, west, east = scope.bounds
    # An administrative centre is a provider relation member, not a guessed
    # capital. Bind that relation to the SAME cached full geometry which was
    # resolved for this request. Unbound/legacy scopes use the bounded list.
    region_id = None
    try:
        cached = places._load_cache()
    except (OSError, ValueError, TypeError):
        cached = {}
    for record in (cached.values() if isinstance(cached, dict) else ()):
        if (isinstance(record, dict) and record.get('geojson') == scope.geometry
                and record.get('name') == scope.name and isinstance(record.get('type'),str) and record['type'] in ADMIN_TYPES
                and places._place_matches_query(f"{scope.name.split(',',1)[0]}, {scope.country}", record)):
            identity = places.map_record_identity(record)
            if identity and identity[:2] == ('osm', 'relation'):
                region_id = identity[2]
                break
    if region_id is not None:
        centre = _admin_centre(region_id,scope,result,near_bases)
        if centre is not None:
            return centre
        if not _reserve(result):
            return None
    query = ('[out:json][timeout:12][maxsize:67108864];' +
             f'nwr["place"~"^(city|town)$"]["name"]({south},{west},{north},{east});' +
             f'out body center {MAX_TOWN_RESULTS};')
    try:
        response = httpx.get(_SHOPPING_DISCOVERY_URL, params={'data': query},
                             headers={'User-Agent': places.USER_AGENT}, timeout=15)
        response.raise_for_status()
        if len(response.content) > MAX_BYTES:
            _issue(result, 'town_response_oversized')
            return None
        payload = response.json()
        rows = payload.get('elements') if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or payload.get('remark') or not isinstance(rows, list) or len(rows) > MAX_TOWN_RESULTS:
            _issue(result, 'town_response_incomplete')
            return None
        candidates = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            tags = row.get('tags', {})
            coords = row if row.get('type') == 'node' else row.get('center', {})
            title = tags.get('name') if isinstance(tags, dict) else None
            if (not isinstance(title, str) or not 1 <= len(title) <= 80 or not title.isprintable()
                    or title != title.strip() or ',' in title or tags.get('place') not in {'city', 'town'}
                    or row.get('type') not in {'node', 'way', 'relation'}
                    or type(row.get('id')) is not int or row['id'] <= 0
                    or not isinstance(coords, dict)):
                continue
            try:
                lat, lon = coords['lat'], coords['lon']
                if any(type(value) not in (int, float) or not math.isfinite(value) for value in (lat, lon)):
                    continue
                if not -90 <= lat <= 90 or not -180 <= lon <= 180 or _distance_from_bounds(lat, lon, scope) != 0:
                    continue
                if near_bases and not any(places.distance_km(base['lat'],base['lon'],lat,lon) <= 100 for base in near_bases):
                    continue
                population = str(tags.get('population', '0')).replace(' ', '')
                population = int(population) if population.isdigit() and len(population) <= 10 else 0
            except (KeyError, ValueError, OverflowError):
                continue
            candidates.append((tags['place'] != 'city', -population, title, row['type'], row['id']))
        for _, _, title, kind, osm_id in sorted(candidates)[:2]:
            if not _reserve(result):
                return None
            with places._wait_for_turn():
                response = httpx.get(_MAP_LOOKUP_URL, params={
                    'osm_ids': {'node':'N', 'way':'W', 'relation':'R'}[kind] + str(osm_id),
                    'format':'json', 'addressdetails':1, 'namedetails':1, 'accept-language':'ru,en'},
                    headers={'User-Agent':places.USER_AGENT}, timeout=15)
            response.raise_for_status()
            if len(response.content) > MAX_BYTES:
                return None
            data = response.json()
            if not isinstance(data, list) or len(data) != 1:
                continue
            record = _from_provider(data[0])
            actual = _provider_title(record)
            if (places.map_record_identity(record) != ('osm', kind, osm_id)
                    or record.get('category') not in {'place', 'boundary'}
                    or not ({record.get('type'), record.get('addresstype')} & {'city','town'})
                    or not actual or not places._place_matches_query(f'{title}, {scope.country}', record)
                    or not places._place_matches_query(f'{actual}, {scope.country}', record)
                    or not inside_admin(record, scope)
                    or near_bases and not any(places.distance_km(base['lat'],base['lon'],float(record['lat']),float(record['lon'])) <= 100
                                              for base in near_bases)):
                continue
            return {**record, 'base_query':f'{actual}, {scope.country}'}
    except (httpx.HTTPError, places.GeocodeUnavailable, ValueError, TypeError, KeyError, OverflowError) as error:
        _issue(result, 'town_http_error' if isinstance(error,httpx.HTTPError) else 'town_response_invalid')
        return None
    _issue(result, 'town_not_confirmed')
    return None


def category_records(request, scope, base, result):
    """Up to three requested named categories around one verified local base."""
    from city_candidates import CityCandidateResult, _CATEGORY_FILTERS, _category_search, _canonical_record
    if not administrative_scope(scope):
        return []
    address = base.get('map_address', {})
    if places._country_identity(str(address.get('country_code', ''))) != places._country_identity(scope.country):
        return []
    center = {'lat':base['lat'], 'lon':base['lon'], 'country_code':address['country_code'].lower()}
    categories = list(dict.fromkeys([category for category in request.interest_categories
                                     if category in _CATEGORY_FILTERS] + ['museums', 'parks', 'main_sights']))[:3]
    records = []
    for category in categories:
        if not _reserve(result):
            break
        scratch = CityCandidateResult()
        for proposal in _category_search(category, scope.country, center, scratch):
            record = _canonical_record(proposal['title'], proposal['record'], center)
            if record is not None and inside_admin(record, scope):
                records.append(record)
        if scratch.diagnostics.get('category_fallback_status') == 'unavailable':
            _issue(result, 'category_search_unavailable')
    return records
