"use strict";

// Offline UI regressions: node --test test_web_ui.js
// Load the real browser code in a VM; no server, database or network is used.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const appSource = fs.readFileSync(path.join(__dirname, "web/static/app.js"), "utf8")
  .replace(/bindEvents\(\);\s*init\(\);\s*$/, "");

class FakeElement {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.hidden = false;
    this.disabled = false;
    this.open = false;
    this.value = "";
    this.textContent = "";
    this.listeners = new Map();
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.style = {};
    this.dataset = {};
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  dispatch(type) {
    const event = { defaultPrevented: false, preventDefault() { this.defaultPrevented = true; } };
    this.listeners.get(type)?.(event);
    return event;
  }

  close() { this.open = false; }
  showModal() { this.open = true; }
  focus() {}
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this[name] = value; }
  reportValidity() { return true; }
  reset() {}
  querySelectorAll() { return this.controls || []; }
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function trip(id, day = 1) {
  return { id, itinerary: { days: [{ day }] }, checklist: [] };
}

function browser() {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new FakeElement());
    return elements.get(id);
  };
  const editControls = ["editInstruction", "submitEdit", "cancelEdit", "closeEditDialog"].map(element);
  element("editForm").controls = editControls;
  element("tripView").hidden = true;
  element("loading").hidden = true;
  const windowEvents = new Map();
  const location = { hash: "#trips" };
  const historyCalls = [];
  const context = vm.createContext({
    document: { getElementById: element, querySelectorAll: () => [], createElement: (tag) => new FakeElement(tag), cookie: "" },
    window: {
      location,
      scrollTo() {},
      addEventListener: (type, listener) => windowEvents.set(type, listener),
    },
    history: {
      pushState: (_state, _title, hash) => { location.hash = hash; historyCalls.push(hash); },
      replaceState: (_state, _title, hash) => { location.hash = hash; historyCalls.push(hash); },
    },
    fetch: () => { throw new Error("Network is forbidden in UI regression tests"); },
    console, URLSearchParams, Headers, setTimeout, clearTimeout, setInterval, clearInterval,
  });
  vm.runInContext(appSource, context, { filename: "web/static/app.js" });
  const state = vm.runInContext("state", context);
  state.me = { name: "Offline user" };
  const refreshTripSummary = context.refreshTripSummary;
  const originalRenderTrip = context.renderTrip;
  const originalRenderChecklist = context.renderChecklist;
  const renders = [];
  const checklistRenders = [];
  const toasts = [];
  let summaryRefreshes = 0;
  let waitStops = 0;
  context.renderTrip = () => {
    assert.ok(state.trip);
    assert.equal(element("tripView").hidden, false, "route must be visible before rendering its map");
    renders.push(state.trip.id);
  };
  context.renderChecklist = () => {
    assert.ok(state.trip, "a checklist must never render without its trip");
    checklistRenders.push(state.trip.id);
  };
  context.refreshTripSummary = async () => { summaryRefreshes += 1; };
  context.toast = (message) => toasts.push(message);
  context.startAIWait = () => {
    context.setLoading(true);
    return () => { waitStops += 1; };
  };
  context.bindEvents();
  return {
    context, state, element, editControls, renders, checklistRenders, toasts, location,
    historyCalls, windowEvents, refreshTripSummary, originalRenderTrip, originalRenderChecklist,
    get summaryRefreshes() { return summaryRefreshes; },
    get waitStops() { return waitStops; },
  };
}

function startEdit(ui, id = 1, day = 2) {
  ui.state.trip = trip(id, day);
  ui.state.selectedDay = day;
  ui.element("tripView").hidden = false;
  ui.context.openEditDialog();
  ui.element("editInstruction").value = "Добавить музей";
  return ui.context.submitEdit({ preventDefault() {} });
}

test("late openTrip response cannot reopen the detail after returning to the list", async () => {
  const ui = browser();
  const response = deferred();
  ui.context.api = () => response.promise;
  const request = ui.context.openTrip(1);
  ui.context.showTrips();
  response.resolve(trip(1));
  await request;
  assert.equal(ui.state.trip, null);
  assert.equal(ui.element("tripsView").hidden, false);
  assert.equal(ui.element("loading").hidden, true);
  assert.deepEqual(ui.renders, []);
  assert.deepEqual(ui.historyCalls, ["#trips"]);
});

test("latest route wins even when two route responses arrive in reverse order", async () => {
  const ui = browser();
  const first = deferred();
  const second = deferred();
  ui.context.api = (url) => url.endsWith("/1") ? first.promise : second.promise;
  const firstRequest = ui.context.openTrip(1);
  const secondRequest = ui.context.openTrip(2);
  second.resolve(trip(2, 3));
  await secondRequest;
  first.resolve(trip(1));
  await firstRequest;
  assert.equal(ui.state.trip.id, 2);
  assert.equal(ui.state.selectedDay, 3);
  assert.deepEqual(ui.renders, [2]);
  assert.deepEqual(ui.historyCalls, ["#trip/2"]);
});

test("stale route errors do not hide the current loading state or return to the list", async () => {
  const ui = browser();
  const first = deferred();
  const second = deferred();
  ui.context.api = (url) => url.endsWith("/1") ? first.promise : second.promise;
  const firstRequest = ui.context.openTrip(1);
  const secondRequest = ui.context.openTrip(2);
  first.reject(Object.assign(new Error("Old trip missing"), { status: 404 }));
  await firstRequest;
  assert.equal(ui.element("loading").hidden, false);
  assert.deepEqual(ui.toasts, []);
  assert.deepEqual(ui.historyCalls, []);
  second.resolve(trip(2));
  await secondRequest;
  assert.equal(ui.state.trip.id, 2);
});

test("edit locks all controls and Cancel/Escape, captures its trip and day, then unlocks", async () => {
  const ui = browser();
  const response = deferred();
  const requests = [];
  ui.context.api = (url, options) => { requests.push([url, options]); return response.promise; };
  const request = startEdit(ui, 7, 2);
  assert.ok(ui.editControls.every((control) => control.disabled));
  ui.element("cancelEdit").dispatch("click");
  ui.element("closeEditDialog").dispatch("click");
  assert.equal(ui.element("editDialog").open, true);
  assert.equal(ui.element("editDialog").dispatch("cancel").defaultPrevented, true);
  ui.state.selectedDay = 3;
  await ui.context.submitEdit({ preventDefault() {} });
  assert.equal(requests.length, 1, "repeat submission must not make another AI request");
  assert.equal(requests[0][0], "/api/trips/7/days/2/edit");
  response.resolve(trip(7, 2));
  await request;
  assert.equal(ui.state.selectedDay, 2);
  assert.match(ui.toasts[0], /День 2 обновлён/);
  assert.equal(ui.element("editDialog").open, false);
  assert.equal(ui.state.editingTrip, false);
  assert.ok(ui.editControls.every((control) => !control.disabled));
  assert.equal(ui.summaryRefreshes, 1);
});

test("browser Back during edit closes its dialog and late success leaves the list intact", async () => {
  const ui = browser();
  const response = deferred();
  ui.context.api = () => response.promise;
  const request = startEdit(ui);
  ui.location.hash = "#trips";
  await ui.windowEvents.get("popstate")();
  assert.equal(ui.element("editDialog").open, false);
  assert.ok(ui.waitStops > 0, "navigation stops the old waiting display");
  assert.equal(ui.state.trip, null);
  response.resolve(trip(1, 2));
  await request;
  assert.equal(ui.state.trip, null);
  assert.deepEqual(ui.renders, []);
  assert.deepEqual(ui.toasts, []);
  assert.equal(ui.summaryRefreshes, 1, "saved server changes still refresh list summaries");
});

test("late edit result cannot overwrite another route or hide its active loader", async () => {
  const ui = browser();
  const edit = deferred();
  const route = deferred();
  ui.context.api = (url) => url.endsWith("/edit") ? edit.promise : route.promise;
  const editRequest = startEdit(ui);
  const routeRequest = ui.context.openTrip(9);
  edit.resolve(trip(1, 2));
  await editRequest;
  assert.equal(ui.element("loading").hidden, false);
  assert.deepEqual(ui.renders, []);
  route.resolve(trip(9));
  await routeRequest;
  assert.equal(ui.state.trip.id, 9);
  assert.deepEqual(ui.renders, [9]);
});

test("reopening the same trip also invalidates the earlier edit display", async () => {
  const ui = browser();
  const edit = deferred();
  const reopened = trip(1, 5);
  ui.context.api = (url) => url.endsWith("/edit") ? edit.promise : Promise.resolve(reopened);
  const request = startEdit(ui);
  await ui.context.openTrip(1);
  edit.resolve(trip(1, 2));
  await request;
  assert.equal(ui.state.trip, reopened);
  assert.equal(ui.state.selectedDay, 5);
  assert.deepEqual(ui.renders, [1]);
});

test("current edit failure stays visible and unlocks the dialog for retry or cancellation", async () => {
  const ui = browser();
  const response = deferred();
  ui.context.api = () => response.promise;
  const request = startEdit(ui);
  response.reject(Object.assign(new Error("Проверка не пройдена"), { status: 422 }));
  await request;
  assert.equal(ui.element("editMessage").textContent, "Проверка не пройдена");
  assert.equal(ui.element("editDialog").open, true);
  assert.ok(ui.editControls.every((control) => !control.disabled));
  ui.element("cancelEdit").dispatch("click");
  assert.equal(ui.element("editDialog").open, false);
  assert.equal(ui.element("editDialog").dispatch("cancel").defaultPrevented, false);
});

test("late edit error does not write an error into a different view", async () => {
  const ui = browser();
  const response = deferred();
  ui.context.api = () => response.promise;
  const request = startEdit(ui);
  ui.context.showTrips();
  ui.element("editMessage").textContent = "new view";
  response.reject(new Error("Old network failure"));
  await request;
  assert.equal(ui.element("editMessage").textContent, "new view");
  assert.equal(ui.state.trip, null);
  assert.equal(ui.state.editingTrip, false);
});

for (const destination of ["list", "different trip", "reloaded same trip"]) {
  test(`checklist completion after navigation to ${destination} does not render stale details`, async () => {
    const ui = browser();
    const response = deferred();
    ui.state.trip = trip(1);
    const item = { id: 10, done: 0 };
    ui.state.trip.checklist = [item];
    const checkbox = new FakeElement();
    ui.context.api = () => response.promise;
    const request = ui.context.updateChecklistItem(item, true, checkbox);
    ui.context.showTrips();
    if (destination !== "list") ui.state.trip = trip(destination === "different trip" ? 9 : 1);
    const currentTrip = ui.state.trip;
    response.resolve({});
    await request;
    assert.equal(item.done, 1);
    assert.equal(ui.state.trip, currentTrip);
    assert.deepEqual(ui.checklistRenders, []);
    assert.deepEqual(ui.toasts, []);
    assert.equal(checkbox.disabled, false);
    assert.equal(ui.summaryRefreshes, 1);
  });
}

test("checklist success still updates its current view", async () => {
  const ui = browser();
  const item = { id: 10, done: 0 };
  ui.state.trip = trip(1);
  ui.state.trip.checklist = [item];
  ui.context.api = async () => ({});
  await ui.context.updateChecklistItem(item, true, new FakeElement());
  assert.equal(item.done, 1);
  assert.deepEqual(ui.checklistRenders, [1]);
});

test("logout invalidates an already loading detail response", async () => {
  const ui = browser();
  const response = deferred();
  ui.context.api = (url) => url === "/api/logout" ? Promise.resolve({}) : response.promise;
  const request = ui.context.openTrip(1);
  await ui.context.logout();
  response.resolve(trip(1));
  await request;
  assert.equal(ui.state.trip, null);
  assert.deepEqual(ui.renders, []);
  assert.equal(ui.element("appShell").hidden, true);
  assert.equal(ui.element("loading").hidden, true);
  assert.equal(ui.location.hash, "#login");
});

for (const operation of ["edit", "checklist"]) {
  test(`late ${operation} completion after logout cannot refresh another session's list`, async () => {
    const ui = browser();
    const response = deferred();
    ui.context.api = (url) => url === "/api/logout" ? Promise.resolve({}) : response.promise;
    let request;
    if (operation === "edit") {
      request = startEdit(ui);
    } else {
      const item = { id: 10, done: 0 };
      ui.state.trip = trip(1);
      ui.state.trip.checklist = [item];
      request = ui.context.updateChecklistItem(item, true, new FakeElement());
    }
    await ui.context.logout();
    ui.state.me = { name: "Different session" };
    response.resolve(trip(1));
    await request;
    assert.equal(ui.summaryRefreshes, 0);
    assert.equal(ui.state.trip, null);
    assert.deepEqual(ui.renders, []);
    assert.deepEqual(ui.checklistRenders, []);
  });
}

test("a summary response started before logout cannot populate a later session", async () => {
  const ui = browser();
  const response = deferred();
  ui.context.api = (url) => url === "/api/logout" ? Promise.resolve({}) : response.promise;
  ui.context.renderTrips = () => { throw new Error("stale summary must not render"); };
  const request = ui.refreshTripSummary(ui.state.me);
  await ui.context.logout();
  ui.state.me = { name: "Different session" };
  const currentTrips = [];
  ui.state.trips = currentTrips;
  response.resolve({ trips: [trip(1)] });
  await request;
  assert.equal(ui.state.trips, currentTrips);
});

test("a queued change from a detached checklist does not send a request for another trip", async () => {
  const ui = browser();
  ui.state.trip = trip(9);
  ui.context.api = () => { throw new Error("detached item must not be submitted"); };
  await ui.context.updateChecklistItem({ id: 10, done: 0 }, true, new FakeElement());
  assert.deepEqual(ui.toasts, []);
  assert.equal(ui.summaryRefreshes, 0);
});

test("structured saved-trip errors preserve the server's do-not-repeat message", () => {
  const ui = browser();
  const message = "Маршрут сохранён. Откройте его заново, не повторяйте генерацию.";
  assert.equal(ui.context.apiErrorMessage({ detail: { message, saved: true, trip_id: 7 } }, "fallback"), message);
});

test("organisation attribution uses only a verified provider's exact safe URL", () => {
  const ui = browser();
  const point = { verified: true, provider: "2gis", source_url: "https://2gis.ru/irkutsk/firm/1548640652976963" };
  const link = ui.context.mapSourceLink(point);
  assert.equal(link.textContent, "Источник координат: 2ГИС");
  assert.equal(link.href, point.source_url);
  assert.equal(link.rel, "noopener noreferrer");
  assert.equal(link.target, "_blank");
  for (const change of [
    { verified: false }, { verified: "true" }, { provider: "unknown" }, { source_url: "javascript:alert(1)" },
    { source_url: "https://2gis.ru.evil.invalid/irkutsk/firm/123" },
    { source_url: "https://2gis.ru/irkutsk/firm/123?redirect=evil" }, { source_url: "" },
  ]) assert.equal(ui.context.mapSourceLink({ ...point, ...change }), null);
});

test("OSM coordinate link remains separate from independent provider attribution", () => {
  const ui = browser();
  const point = { verified: true, provider: "osm", source_url: "https://www.openstreetmap.org/?mlat=51.000000&mlon=101.000000#map=17/51.000000/101.000000" };
  assert.equal(ui.context.mapSourceLink(point).textContent, "Открыть на OpenStreetMap");
  assert.equal(ui.context.mapSourceLink({ ...point, provider: "2gis" }), null);
  assert.match(appSource, /attribution: '&copy; <a href="https:\/\/www\.openstreetmap\.org\/copyright">OpenStreetMap/);
});

function detailTrip(id = 101) {
  return {
    id, destination: "Байкал, Россия", start_date: "2026-10-28", end_date: "2026-10-28",
    request: { adults: 2, children_ages: [], currency: "RUB", pace: "средний" },
    itinerary: {
      summary: "Предварительный план. Известные тарифы и резервы учитываются отдельно.",
      tips: ["Стоимость транспорта нужно уточнить.", "Запись и свободные места не подтверждены."],
      days: [{ day: 1, date: "2026-10-28", theme: "Программа дня", items: [] }],
    },
    budget: { limit: 60000, remaining: 35000, program_total: 0, planned_total: 25000, days: [], scope_note: "Плановые суммы." },
    checklist: [], map_points: [],
  };
}

function elementText(element) {
  return [element.textContent, ...element.children.map(elementText)].join(" ");
}

test("real city rendering labels only zero-price visits and discloses the unchanged preliminary total", () => {
  const ui = browser();
  ui.state.trip = detailTrip();
  ui.state.trip.itinerary.city_schedule = true;
  ui.state.trip.itinerary.source_schedule = false;
  ui.state.trip.itinerary.days[0].items = [
    { kind: "attraction", title: "Музей без тарифа", time: "09:00", cost: 0 },
    { kind: "attraction", title: "Платное посещение", time: "11:00", cost: 300 },
    { kind: "meal", title: "Обед", time: "13:00", cost: 0 },
    { kind: "rest", title: "Отдых", time: "14:00", cost: 0 },
  ];
  Object.assign(ui.state.trip.budget, { currency: "RUB", program_total: 300, planned_total: 25300, remaining: 34700 });
  ui.state.selectedDay = 1;
  const before = JSON.stringify(ui.state.trip);
  ui.originalRenderTrip();
  const dayText = elementText(ui.element("timeline"));
  assert.equal((dayText.match(/Цена не уточнена/g) || []).length, 1);
  assert.ok(dayText.includes("300"));
  for (const id of ["tripNotes", "budgetPanel"]) {
    const text = elementText(ui.element(id));
    assert.ok(text.includes("Посещений с неуточнённой ценой: 1."));
    assert.ok(text.includes("не включает неизвестную стоимость входа"));
  }
  assert.ok(elementText(ui.element("budgetPanel")).includes(ui.context.formatMoney(25300, "RUB")));
  assert.equal(JSON.stringify(ui.state.trip), before, "rendering must not invent prices or reserves");
});

test("unknown-price canonical server text is literal and is not duplicated within the budget", () => {
  const ui = browser();
  ui.state.trip = detailTrip();
  Object.assign(ui.state.trip.itinerary, { city_schedule: true, source_schedule: false,
    unpriced_visit_note: "Посещений с неуточнённой ценой: 1. <b>Неизвестный вход не включён</b>" });
  ui.state.trip.itinerary.days[0].items = [{ kind: "attraction", title: "Музей", time: "09:00", cost: 0 }];
  const note = ui.state.trip.itinerary.unpriced_visit_note;
  ui.state.trip.budget.scope_note += " " + note;
  ui.state.selectedDay = 1;
  ui.originalRenderTrip();
  assert.equal(elementText(ui.element("budgetPanel")).split(note).length - 1, 1);
  assert.ok(elementText(ui.element("tripNotes")).includes(note));
});

test("budget separates each day's running total from the whole-trip estimate and lists missing costs safely", () => {
  const ui = browser();
  ui.state.trip = detailTrip();
  Object.assign(ui.state.trip.budget, { currency: "USD", limit: 20000, planned_total: 8124,
    program_total: 124, remaining: 11876, travel_cost: 3000, lodging_cost: 5000,
    days: [1, 2, 3, 4].map((day) => ({day, day_total: 31, planned_through_day: 8000 + day * 31,
      remaining_after_day: 12000 - day * 31})),
    coverage: {status: "incomplete", missing: [{code: "admission", message: "<b>Цена не уточнена</b>"}],
      notes: ["<script>transport note</script>"]},
  });
  const before = JSON.stringify(ui.state.trip);
  for (const day of [1, 2, 3, 4]) {
    ui.state.selectedDay = day;
    ui.context.renderBudget();
    const text = elementText(ui.element("budgetPanel"));
    assert.ok(text.includes("Смета всей поездки"));
    assert.ok(text.includes("Смета неполная"));
    assert.ok(text.includes(`Программа дня ${day}: ${ui.context.formatMoney(31, "USD")}`));
    assert.ok(text.includes(ui.context.formatMoney(8000 + day * 31, "USD")));
    assert.ok(text.includes(ui.context.formatMoney(12000 - day * 31, "USD")));
    assert.ok(text.includes(ui.context.formatMoney(11876, "USD")), "whole-trip remaining stays invariant");
    assert.ok(text.includes("<b>Цена не уточнена</b>"));
    assert.ok(text.includes("<script>transport note</script>"));
  }
  assert.equal(JSON.stringify(ui.state.trip), before);
});

test("legacy, source and nonvisit zero costs never gain an unknown admission label", () => {
  const ui = browser();
  for (const flags of [{}, { city_schedule: false, source_schedule: true },
    { city_schedule: true, source_schedule: true }, { city_schedule: "true", source_schedule: false }]) {
    ui.state.trip = detailTrip();
    Object.assign(ui.state.trip.itinerary, flags);
    ui.state.trip.itinerary.days[0].items = [{ kind: "attraction", title: "Музей", time: "09:00", cost: 0 }];
    ui.state.selectedDay = 1;
    ui.originalRenderTrip();
    for (const id of ["timeline", "tripNotes", "budgetPanel"]) {
      assert.ok(!elementText(ui.element(id)).includes("Цена не уточнена"));
      assert.ok(!elementText(ui.element(id)).includes("Посещений с неуточнённой ценой"));
    }
  }
  Object.assign(ui.state.trip.itinerary, { city_schedule: true, source_schedule: false });
  ui.state.trip.itinerary.days[0].items = [{ kind: "meal", title: "Обед", time: "13:00", cost: 0 },
    { kind: "rest", title: "Отдых", time: "14:00", cost: 0 }];
  ui.originalRenderTrip();
  assert.equal(ui.context.cityUnpricedVisitNote(ui.state.trip.itinerary), "");
  assert.ok(!elementText(ui.element("timeline")).includes("Цена не уточнена"));
});

test("region hint and route header disclose the lake shoreline margin without changing city copy", () => {
  const ui = browser();
  ui.element("tripDestinationKind").value = "region";
  ui.context.syncDestinationKind();
  const hint = ui.element("tripDestinationHint").textContent;
  for (const text of ["береговая зона до 5 км", "для остальных регионов — допуск до 40 км",
    "технический допуск, не официальная граница", "Базы дней и переезды",
    "сезонную доступность", "стоимость", "не подтверждает бронирование"]) {
    assert.ok(hint.includes(text), text);
  }
  ui.state.trip = detailTrip();
  ui.state.trip.request.destination_kind = "region";
  ui.context.renderTripHeader();
  assert.ok(elementText(ui.element("tripHeader")).includes(hint));
  ui.element("tripDestinationKind").value = "city";
  ui.context.syncDestinationKind();
  assert.match(ui.element("tripDestinationHint").textContent, /Маршрут планируется в городском радиусе/);
  assert.match(ui.element("tripDestinationHint").textContent, /Москва, Москва, Россия/);
  assert.match(ui.element("tripDestinationHint").textContent, /Сингапур, Сингапур.*Ватикан, Ватикан/);
  ui.state.trip.request.destination_kind = "city";
  ui.context.renderTripHeader();
  assert.ok(!elementText(ui.element("tripHeader")).includes("береговая зона"));
});

function creationBrowser() {
  const ui = browser();
  ui.context.newTripPayload = () => ({ destination: "Байкал, Россия" });
  ui.context.syncChildrenMode = () => {};
  ui.context.syncDestinationKind = () => {};
  ui.context.syncLocationMode = () => {};
  ui.context.renderTrips = () => {};
  ui.element("infoDialog").open = true;
  return ui;
}

function realCreationForm() {
  const ui = browser();
  ui.state.me = { name: "Offline user", adults: 2, children_ages: [5], currency: "RUB",
    pace: "средний", wishes: "Только Байкал и побережье, без Иркутска" };
  ui.context.openNewTripDialog();
  for (const [id, value] of Object.entries({ tripDestination: "Сочи, Россия", tripStartDate: "2099-06-01",
    tripEndDate: "2099-06-04", tripBudget: "60000", tripTravelCost: "10000", tripLodgingCost: "15000" })) {
    ui.element(id).value = value;
  }
  ui.context.document.querySelectorAll = (selector) => selector === 'input[name="interest_categories"]:checked'
    ? [{ value: "main_sights" }] : [];
  ui.element("newTripForm").controls = [ui.element("tripWishes")];
  return ui;
}

test("new trip starts without previous destination wishes and retains shared profile defaults", () => {
  const ui = realCreationForm();
  const profile = JSON.stringify(ui.state.me);
  assert.equal(ui.element("tripWishes").value, "");
  assert.equal(ui.element("tripAdults").value, "2");
  assert.equal(ui.element("tripChildrenAges").value, "5");
  assert.equal(ui.element("tripNoChildren").checked, false);
  assert.equal(ui.element("tripCurrency").value, "RUB");
  assert.equal(ui.element("tripPace").value, "средний");
  assert.equal(ui.context.newTripPayload().wishes, "");
  assert.equal(JSON.stringify(ui.state.me), profile, "opening a form must not rewrite the stored profile");
});

test("manual new-trip wishes are included unchanged when destination controls change", () => {
  const ui = realCreationForm();
  const manual = "Хочу музеи Сочи и прогулки у моря";
  ui.element("tripWishes").value = manual;
  ui.element("tripDestinationKind").value = "region";
  ui.element("tripDestinationKind").dispatch("change");
  ui.element("tripDestinationKind").value = "city";
  ui.element("tripDestinationKind").dispatch("change");
  const payload = ui.context.newTripPayload();
  assert.equal(payload.destination, "Сочи, Россия");
  assert.equal(payload.wishes, manual);
  assert.equal(ui.element("tripWishes").value, manual);
  assert.ok(!payload.wishes.includes("Байкал"));
});

test("local validation failure preserves the manual wishes draft for corrected resubmission", async () => {
  const ui = realCreationForm();
  const manual = "Без долгих переездов по Сочи";
  ui.element("tripWishes").value = manual;
  ui.element("tripBudget").value = "0";
  const payloads = [];
  ui.context.api = async (url, options) => {
    assert.equal(url, "/api/trips");
    payloads.push(JSON.parse(options.body));
    throw Object.assign(new Error("Уточните условия поездки"), { status: 422 });
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(payloads.length, 0);
  assert.match(ui.element("newTripMessage").textContent, /бюджет больше нуля/);
  assert.equal(ui.element("tripWishes").value, manual);
  assert.equal(ui.element("infoDialog").open, true);
  ui.element("tripBudget").value = "60000";
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(payloads.length, 1);
  assert.equal(payloads[0].wishes, manual);
  assert.equal(ui.element("tripWishes").value, manual);
  assert.equal(ui.state.createdTripId, null);
});

test("server and connection failures preserve the in-flight draft and explicit retry payload", async () => {
  const ui = realCreationForm();
  const manual = "Нужна спокойная программа именно в Сочи";
  ui.element("tripWishes").value = manual;
  const firstResponse = deferred();
  const payloads = [];
  ui.context.api = async (url, options) => {
    assert.equal(url, "/api/trips");
    payloads.push(JSON.parse(options.body));
    if (payloads.length === 1) return firstResponse.promise;
    throw new Error("Synthetic connection failure");
  };
  const pending = ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.state.creatingTrip, true);
  ui.context.openNewTripDialog(); // Opening while busy must not clear the current draft.
  assert.equal(ui.element("tripWishes").value, manual);
  firstResponse.reject(Object.assign(new Error("Программа требует уточнения"), { status: 422 }));
  await pending;
  assert.equal(ui.element("tripWishes").value, manual);
  assert.equal(ui.element("tripWishes").disabled, false);
  assert.equal(ui.element("infoDialog").open, true);
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(payloads.length, 2);
  assert.deepEqual(payloads[1], payloads[0]);
  assert.equal(payloads[1].wishes, manual);
  assert.equal(ui.element("tripWishes").value, manual);
  assert.equal(ui.state.creatingTrip, false);
  assert.equal(ui.state.createdTripId, null);
});

test("city MVP disclosure is visible before creation only for an enabled city and clears on switch", () => {
  const ui = browser();
  const literal = "Предварительная программа; возможны свободные дни. <b>Условия не подтверждены</b>";
  ui.state.me.capabilities = { city_candidates_first: true, city_mvp_disclosure: literal };
  ui.context.openNewTripDialog();
  assert.equal(ui.element("infoDialog").open, true);
  assert.equal(ui.element("cityMvpDisclosure").hidden, false);
  assert.equal(ui.element("cityMvpDisclosure").textContent, literal);
  assert.equal(ui.element("cityMvpDisclosure").children.length, 0, "disclosure is literal text, not HTML");
  ui.element("tripDestinationKind").value = "region";
  ui.element("tripDestinationKind").dispatch("change");
  assert.equal(ui.element("cityMvpDisclosure").hidden, true);
  assert.equal(ui.element("cityMvpDisclosure").textContent, "");
  assert.match(ui.element("tripDestinationHint").textContent, /до 5 км/);
  ui.element("tripDestinationKind").value = "city";
  ui.element("tripDestinationKind").dispatch("change");
  assert.equal(ui.element("cityMvpDisclosure").hidden, false);
  for (const capabilities of [undefined, {}, { city_candidates_first: false },
    { city_candidates_first: "true" }, { city_candidates_first: 1 }]) {
    ui.state.me.capabilities = capabilities;
    ui.context.syncDestinationKind();
    assert.equal(ui.element("cityMvpDisclosure").hidden, true);
    assert.equal(ui.element("cityMvpDisclosure").textContent, "");
  }
});

test("enabled city MVP has a readable fallback disclosure if public text is absent", () => {
  const ui = browser();
  ui.state.me.capabilities = { city_candidates_first: true };
  ui.context.openNewTripDialog();
  const text = ui.element("cityMvpDisclosure").textContent;
  for (const expected of ["Предварительная", "свободной", "цены", "помещения", "не рассчитаны", "Редактирование"]) {
    assert.ok(text.includes(expected), expected);
  }
  assert.equal(ui.element("cityMvpDisclosure").hidden, false);
});

test("real trip rendering exposes summary and every warning as text and clears stale notes", () => {
  const ui = browser();
  ui.context.renderTrip = ui.originalRenderTrip;
  ui.context.renderChecklist = ui.originalRenderChecklist;
  ui.state.trip = detailTrip();
  const literal = '<img src=x onerror="throw new Error()"> — это текст совета';
  ui.state.trip.itinerary.tips.push(literal);
  ui.context.renderTrip();
  assert.equal(ui.element("tripNotes").hidden, false);
  const text = elementText(ui.element("tripNotes"));
  assert.ok(text.includes(ui.state.trip.itinerary.summary));
  for (const tip of ui.state.trip.itinerary.tips) assert.ok(text.includes(tip));
  const explanation = ui.element("tripNotes").children.find((child) => child.tagName === "DETAILS");
  assert.ok(explanation, "long notes have an accessible native disclosure");
  assert.equal(explanation.children[0].tagName, "SUMMARY");
  assert.equal(explanation.children[1].children.at(-1).tagName, "UL");
  assert.equal(explanation.children[1].children.at(-1).children.at(-1).tagName, "LI");
  ui.state.trip = detailTrip(102);
  ui.state.trip.itinerary.summary = "";
  ui.state.trip.itinerary.tips = [];
  ui.context.renderTrip();
  assert.equal(ui.element("tripNotes").hidden, true);
  assert.equal(ui.element("tripNotes").children.length, 0);
});

function freeDaysDetail() {
  const value = detailTrip();
  value.end_date = "2026-11-03";
  value.itinerary.source_schedule = true;
  value.itinerary.program_report = { schema_version: 1, reason: "limited_verified_program",
    requested_days: 7, program_days: 3, free_days: 4, eligible_unique_visits: 8, selected_unique_visits: 5 };
  value.itinerary.days = Array.from({ length: 7 }, (_, index) => ({
    day: index + 1, date: index < 4 ? `2026-10-${28 + index}` : `2026-11-0${index - 3}`,
    day_mode: index < 3 ? "program" : "free", theme: index < 3 ? "Программа дня" : "Свободный режим",
    items: index < 3 ? [] : [
      { time: "13:00", kind: "meal", title: "Обед", description: "Питание учтено.", cost: 1400 },
      { time: "14:00", kind: "rest", title: "Отдых", description: "Без обещанных мест.", cost: 0 },
    ],
  }));
  return value;
}

test("destination illustrations use exact local aliases and season without accepting remote URLs", () => {
  const ui = browser();
  const dates = { start_date: "2026-10-28", end_date: "2026-11-03" };
  for (const [destination, file] of [[" Москва , Россия ", "moscow"], ["Сочи", "sochi"],
    ["Озеро Байкал, Россия", "baikal"], ["Nha Trang, Vietnam", "nha-trang"]]) {
    const cover = ui.context.destinationCover({ ...dates, destination });
    assert.equal(cover.src, `/static/images/${file}.png`);
    assert.match(cover.caption, /ИИ-иллюстрация.*не фотография/);
    assert.match(cover.alt, /не фотография/);
  }
  for (const destination of ["Moscow, USA", "Москва, США", "Сочи и Москва", "Неизвестный город",
    "https://tracker.example/cover.png", "../../etc/passwd", '<img src=x onerror="alert(1)">']) {
    assert.equal(ui.context.destinationCover({ ...dates, destination }), null);
  }
  for (const start_date of ["2026-01-28", "2026-07-28", "", null]) {
    assert.equal(ui.context.destinationCover({ ...dates, start_date, destination: "Москва" }), null);
  }
  assert.equal(ui.context.destinationCover({ ...dates, end_date: "2026-12-03", destination: "Москва" }), null);
});

test("covers label AI artwork and missing assets degrade to a neutral atlas without another request", () => {
  const ui = browser();
  const cover = ui.context.createTripCover({ destination: "Москва, Россия",
    start_date: "2026-10-28", end_date: "2026-10-30" });
  const photo = cover.children.find((child) => child.tagName === "IMG");
  const placeholder = cover.children[0];
  assert.ok(photo);
  assert.equal(photo.loading, "lazy");
  assert.equal(placeholder.hidden, true);
  assert.match(elementText(cover), /ИИ-иллюстрация/);
  photo.dispatch("error");
  assert.equal(photo.hidden, true);
  assert.equal(placeholder.hidden, false);
  assert.equal(cover.children.at(-1).textContent, "Иллюстрация путешествия");
  assert.equal(photo.src, "/static/images/moscow.png", "failure must not call another provider or retry a loop");
  const generic = ui.context.createTripCover({ destination: "Марс" }, true);
  assert.ok(!generic.children.some((child) => child.tagName === "IMG"));
  assert.equal(generic.children[0].hidden, false);
  assert.equal(generic.children.at(-1).textContent, "Иллюстрация путешествия");
});

test("premium cards retain original Open/PDF actions and exact supplied costs", () => {
  const ui = browser();
  let opened;
  ui.context.openTrip = (id) => { opened = id; };
  const card = ui.context.createTripCard({ id: 77, destination: "Байкал, Россия", destination_kind: "region",
    start_date: "2026-10-28", end_date: "2026-11-03", title: "Маршрут", status: "upcoming",
    days_count: 7, planned_total: 52000, currency: "RUB", checklist: { done: 0, total: 30 } });
  assert.equal(card.children[0].className, "trip-card-cover");
  const content = card.children[1];
  assert.equal(content.className, "trip-card-content");
  assert.match(elementText(content), /Байкал/);
  assert.ok(elementText(content).includes(ui.context.formatMoney(52000, "RUB")));
  content.children.at(-1).children[0].dispatch("click");
  assert.equal(opened, 77);
  content.children.at(-1).children[1].dispatch("click");
  assert.equal(ui.location.href, "/api/trips/77/pdf");
});

test("coverage and unknown prices remain visible outside the optional long notes disclosure", () => {
  const ui = browser();
  ui.state.trip = freeDaysDetail();
  ui.state.trip.itinerary.source_schedule = false;
  ui.state.trip.itinerary.city_schedule = true;
  ui.state.trip.itinerary.program_report.coverage_basis = "map";
  ui.state.trip.itinerary.days[0].items.push({ kind: "attraction", cost: 0 });
  ui.context.renderTripNotes();
  const alwaysVisible = ui.element("tripNotes").children.filter((child) => child.tagName !== "DETAILS");
  const text = alwaysVisible.map(elementText).join(" ");
  assert.match(text, /Дней с программой: 3/);
  assert.match(text, /Дней свободного режима: 4/);
  assert.match(text, /неуточнённой ценой: 1/);
  assert.match(text, /нулевой резерв не означает бесплатный вход/);
});

test("partial quality and missing interests stay visible outside collapsed notes", () => {
  const ui = browser();
  ui.state.trip = freeDaysDetail();
  ui.state.trip.generation_quality = { quality_status: "partial", program_days: 1, requested_days: 7,
    reason_texts: ["Найденного подтверждённого набора недостаточно."],
    interest_texts: ["зоопарки: не удалось подтвердить места"] };
  ui.context.renderTripNotes();
  const text = ui.element("tripNotes").children.filter((child) => child.tagName !== "DETAILS").map(elementText).join(" ");
  assert.match(text, /Программа составлена частично/);
  assert.match(text, /Дней с программой: 1 из 7/);
  assert.match(text, /зоопарки: не удалось подтвердить/);
});

test("memory ranking limitation is visible in the quality panel", () => {
  const ui = browser();
  ui.state.trip = freeDaysDetail();
  ui.state.trip.generation_quality = { quality_status: "complete", program_days: 7, requested_days: 7,
    memory_note: "Память влияет на приоритеты. Обязательные ограничения укажите в текущей анкете." };
  ui.context.renderTripNotes();
  const text = ui.element("tripNotes").children.filter((child) => child.tagName !== "DETAILS").map(elementText).join(" ");
  assert.match(text, /Обязательные ограничения укажите в текущей анкете/);
});

test("real rendering distinguishes verified program coverage from explicitly free calendar days", () => {
  const ui = browser();
  ui.context.renderTrip = ui.originalRenderTrip;
  ui.context.renderChecklist = ui.originalRenderChecklist;
  ui.state.trip = freeDaysDetail();
  ui.state.selectedDay = 4;
  const budgetBefore = JSON.stringify(ui.state.trip.budget);
  ui.context.renderTrip();
  const text = elementText(ui.element("tripNotes"));
  for (const expected of ["Проверено подходящих мест: 8", "Включено: 5", "Дней с программой: 3",
    "Дней свободного режима: 4", "Даты сохранены", "не утверждение, что других мест нет",
    "Резерв на обед учтён во все дни; личные расходы не запланированы", ...ui.state.trip.itinerary.tips]) {
    assert.ok(text.includes(expected), expected);
  }
  assert.equal(ui.element("dayTabs").children.length, 7);
  assert.ok(!elementText(ui.element("dayTabs").children[0]).includes("Свободный режим"));
  assert.match(elementText(ui.element("dayTabs").children[3]), /Свободный режим/);
  assert.match(elementText(ui.element("timeline")), /Свободный день: посещения не запланированы/);
  assert.match(elementText(ui.element("timeline")), /Обед.*Отдых/);
  assert.equal(ui.element("map").hidden, true);
  assert.match(ui.element("mapEmpty").textContent, /Свободный день.*точки на карте не требуются/);
  assert.equal(JSON.stringify(ui.state.trip.budget), budgetBefore, "display must not discount meals or reserves");
});

test("free-day map resets stale provider failures and switching back preserves the ordinary empty-map warning", () => {
  const ui = browser();
  ui.state.trip = freeDaysDetail();
  ui.element("mapEmpty").textContent = "Карта временно недоступна. Координаты мест сохранены.";
  ui.state.trip.map_points = [{ day: 4, verified: true, lat: 51, lon: 104 }];
  ui.state.selectedDay = 4;
  ui.context.renderSelectedDay();
  assert.match(ui.element("mapEmpty").textContent, /Свободный день/);
  assert.equal(ui.element("map").hidden, true, "free day must not display stale map points");
  ui.state.selectedDay = 1;
  ui.context.renderSelectedDay();
  assert.equal(ui.element("mapEmpty").textContent, "Для этого дня пока нет подтверждённых координат.");
  assert.ok(!elementText(ui.element("timeline")).includes("Свободный режим"));
});

test("coverage requires the explicit server report, not empty days or an arbitrary summary", () => {
  const ui = browser();
  const value = freeDaysDetail();
  for (const report of [null, {}, { ...value.itinerary.program_report, schema_version: 2 },
    { ...value.itinerary.program_report, free_days: 99 },
    { ...value.itinerary.program_report, eligible_unique_visits: '<img src=x>' },
    { ...value.itinerary.program_report, selected_unique_visits: 9 }]) {
    assert.equal(ui.context.programCoverageText({ ...value.itinerary, program_report: report }), "");
  }
  assert.equal(ui.context.programCoverageText({ ...value.itinerary, source_schedule: false }), "");
  ui.state.trip = detailTrip();
  ui.context.renderDayTabs();
  assert.ok(!elementText(ui.element("dayTabs")).includes("Свободный"));
});

test("canonical server disclosure wins over numeric fallback without losing warnings", () => {
  const ui = browser();
  ui.state.trip = freeDaysDetail();
  ui.state.trip.itinerary.program_report_text = "Сведения о свободных днях требуют перепроверки.";
  ui.context.renderTripNotes();
  let text = elementText(ui.element("tripNotes"));
  assert.match(text, /Сведения о свободных днях требуют перепроверки/);
  assert.ok(!text.includes("Проверено подходящих мест: 8"));
  for (const tip of ui.state.trip.itinerary.tips) assert.ok(text.includes(tip));
  ui.state.trip.itinerary.program_report_text = "";
  ui.context.renderTripNotes();
  text = elementText(ui.element("tripNotes"));
  assert.ok(!text.includes("Программа и свободные дни"), "empty canonical disclosure is also authoritative");
});

test("city-map free-day disclosure does not claim source-backed visit facts", () => {
  const ui = browser();
  const value = freeDaysDetail();
  value.itinerary.source_schedule = false;
  value.itinerary.city_schedule = true;
  value.itinerary.program_report.coverage_basis = "map";
  const text = ui.context.programCoverageText(value.itinerary);
  assert.match(text, /Подтверждено мест на карте: 8/);
  assert.match(text, /но не условия посещения, часы работы, цены, длительность или наличие помещения/);
  assert.ok(!text.includes("Проверено подходящих мест"));
  assert.equal(ui.context.programCoverageText({ ...value.itinerary, source_schedule: true }), "");
  assert.equal(ui.context.programCoverageText({ ...value.itinerary, city_schedule: false }), "");
  assert.equal(ui.context.programCoverageText({ ...value.itinerary,
    program_report: { ...value.itinerary.program_report, coverage_basis: "source" } }), "");
});

test("city candidate plans disable both edit controls and never submit a stale dialog", async () => {
  const ui = browser();
  ui.state.trip = freeDaysDetail();
  ui.state.trip.itinerary.source_schedule = false;
  ui.state.trip.itinerary.city_schedule = true;
  ui.state.trip.edit_available = false;
  ui.state.trip.edit_unavailable_reason = "Повторная проверка изменений пока не поддерживается.";
  ui.state.selectedDay = 4;
  ui.context.renderTripHeader();
  assert.equal(ui.element("editDayButton").disabled, true);
  assert.equal(ui.element("tripHeader").children[1].children[1].disabled, true);
  assert.match(elementText(ui.element("tripHeader")), /Повторная проверка изменений/);
  ui.context.openEditDialog();
  assert.equal(ui.element("editDialog").open, false);
  ui.element("editInstruction").value = "Перенеси обед";
  ui.context.api = async () => { throw new Error("disabled editing must not POST"); };
  await ui.context.submitEdit({ preventDefault() {} });
  assert.equal(ui.state.editingTrip, false);
  assert.equal(ui.toasts.length, 2);
  ui.state.trip = detailTrip();
  ui.context.renderTripHeader();
  assert.equal(ui.element("editDayButton").disabled, false);
  assert.equal(ui.element("tripHeader").children[1].children[1].disabled, false);
  ui.context.openEditDialog();
  assert.equal(ui.element("editDialog").open, true);
});

test("city candidate editing uses server capability and discloses bounded existing pool", () => {
  const ui = browser();
  ui.state.trip = freeDaysDetail();
  ui.state.trip.itinerary.city_schedule = true;
  ui.state.trip.itinerary.source_schedule = false;
  ui.state.trip.edit_available = true;
  ui.state.trip.edit_scope_note = "Изменение только в пределах проверенного набора мест.";
  ui.state.selectedDay = 2;
  ui.context.renderTripHeader();
  assert.equal(ui.element("editDayButton").disabled, false);
  assert.match(elementText(ui.element("tripHeader")), /проверенного набора/);
  ui.context.openEditDialog();
  assert.equal(ui.element("editDialog").open, true);
  assert.match(ui.element("editScopeNote").textContent, /проверенного набора/);
  assert.match(ui.element("editInstruction").placeholder, /перенеси обед/);
});

test("successful creation displays the returned detail when list refresh fails without another POST", async () => {
  const ui = creationBrowser();
  ui.context.renderTrip = ui.originalRenderTrip;
  ui.context.renderChecklist = ui.originalRenderChecklist;
  const saved = detailTrip();
  const calls = [];
  ui.context.api = async (url) => {
    calls.push(url);
    if (url === "/api/trips") return saved;
    throw Object.assign(new Error("Список недоступен"), { status: 503 });
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.state.trip, saved);
  assert.equal(ui.state.createdTripId, 101);
  assert.equal(ui.element("infoDialog").open, false);
  assert.equal(ui.location.hash, "#trip/101");
  assert.ok(elementText(ui.element("tripNotes")).includes(saved.itinerary.tips[0]));
  assert.match(elementText(ui.element("savedTripNotice")), /№101.*сохранён.*Список поездок временно не обновился/);
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.deepEqual(calls, ["/api/trips", "/api/trips?limit=100"]);
});

for (const listFails of [false, true]) {
  test(`saved ID survives detail failure and list ${listFails ? "failure" : "success"}; recovery only reads`, async () => {
    const ui = creationBrowser();
    let recovered = false;
    const calls = [];
    ui.context.api = async (url) => {
      calls.push(url);
      if (url === "/api/trips") return { id: 101 };
      if (url === "/api/trips/101" && recovered) return detailTrip();
      if (url === "/api/trips?limit=100" && !listFails) return { trips: [] };
      throw Object.assign(new Error("Временная ошибка чтения"), { status: 503 });
    };
    await ui.context.submitNewTrip({ preventDefault() {} });
    assert.equal(ui.state.createdTripId, 101);
    assert.equal(ui.element("infoDialog").open, false);
    assert.equal(ui.element("savedTripNotice").hidden, false);
    assert.match(elementText(ui.element("savedTripNotice")), /№101.*сохранён.*Карточка пока не открыта/);
    assert.equal(elementText(ui.element("savedTripNotice")).includes("Список поездок временно не обновился"), listFails);
    await ui.context.submitNewTrip({ preventDefault() {} });
    assert.equal(calls.filter((url) => url === "/api/trips").length, 1);
    recovered = true;
    await ui.element("savedTripNotice").children[1].listeners.get("click")();
    assert.equal(ui.state.trip.id, 101);
    assert.equal(ui.element("savedTripNotice").hidden, true);
    assert.equal(calls.filter((url) => url === "/api/trips").length, 1);
  });
}

test("saved server response errors retain their ID for read-only recovery", async () => {
  const ui = creationBrowser();
  let posts = 0;
  ui.context.fetch = async (url) => {
    if (url === "/api/trips") {
      posts += 1;
      return { ok: false, status: 503, headers: new Headers({ "content-type": "application/json" }),
        json: async () => ({ detail: { saved: true, trip_id: 101, message: "Маршрут сохранён." } }) };
    }
    return { ok: true, status: 200, headers: new Headers({ "content-type": "application/json" }),
      json: async () => url === "/api/trips/101" ? detailTrip() : { trips: [] } };
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.state.trip.id, 101);
  assert.equal(ui.state.createdTripId, 101);
  assert.equal(ui.element("infoDialog").open, false);
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(posts, 1);
});

test("only explicitly starting a new form clears the saved-creation guard", async () => {
  const ui = creationBrowser();
  ui.state.createdTripId = 101;
  ui.context.openNewTripDialog();
  assert.equal(ui.state.createdTripId, null);
  assert.equal(ui.element("infoDialog").open, true);
});

const JOB_KEY = "11111111-1111-4111-8111-111111111111";

function coverBrowser() {
  const ui = browser();
  ui.state.me.capabilities = { cover_images: true };
  ui.state.trip = detailTrip();
  ui.element("tripView").hidden = false;
  ui.context.window.confirm = () => { throw new Error("cover flow must use in-page confirmation, never a native dialog"); };
  const timers = new Map();
  let nextTimer = 0;
  ui.context.setTimeout = (callback) => { timers.set(++nextTimer, callback); return nextTimer; };
  ui.context.clearTimeout = (id) => timers.delete(id);
  return { ...ui, timers };
}

async function settleCovers() {
  await new Promise((resolve) => setImmediate(resolve));
}

function coverPayload(status, available = true, id = 101, hidden = false) {
  return { status, available, hidden, image_url: status === "ready" && !hidden ? `/api/trips/${id}/cover/image` : null };
}

test("opening an existing trip reads its cover but never creates a paid image", async () => {
  const ui = coverBrowser();
  const calls = [];
  ui.context.api = async (path, options) => { calls.push([path, options?.method || "GET"]); return coverPayload("empty"); };
  ui.context.startTripCoverView();
  await settleCovers();
  assert.deepEqual(calls, [["/api/trips/101/cover", "GET"]]);
  const text = elementText(ui.element("tripCoverControls"));
  assert.match(text, /Создать ИИ-обложку/);
  assert.match(text, /отдельный платный запрос/);
  assert.match(text, /не фотографией/);
  assert.equal(ui.timers.size, 0);
});

test("saved cover accepts only the exact current trip private image URL even while generation is disabled", async () => {
  const ui = coverBrowser();
  for (const image_url of ["https://tracker.example/image.png", "//tracker.example/pixel",
    "/api/trips/999/cover/image", "/api/trips/101/cover/image?token=secret", "data:image/svg+xml,unsafe"] ) {
    assert.equal(ui.context.savedCoverAsset({ id: 101 }, { status: "ready", image_url }), null);
    assert.throws(() => ui.context.coverStatusValue({ status: "ready", available: true, image_url }, 101));
  }
  ui.context.api = async () => coverPayload("ready", false);
  ui.context.startTripCoverView();
  await settleCovers();
  const image = ui.element("tripHero").children[0].children.find((child) => child.tagName === "IMG");
  assert.equal(image.src, "/api/trips/101/cover/image");
  assert.match(image.alt, /ИИ-иллюстрация.*не фотография/);
  assert.ok(!elementText(ui.element("tripCoverControls")).includes("Создать ИИ-обложку"));
  assert.equal(ui.timers.size, 0);
});

test("ready covers hide and restore through PATCH without confirmation, POST, regeneration or loss of local fallback", async () => {
  const ui = coverBrowser();
  const calls = [];
  let hidden = false;
  ui.context.api = async (path, options) => {
    calls.push([path, options?.method || "GET", options?.body]);
    if (options?.method === "PATCH") hidden = JSON.parse(options.body).hidden;
    return coverPayload("ready", false, 101, hidden);
  };
  ui.context.startTripCoverView();
  await settleCovers();
  assert.match(elementText(ui.element("tripCoverControls")), /Скрыть ИИ-обложку/);
  ui.element("tripCoverControls").children[1].dispatch("click");
  await settleCovers();
  assert.equal(hidden, true);
  assert.equal(ui.element("tripHero").children[0].children.find((child) => child.tagName === "IMG").src,
    "/static/images/baikal.png");
  assert.match(elementText(ui.element("tripCoverControls")), /ИИ-обложка скрыта.*можно вернуть без новой генерации/);
  assert.match(elementText(ui.element("tripCoverControls")), /Вернуть ИИ-обложку/);
  assert.equal(ui.context.cachedCover(101).value.hidden, true);
  ui.element("tripCoverControls").children[1].dispatch("click");
  await settleCovers();
  assert.equal(hidden, false);
  assert.equal(ui.element("tripHero").children[0].children.find((child) => child.tagName === "IMG").src,
    "/api/trips/101/cover/image");
  assert.deepEqual(calls.map(([, method]) => method), ["GET", "PATCH", "PATCH"]);
  assert.deepEqual(calls.filter(([, method]) => method === "PATCH").map(([, , body]) => JSON.parse(body)),
    [{ hidden: true }, { hidden: false }]);
  assert.equal(ui.timers.size, 0);
});

test("hidden ready cover schema allows only a null image URL and always suppresses the generated asset", () => {
  const ui = coverBrowser();
  const hidden = coverPayload("ready", true, 101, true);
  assert.equal(ui.context.coverStatusValue(hidden, 101).hidden, true);
  assert.equal(ui.context.savedCoverAsset({ id: 101 }, hidden), null);
  assert.equal(ui.context.savedCoverAsset({ id: 101 }, { ...hidden, image_url: "/api/trips/101/cover/image" }), null);
  for (const change of [{ hidden: "true" }, { hidden: 1 },
    { image_url: "/api/trips/101/cover/image" }, { image_url: "https://outside.example/image.png" }]) {
    assert.throws(() => ui.context.coverStatusValue({ ...hidden, ...change }, 101));
  }
});

test("hidden cards show a local illustration or a neutral atlas and never load the private generated image", async () => {
  const ui = coverBrowser();
  const calls = [];
  ui.context.api = async (path, options) => {
    calls.push(options?.method || "GET");
    return coverPayload("ready", false, Number(path.split("/")[3]), true);
  };
  const mounts = [{ trip: detailTrip(101), mount: new FakeElement() },
    { trip: { ...detailTrip(202), destination: "Воркута, Россия" }, mount: new FakeElement() }];
  await ui.context.hydrateTripCardCovers(mounts, ui.state.coverListRevision);
  assert.equal(mounts[0].mount.children.find((child) => child.tagName === "IMG").src, "/static/images/baikal.png");
  assert.ok(!mounts[1].mount.children.some((child) => child.tagName === "IMG"));
  assert.match(elementText(mounts[1].mount), /Иллюстрация путешествия/);
  assert.deepEqual(calls, ["GET", "GET"]);
});

test("visibility changes lock duplicate clicks and ignore late responses after navigation or logout", async () => {
  const ui = coverBrowser();
  const response = deferred();
  const calls = [];
  ui.context.api = async (_path, options) => {
    calls.push(options?.method || "GET");
    return options?.method === "PATCH" ? response.promise : coverPayload("ready");
  };
  ui.context.startTripCoverView();
  await settleCovers();
  const session = ui.state.coverView;
  const request = ui.context.setTripCoverHidden(session, true);
  assert.equal(ui.element("tripCoverControls").children[1].disabled, true);
  await ui.context.setTripCoverHidden(session, true);
  await ui.context.refreshTripCover(session);
  ui.context.beginViewChange();
  ui.state.me = { name: "Another account" };
  ui.state.trip = detailTrip(202);
  const sentinel = new FakeElement("p");
  ui.element("tripHero").replaceChildren(sentinel);
  response.resolve(coverPayload("ready", true, 101, true));
  await request;
  assert.deepEqual(calls, ["GET", "PATCH"]);
  assert.equal(ui.element("tripHero").children[0], sentinel);
  assert.equal(ui.context.cachedCover(101), null);
  assert.equal(ui.timers.size, 0);
});

test("uncertain visibility PATCH recovers with GET only and does not offer image generation", async () => {
  const ui = coverBrowser();
  const calls = [];
  let hidden = false;
  ui.context.api = async (_path, options) => {
    const method = options?.method || "GET";
    calls.push(method);
    if (method === "PATCH") { hidden = true; throw new Error("lost response"); }
    return coverPayload("ready", true, 101, hidden);
  };
  ui.context.startTripCoverView();
  await settleCovers();
  await ui.context.setTripCoverHidden(ui.state.coverView, true);
  assert.match(elementText(ui.element("tripCoverControls")), /Проверьте статус обложки/);
  assert.ok(!elementText(ui.element("tripCoverControls")).includes("Создать ИИ-обложку"));
  ui.element("tripCoverControls").children.at(-1).dispatch("click");
  await settleCovers();
  assert.equal(ui.state.coverView.status.hidden, true);
  assert.deepEqual(calls, ["GET", "PATCH", "GET"]);
});

test("image creation uses an explicit inline confirmation and cancellation without native dialogs or duplicate POST", async () => {
  const ui = coverBrowser();
  const response = deferred();
  const calls = [];
  ui.context.api = async (path, options) => {
    calls.push(options?.method || "GET");
    return options?.method === "POST" ? response.promise : coverPayload("empty");
  };
  ui.context.startTripCoverView();
  await settleCovers();
  const session = ui.state.coverView;
  await ui.context.submitTripCover(session);
  assert.deepEqual(calls, ["GET"]);
  ui.element("tripCoverControls").children.at(-1).dispatch("click");
  assert.equal(session.confirming, true);
  let consent = ui.element("tripCoverControls").children.at(-1);
  assert.equal(consent.role, "group");
  assert.equal(consent["aria-labelledby"], "coverConsentQuestion");
  assert.match(elementText(consent), /один платный запрос/);
  assert.match(elementText(consent), /Подтвердить генерацию/);
  assert.ok(elementText(consent).includes(ui.state.trip.destination));
  const canceledConfirm = consent.children[1].children[0];
  consent.children[1].children[1].dispatch("click");
  assert.equal(session.confirming, false);
  canceledConfirm.dispatch("click");
  assert.deepEqual(calls, ["GET"]);
  ui.context.requestTripCoverConfirmation(session);
  consent = ui.element("tripCoverControls").children.at(-1);
  const request = ui.context.submitTripCover(session);
  consent.children[1].children[0].dispatch("click");
  assert.deepEqual(calls, ["GET", "POST"]);
  assert.match(elementText(ui.element("tripCoverControls")), /один запрос/);
  response.resolve(coverPayload("queued"));
  await request;
  assert.equal(ui.timers.size, 1);
  await ui.context.submitTripCover(session);
  assert.deepEqual(calls, ["GET", "POST"]);
});

test("cover polling is bounded and navigation cancels timers without canceling or duplicating a paid server job", async () => {
  const ui = coverBrowser();
  const calls = [];
  ui.context.api = async (path, options) => { calls.push(options?.method || "GET"); return coverPayload("running"); };
  ui.context.startTripCoverView();
  await settleCovers();
  for (let step = 0; step < 70 && ui.timers.size; step += 1) {
    const [id, callback] = ui.timers.entries().next().value;
    ui.timers.delete(id);
    callback();
    await settleCovers();
  }
  assert.equal(calls.length, 60);
  assert.ok(calls.every((method) => method === "GET"));
  assert.equal(ui.timers.size, 0);
  assert.match(elementText(ui.element("tripCoverControls")), /проверка обложки завершена/);
  ui.element("tripCoverControls").children.at(-1).dispatch("click");
  await settleCovers();
  assert.equal(ui.timers.size, 1);
  ui.context.beginViewChange();
  assert.equal(ui.timers.size, 0);
  assert.equal(ui.state.coverView, null);
});

test("stale inline cover confirmation cannot submit after navigation even when reopening the same trip", async () => {
  for (const nextTripId of [101, 202]) {
    const ui = coverBrowser();
    const calls = [];
    ui.context.api = async (path, options) => { calls.push([path, options?.method || "GET"]); return coverPayload("empty"); };
    ui.context.startTripCoverView();
    await settleCovers();
    const oldSession = ui.state.coverView;
    ui.context.requestTripCoverConfirmation(oldSession);
    const oldActions = ui.element("tripCoverControls").children.at(-1).children[1];
    ui.context.beginViewChange();
    ui.state.trip = detailTrip(nextTripId);
    ui.context.startTripCoverView();
    await settleCovers();
    const currentSession = ui.state.coverView;
    oldActions.children[0].dispatch("click");
    oldActions.children[1].dispatch("click");
    await ui.context.submitTripCover(oldSession);
    assert.equal(ui.state.coverView, currentSession);
    assert.equal(currentSession.confirming, false);
    assert.ok(calls.every(([, method]) => method === "GET"));
  }
});

test("Escape cancels the in-page cover confirmation without sending a paid request", async () => {
  const ui = coverBrowser();
  const calls = [];
  ui.context.api = async (_path, options) => { calls.push(options?.method || "GET"); return coverPayload("empty"); };
  ui.context.startTripCoverView();
  await settleCovers();
  ui.context.requestTripCoverConfirmation(ui.state.coverView);
  const consent = ui.element("tripCoverControls").children.at(-1);
  let prevented = false;
  consent.listeners.get("keydown")({ key: "Escape", preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(ui.state.coverView.confirming, false);
  assert.match(elementText(ui.element("tripCoverControls")), /Создать ИИ-обложку/);
  assert.deepEqual(calls, ["GET"]);
});

test("late cover GET and POST responses cannot overwrite another trip or cross logout", async () => {
  for (const method of ["GET", "POST"]) {
    const ui = coverBrowser();
    const response = deferred();
    ui.context.api = async (_path, options) => (options?.method || "GET") === method
      ? response.promise : coverPayload("empty");
    ui.context.startTripCoverView();
    if (method === "POST") {
      await settleCovers();
      ui.context.requestTripCoverConfirmation(ui.state.coverView);
    }
    const request = method === "POST" ? ui.context.submitTripCover(ui.state.coverView) : Promise.resolve();
    ui.context.beginViewChange();
    ui.state.me = { name: "Another account" };
    ui.state.trip = detailTrip(202);
    response.resolve(coverPayload("ready"));
    await request;
    await settleCovers();
    assert.equal(ui.element("tripHero").children.length, 0);
    assert.equal(ui.context.cachedCover(101), null);
    assert.equal(ui.timers.size, 0);
  }
});

test("lost image POST response leaves read-only recovery and never repeats submission", async () => {
  const ui = coverBrowser();
  const calls = [];
  let result = "empty";
  ui.context.api = async (_path, options) => {
    const method = options?.method || "GET";
    calls.push(method);
    if (method === "POST") throw new Error("connection lost");
    return coverPayload(result);
  };
  ui.context.startTripCoverView();
  await settleCovers();
  ui.context.requestTripCoverConfirmation(ui.state.coverView);
  await ui.context.submitTripCover(ui.state.coverView);
  assert.match(elementText(ui.element("tripCoverControls")), /запрос мог быть принят/);
  assert.ok(!elementText(ui.element("tripCoverControls")).includes("Создать ИИ-обложку"));
  result = "ready";
  ui.element("tripCoverControls").children.at(-1).dispatch("click");
  await settleCovers();
  await ui.context.submitTripCover(ui.state.coverView);
  assert.deepEqual(calls, ["GET", "POST", "GET"]);
  assert.equal(ui.element("tripHero").children[0].children.find((child) => child.tagName === "IMG").src,
    "/api/trips/101/cover/image");
});

test("disabled, failed and interrupted covers do not expose paid retry; read failures offer GET only", async () => {
  for (const status of ["disabled", "failed", "interrupted"]) {
    const ui = coverBrowser();
    ui.context.api = async () => coverPayload(status, status !== "disabled");
    ui.context.startTripCoverView();
    await settleCovers();
    assert.ok(!elementText(ui.element("tripCoverControls")).includes("Создать ИИ-обложку"));
    assert.equal(ui.timers.size, 0);
    if (status === "disabled") assert.equal(ui.element("tripCoverControls").hidden, true);
  }
  const ui = coverBrowser();
  const calls = [];
  ui.context.api = async (_path, options) => { calls.push(options?.method || "GET"); throw new Error("offline"); };
  ui.context.startTripCoverView();
  await settleCovers();
  assert.match(elementText(ui.element("tripCoverControls")), /Маршрут сохранён/);
  assert.ok(!elementText(ui.element("tripCoverControls")).includes("Создать ИИ-обложку"));
  ui.element("tripCoverControls").children.at(-1).dispatch("click");
  await settleCovers();
  assert.deepEqual(calls, ["GET", "GET"]);
  assert.equal(ui.timers.size, 0);
});

test("unconfirmed cover subject explains free fallback without a paid action", async () => {
  const ui = coverBrowser();
  const methods = [];
  ui.context.api = async (_path, options) => {
    methods.push(options?.method || "GET");
    return { ...coverPayload("empty", false), reason: "no_verified_subject" };
  };
  ui.context.startTripCoverView();
  await settleCovers();
  const text = elementText(ui.element("tripCoverControls"));
  assert.match(text, /Пока нет подтверждённой достопримечательности/);
  assert.match(text, /платная генерация не запускается/);
  assert.ok(!text.includes("Создать ИИ-обложку"));
  assert.deepEqual(methods, ["GET"]);
});

test("card cover hydration uses at most 24 reads with concurrency three and session cache", async () => {
  const ui = coverBrowser();
  let calls = 0;
  let active = 0;
  let peak = 0;
  ui.context.api = async (path, options) => {
    assert.equal(options?.method, undefined);
    calls += 1;
    active += 1;
    peak = Math.max(peak, active);
    await Promise.resolve();
    active -= 1;
    return coverPayload("ready", true, Number(path.split("/")[3]));
  };
  const mounts = Array.from({ length: 30 }, (_, index) => ({ trip: { id: index + 1, destination: "Город" },
    mount: new FakeElement() }));
  await ui.context.hydrateTripCardCovers(mounts, ui.state.coverListRevision);
  assert.equal(calls, 24);
  assert.ok(peak <= 3);
  assert.equal(mounts[0].mount.children.find((child) => child.tagName === "IMG").src, "/api/trips/1/cover/image");
  await ui.context.hydrateTripCardCovers(mounts, ui.state.coverListRevision);
  assert.equal(calls, 24);
  ui.context.clearCoverState();
  assert.equal(ui.state.coverCache.size, 0);
});

test("without declared cover API support opening details and lists cannot make image requests", async () => {
  const ui = browser();
  ui.state.trip = detailTrip();
  ui.context.api = async () => { throw new Error("cover endpoint must not be called"); };
  ui.context.startTripCoverView();
  await ui.context.hydrateTripCardCovers([{ trip: { id: 101 }, mount: new FakeElement() }], 0);
  assert.equal(ui.state.coverView, null);
  assert.equal(ui.element("tripCoverControls").hidden, true);
});

const JOB_ID = "22222222-2222-4222-8222-222222222222";

function jobsBrowser(storage = new Map(), owner = "a".repeat(64)) {
  const ui = creationBrowser();
  ui.state.me.generation_jobs = { enabled: true, owner };
  ui.context.localStorage = {
    setItem: (key, value) => storage.set(key, value),
    getItem: (key) => storage.get(key) ?? null,
    removeItem: (key) => storage.delete(key),
  };
  ui.context.crypto = { randomUUID: () => JOB_KEY };
  const timers = new Map();
  let nextTimer = 0;
  ui.context.setTimeout = (callback) => { timers.set(++nextTimer, callback); return nextTimer; };
  ui.context.clearTimeout = (id) => timers.delete(id);
  ui.context.window.confirm = () => false;
  return Object.assign(ui, { storage, timers, storageKey: `family-trip-generation:${owner}` });
}

function jobStatus(status = "running", tripId = null) {
  return { job_id: JOB_ID, status, trip_id: tripId, message: `Состояние: ${status}. Без повторной генерации.` };
}

function deletionBrowser() {
  const ui = browser();
  ui.state.trips = [detailTrip(101), detailTrip(102)];
  ui.context.renderTrips = () => {};
  ui.context.refreshAccountAccess = async () => true;
  ui.confirmations = [];
  ui.context.window.confirm = (message) => { ui.confirmations.push(message); return true; };
  return ui;
}

test("trip deletion asks about the exact trip and cancellation makes no request", async () => {
  const ui = deletionBrowser();
  ui.context.window.confirm = (message) => { ui.confirmations.push(message); return false; };
  ui.context.api = async () => assert.fail("cancel must not call an endpoint");
  await ui.context.deleteTrip(ui.state.trips[0]);
  assert.equal(ui.state.trips.length, 2);
  assert.match(ui.confirmations[0], /№101.*Байкал/);
  assert.match(ui.confirmations[0], /Отменить удаление.*нельзя/);
  assert.match(ui.confirmations[0], /попытки не возвращаются/);
});

test("confirmed DELETE includes CSRF, removes only its trip and does not restore exhausted quota", async () => {
  const ui = deletionBrowser();
  ui.state.me.access = { allowed: true, owner: false, route_limit: 5, route_remaining: 0 };
  ui.context.document.cookie = "family_trip_csrf=synthetic-csrf";
  let calls = 0;
  ui.context.fetch = async (url, options) => {
    calls += 1;
    assert.equal(url, "/api/trips/101");
    assert.equal(options.method, "DELETE");
    assert.equal(options.headers.get("X-CSRF-Token"), "synthetic-csrf");
    return { ok: true, status: 200, headers: new Headers({ "content-type": "application/json" }),
      json: async () => ({ deleted: true, trip_id: 101, cleanup_pending: false }) };
  };
  await ui.context.deleteTrip(ui.state.trips[0]);
  assert.equal(calls, 1);
  assert.deepEqual(Array.from(ui.state.trips, (x) => x.id), [102]);
  assert.equal(ui.state.me.access.route_remaining, 0);
  assert.ok(ui.state.deletedTrips.has(101));
});

test("double deletion click is single-flight and controls unlock on failure", async () => {
  const ui = deletionBrowser();
  const response = deferred();
  const selected = ui.state.trips[0];
  let calls = 0;
  ui.context.api = async () => { calls += 1; return response.promise; };
  const button = ui.context.createDeleteTripButton(selected);
  const pending = ui.context.deleteTrip(selected, ui.state.me, button);
  assert.equal(button.disabled, true);
  await ui.context.deleteTrip(selected);
  assert.equal(calls, 1);
  response.reject(Object.assign(new Error("Сейчас формируется обложка"), { status: 409 }));
  await pending;
  assert.equal(ui.state.trips.length, 2);
  assert.equal(button.disabled, false);
  assert.equal(ui.state.deletingTrips.size, 0);
  assert.match(ui.toasts.at(-1), /обложка/);
});

test("pending cleanup is a deleted route, not an invitation to regenerate or restore it", async () => {
  const ui = deletionBrowser();
  ui.state.trip = ui.state.trips[0];
  ui.state.coverCache.set(101, { account: ui.state.me });
  ui.context.api = async () => ({ deleted: true, trip_id: 101, cleanup_pending: true });
  await ui.context.deleteTrip(ui.state.trip);
  assert.equal(ui.state.trip, null);
  assert.equal(ui.state.coverCache.has(101), false);
  assert.equal(ui.location.hash, "#trips");
  assert.match(ui.toasts.at(-1), /Поездка удалена.*Очистка/);
});

test("late deletion cannot navigate away from another currently opened route", async () => {
  const ui = deletionBrowser();
  const response = deferred();
  ui.context.api = async () => response.promise;
  const pending = ui.context.deleteTrip(ui.state.trips[0]);
  const newer = ui.state.trips[1];
  ui.state.trip = newer;
  ui.state.viewRevision += 1;
  ui.location.hash = "#trip/102";
  response.resolve({ deleted: true, trip_id: 101 });
  await pending;
  assert.equal(ui.state.trip, newer);
  assert.equal(ui.location.hash, "#trip/102");
});

test("late deletion and detached delete buttons cannot affect another account", async () => {
  const ui = deletionBrowser();
  const response = deferred();
  let calls = 0;
  ui.context.api = async () => { calls += 1; return response.promise; };
  const stale = ui.context.createDeleteTripButton(ui.state.trips[1]);
  const pending = ui.context.deleteTrip(ui.state.trips[0]);
  ui.state.me = { name: "Other account" };
  ui.state.trips = [detailTrip(201)];
  stale.dispatch("click");
  response.resolve({ deleted: true, trip_id: 101 });
  await pending;
  assert.equal(calls, 1);
  assert.deepEqual(ui.state.trips.map((x) => x.id), [201]);
  assert.equal(ui.state.deletedTrips.size, 0);
});

test("late list and detail responses cannot resurrect a confirmed deleted route", async () => {
  const ui = deletionBrowser();
  const list = deferred();
  const detail = deferred();
  const original = [...ui.state.trips];
  ui.context.api = (url, options) => options?.method === "DELETE"
    ? Promise.resolve({ deleted: true, trip_id: 101 }) : url.includes("?limit=") ? list.promise : detail.promise;
  const loadingList = ui.context.loadTrips();
  const loadingDetail = ui.context.openTrip(101);
  await ui.context.deleteTrip(original[0]);
  list.resolve({ trips: original });
  detail.resolve(original[0]);
  await Promise.all([loadingList, loadingDetail]);
  assert.deepEqual(Array.from(ui.state.trips, (x) => x.id), [102]);
  assert.equal(ui.state.trip, null);
  assert.deepEqual(ui.renders, []);
  await ui.context.completeTripCreation(original[0], ui.state.me, ui.state.viewRevision);
  assert.equal(ui.state.trip, null);
  assert.match(elementText(ui.element("savedTripNotice")), /уже удалена/);
});

for (const listFails of [false, true]) {
  test(`deletion while creation refresh is pending cannot announce a saved route (failure=${listFails})`, async () => {
    const ui = deletionBrowser();
    const list = deferred();
    const original = [...ui.state.trips];
    ui.context.api = (url, options) => options?.method === "DELETE"
      ? Promise.resolve({ deleted: true, trip_id: 101 }) : list.promise;
    const completing = ui.context.completeTripCreation(original[0], ui.state.me, ui.state.viewRevision);
    assert.equal(ui.state.trip.id, 101);
    await ui.context.deleteTrip(original[0]);
    if (listFails) list.reject(new Error("List unavailable"));
    else list.resolve({ trips: original });
    await completing;
    assert.equal(ui.state.trip, null);
    assert.equal(ui.state.createdTripId, 0);
    assert.deepEqual(Array.from(ui.state.trips, (x) => x.id), [102]);
    assert.equal(ui.element("savedTripNotice").hidden, true);
    assert.ok(!ui.toasts.some((text) => /готов и сохранён/.test(text)));
  });
}

for (const outcome of ["network", "wrong-id", "missing-confirmation", "not-found"]) {
  test(`unconfirmed deletion (${outcome}) never removes the card or retries DELETE`, async () => {
    const ui = deletionBrowser();
    let calls = 0;
    ui.context.api = async () => {
      calls += 1;
      if (outcome === "network") throw new Error("lost response");
      if (outcome === "not-found") throw Object.assign(new Error("Поездка не найдена"), { status: 404 });
      return outcome === "wrong-id" ? { deleted: true, trip_id: 102 } : { trip_id: 101 };
    };
    await ui.context.deleteTrip(ui.state.trips[0]);
    assert.equal(calls, 1);
    assert.equal(ui.state.trips.length, 2);
    assert.equal(ui.state.deletedTrips.size, 0);
  });
}

test("logout clears deletion tombstones for a later account", async () => {
  const ui = deletionBrowser();
  ui.state.deletedTrips.add(101);
  ui.context.api = async () => ({ ok: true });
  await ui.context.logout();
  assert.equal(ui.state.deletedTrips.size, 0);
  assert.equal(ui.state.deletingTrips.size, 0);
});

test("trip cards and route header expose a labelled delete action", () => {
  const ui = browser();
  const summary = { ...detailTrip(101), checklist: { done: 0, total: 2 }, days_count: 1, currency: "RUB" };
  const card = ui.context.createTripCard(summary);
  assert.match(elementText(card), /Удалить поездку/);
  ui.state.trip = detailTrip(101);
  ui.context.renderTripHeader();
  assert.match(elementText(ui.element("tripHeader")), /Удалить поездку/);
});

function ambiguousCityStatus() {
  return { ...jobStatus("failed"), clarification: { kind: "city", attempt_used: false,
    options: ["Москва, Москва, Россия", "Москва, Тверская область, Россия"] } };
}

test("city choice restores the draft but never submits or stores its private fields", async () => {
  const ui = jobsBrowser();
  let posts = 0;
  ui.element("tripDestination").value = "Москва, Россия";
  ui.element("tripWishes").value = "Личные пожелания";
  ui.context.api = async (_url, options) => { assert.equal(options?.method, "POST"); posts += 1; return ambiguousCityStatus(); };
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.storage.size, 0);
  assert.equal(ui.state.createdTripId, 0);
  const choice = ui.element("savedTripNotice").children.find((el) => el.textContent === "Москва, Москва, Россия");
  assert.ok(choice);
  choice.dispatch("click");
  assert.equal(ui.element("tripDestination").value, "Москва, Москва, Россия");
  assert.equal(ui.element("tripWishes").value, "Личные пожелания");
  assert.equal(ui.element("infoDialog").open, true);
  assert.equal(ui.state.createdTripId, null);
  assert.equal(posts, 1);
  assert.equal(ui.storage.size, 0);
  choice.dispatch("click");
  assert.equal(posts, 1);
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(posts, 2, "only the explicit submit starts another job");
});

test("recovered city clarification opens a fresh form without claiming to restore a lost draft", async () => {
  const storage = new Map([[`family-trip-generation:${"a".repeat(64)}`, JSON.stringify({ key: JOB_KEY, job_id: JOB_ID })]]);
  const ui = jobsBrowser(storage);
  let gets = 0;
  ui.context.api = async (_url, options) => { assert.equal(options, undefined); gets += 1; return ambiguousCityStatus(); };
  await ui.context.resumeGenerationReceipt();
  ui.element("tripWishes").value = "Нельзя считать это сохранённой анкетой";
  ui.element("savedTripNotice").children.find((el) => el.textContent === "Москва, Москва, Россия").dispatch("click");
  assert.equal(gets, 1);
  assert.equal(ui.element("tripWishes").value, "");
  assert.equal(ui.element("tripDestination").value, "Москва, Москва, Россия");
  assert.match(ui.element("newTripMessage").textContent, /Заполните остальные поля/);
});

for (const invalidate of ["form", "view", "account", "logout"]) {
  test(`old city buttons cannot overwrite a changed ${invalidate}`, async () => {
    const ui = jobsBrowser();
    ui.context.api = async () => ambiguousCityStatus();
    await ui.context.submitNewTrip({ preventDefault() {} });
    const choice = ui.element("savedTripNotice").children.find((el) => el.textContent === "Москва, Москва, Россия");
    if (invalidate === "form") ui.context.openNewTripDialog();
    if (invalidate === "view") ui.context.beginViewChange();
    if (invalidate === "account") ui.state.me = { ...ui.state.me };
    if (invalidate === "logout") await ui.context.logout();
    ui.element("tripDestination").value = "Новая анкета";
    choice.dispatch("click");
    assert.equal(ui.element("tripDestination").value, "Новая анкета");
  });
}

test("city options are filtered and unavailable lookup has only a manual edit action", async () => {
  const ui = jobsBrowser();
  const result = ambiguousCityStatus();
  result.clarification.options = ["<img>, X, Y", "https://evil.test, X, Y", "Москва, Москва, Россия", "Москва, Москва, Россия", "Bad\nName, X, Y"];
  ui.context.api = async () => result;
  await ui.context.submitNewTrip({ preventDefault() {} });
  let actions = ui.element("savedTripNotice").children.filter((el) => el.tagName === "BUTTON");
  assert.equal(actions.length, 2);
  assert.equal(actions[0].textContent, "Москва, Москва, Россия");
  ui.context.openNewTripDialog();
  ui.context.api = async () => ({ ...ambiguousCityStatus(), clarification: { kind: "city", attempt_used: false, options: [] } });
  await ui.context.submitNewTrip({ preventDefault() {} });
  actions = ui.element("savedTripNotice").children.filter((el) => el.tagName === "BUTTON");
  assert.equal(actions.length, 1);
  assert.match(actions[0].textContent, /вручную/);
});

test("clarification is not offered for interrupted or generic paid failures", async () => {
  for (const result of [ { ...ambiguousCityStatus(), status: "interrupted" },
    { ...ambiguousCityStatus(), clarification: { kind: "city", attempt_used: true, options: [] } }, jobStatus("failed") ]) {
    const ui = jobsBrowser();
    ui.context.api = async () => result;
    await ui.context.submitNewTrip({ preventDefault() {} });
    assert.equal(ui.element("savedTripNotice").children.filter((el) => el.tagName === "BUTTON").length, 0);
  }
});

test("legacy endpoint passes structured city clarification without another POST", async () => {
  const ui = creationBrowser();
  let calls = 0;
  ui.context.fetch = async () => { calls += 1; return { ok: false, status: 422,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => ({ detail: { message: "Уточните город, попытка не списана", code: "city_ambiguous",
      clarification: ambiguousCityStatus().clarification } }) }; };
  await ui.context.submitNewTrip({ preventDefault() {} });
  const choice = ui.element("savedTripNotice").children.find((el) => el.textContent === "Москва, Москва, Россия");
  assert.ok(choice);
  choice.dispatch("click");
  assert.equal(calls, 1);
  assert.equal(ui.element("tripDestination").value, "Москва, Москва, Россия");
});

test("durable job POST stores only account-bound receipt and blocks repeated submit", async () => {
  const ui = jobsBrowser();
  const calls = [];
  ui.context.fetch = async (url, options) => {
    calls.push({ url, options });
    assert.equal(url, "/api/generation-jobs");
    assert.equal(options.headers.get("Idempotency-Key"), JOB_KEY);
    assert.equal(options.headers.get("X-CSRF-Token"), "synthetic-csrf");
    return { ok: true, status: 202, headers: new Headers({ "content-type": "application/json" }), json: async () => jobStatus() };
  };
  ui.context.document.cookie = "family_trip_csrf=synthetic-csrf";
  await ui.context.submitNewTrip({ preventDefault() {} });
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(calls.length, 1);
  assert.equal(ui.state.creatingTrip, true);
  assert.equal(ui.timers.size, 1);
  assert.equal(ui.element("infoDialog").open, false);
  assert.equal(ui.element("savedTripNotice").hidden, false);
  assert.deepEqual(JSON.parse(ui.storage.get(ui.storageKey)), { key: JOB_KEY, job_id: JOB_ID });
  assert.ok(!ui.storage.get(ui.storageKey).includes("destination"));
});

test("reload resumes GET polling without questionnaire or POST", async () => {
  const storage = new Map([[`family-trip-generation:${"a".repeat(64)}`, JSON.stringify({ key: JOB_KEY, job_id: JOB_ID })]]);
  const ui = jobsBrowser(storage);
  const calls = [];
  ui.context.api = async (url, options) => {
    calls.push(url);
    assert.equal(options, undefined);
    return jobStatus();
  };
  await ui.context.resumeGenerationReceipt();
  assert.deepEqual(calls, [`/api/generation-jobs/${JOB_ID}`]);
  assert.equal(ui.state.creatingTrip, true);
  assert.equal(ui.timers.size, 1);
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(calls.length, 1);
});

test("lost POST response recovers by original key and never sends another POST", async () => {
  const ui = jobsBrowser();
  const calls = [];
  ui.context.api = async (url, options) => {
    calls.push([url, options?.method || "GET"]);
    if (options?.method === "POST") throw new Error("lost response");
    assert.equal(url, `/api/generation-jobs?key=${JOB_KEY}`);
    return jobStatus();
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.deepEqual(calls, [["/api/generation-jobs", "POST"], [`/api/generation-jobs?key=${JOB_KEY}`, "GET"]]);
  assert.equal(JSON.parse(ui.storage.get(ui.storageKey)).job_id, JOB_ID);
});

test("uncertain POST and failed status GET preserve key across reload without automatic retry", async () => {
  const ui = jobsBrowser();
  const calls = [];
  ui.context.api = async (url, options) => {
    calls.push([url, options?.method || "GET"]);
    throw Object.assign(new Error("unreachable"), options?.method ? {} : { status: 404 });
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(calls.filter(([, method]) => method === "POST").length, 1);
  assert.equal(ui.state.creatingTrip, true);
  assert.match(elementText(ui.element("savedTripNotice")), /Повторная генерация не запускалась/);
  const reload = jobsBrowser(ui.storage);
  reload.context.api = async (url, options) => { assert.equal(options, undefined); return jobStatus(); };
  await reload.context.resumeGenerationReceipt();
  assert.equal(reload.state.creatingTrip, true);
});

for (const state of ["failed", "interrupted"]) {
  test(`terminal ${state} restores explanation and does not automatically generate again`, async () => {
    const ui = jobsBrowser();
    ui.context.api = async () => jobStatus(state);
    await ui.context.submitNewTrip({ preventDefault() {} });
    assert.equal(ui.state.createdTripId, 0);
    assert.equal(ui.state.creatingTrip, false);
    assert.equal(ui.storage.size, 0);
    assert.match(elementText(ui.element("savedTripNotice")), new RegExp(state));
    ui.context.api = async () => assert.fail("terminal state must not trigger a second POST");
    await ui.context.submitNewTrip({ preventDefault() {} });
  });
}

test("saved job ID survives both failed reads and reload then uses existing detail endpoint", async () => {
  const ui = jobsBrowser();
  const calls = [];
  ui.context.api = async (url, options) => {
    calls.push([url, options?.method || "GET"]);
    if (url === "/api/generation-jobs") return jobStatus("succeeded", 101);
    throw new Error("detail/list unavailable");
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.state.createdTripId, 101);
  assert.match(elementText(ui.element("savedTripNotice")), /№101.*сохранён.*Карточка пока не открыта.*Список поездок/);
  assert.equal(ui.storage.size, 1);
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(calls.filter(([, method]) => method === "POST").length, 1);
  const reload = jobsBrowser(ui.storage);
  reload.context.api = async (url, options) => {
    assert.equal(options, undefined);
    if (url.startsWith("/api/generation-jobs/")) return jobStatus("succeeded", 101);
    if (url === "/api/trips/101") return detailTrip();
    if (url === "/api/trips?limit=100") return { trips: [] };
    assert.fail(`unexpected GET ${url}`);
  };
  await reload.context.resumeGenerationReceipt();
  assert.equal(reload.state.trip.id, 101);
  assert.equal(reload.state.createdTripId, 101);
  assert.equal(reload.storage.size, 0);
  assert.equal(reload.timers.size, 0);
});

test("account-specific receipt is never read by another account; logout clears pending and late response", async () => {
  const ui = jobsBrowser();
  const response = deferred();
  ui.context.api = (url) => url === "/api/logout" ? Promise.resolve({ ok: true }) : response.promise;
  const pending = ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.storage.size, 1);
  const other = jobsBrowser(ui.storage, "b".repeat(64));
  other.context.api = () => assert.fail("other owner must never poll this receipt");
  await other.context.resumeGenerationReceipt();
  assert.equal(other.state.generationReceipt, null);
  await ui.context.logout();
  assert.equal(ui.storage.size, 0);
  assert.equal(ui.timers.size, 0);
  response.resolve(jobStatus("succeeded", 101));
  await pending;
  assert.equal(ui.state.trip, null);
  assert.equal(ui.state.me, null);
  assert.equal(ui.storage.size, 0);
});

test("storage denied fails before POST; explicit capacity refusal does not leave a pending job", async () => {
  const blocked = jobsBrowser();
  blocked.context.localStorage.setItem = () => { throw new Error("denied"); };
  blocked.context.api = () => assert.fail("must not send paid operation without recoverable key");
  await blocked.context.submitNewTrip({ preventDefault() {} });
  assert.equal(blocked.state.creatingTrip, false);
  assert.match(blocked.element("newTripMessage").textContent, /генерация не запускалась/);
  const ui = jobsBrowser();
  ui.context.api = async () => { throw Object.assign(new Error("Места обработки заняты"), { status: 409 }); };
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.state.creatingTrip, false);
  assert.equal(ui.storage.size, 0);
  assert.equal(ui.timers.size, 0);
  assert.match(ui.element("newTripMessage").textContent, /заняты/);
});

test("expired authentication keeps receipt but stops polling until login", async () => {
  const ui = jobsBrowser();
  ui.storage.set(ui.storageKey, JSON.stringify({ key: JOB_KEY, job_id: JOB_ID }));
  ui.context.api = async () => { throw Object.assign(new Error("login required"), { status: 401 }); };
  await ui.context.resumeGenerationReceipt();
  assert.equal(ui.storage.size, 1);
  assert.equal(ui.timers.size, 0);
});

test("another tab resumes an existing account receipt instead of overwriting its key", async () => {
  const first = jobsBrowser();
  first.context.api = async () => jobStatus();
  await first.context.submitNewTrip({ preventDefault() {} });
  const second = jobsBrowser(first.storage);
  const calls = [];
  second.context.api = async (url, options) => {
    assert.equal(options, undefined);
    calls.push(url);
    return jobStatus();
  };
  await second.context.submitNewTrip({ preventDefault() {} });
  assert.deepEqual(calls, [`/api/generation-jobs/${JOB_ID}`]);
  assert.equal(JSON.parse(first.storage.get(first.storageKey)).key, JOB_KEY);
  const third = jobsBrowser(first.storage);
  third.context.api = second.context.api;
  third.context.openNewTripDialog();
  await Promise.resolve();
  assert.equal(third.state.creatingTrip, true);
  assert.equal(JSON.parse(first.storage.get(first.storageKey)).key, JOB_KEY);
});

test("late completed job preserves the newer page and offers its saved ID", async () => {
  const ui = jobsBrowser();
  ui.context.api = async () => jobStatus();
  await ui.context.submitNewTrip({ preventDefault() {} });
  ui.context.showTrips();
  const receipt = ui.state.generationReceipt;
  const calls = [];
  ui.context.api = async (url) => { calls.push(url); return { trips: [] }; };
  await ui.context.applyGenerationStatus(jobStatus("succeeded", 101), ui.state.me, receipt);
  assert.deepEqual(calls, ["/api/trips?limit=100"]);
  assert.equal(ui.state.trip, null);
  assert.equal(ui.state.createdTripId, 101);
  assert.match(elementText(ui.element("savedTripNotice")), /№101.*сохранён/);
});

test("reload still recovers pending job when trip-list GET fails", async () => {
  const ui = jobsBrowser();
  ui.storage.set(ui.storageKey, JSON.stringify({ key: JOB_KEY, job_id: JOB_ID }));
  const profile = ui.state.me;
  const calls = [];
  ui.context.api = async (url) => {
    calls.push(url);
    if (url === "/api/me") return profile;
    if (url === "/api/trips?limit=100") throw Object.assign(new Error("list unavailable"), { status: 503 });
    if (url === `/api/generation-jobs/${JOB_ID}`) return jobStatus();
    assert.fail(`unexpected ${url}`);
  };
  await ui.context.init();
  assert.equal(ui.element("appShell").hidden, false);
  assert.equal(ui.element("loginGate").hidden, true);
  assert.equal(ui.state.creatingTrip, true);
  assert.ok(calls.includes(`/api/generation-jobs/${JOB_ID}`));
});

test("uncertain request exposes read-only acknowledgement outside modal without another paid POST", async () => {
  const ui = jobsBrowser();
  let posts = 0;
  ui.context.api = async (_url, options) => {
    if (options?.method === "POST") posts += 1;
    throw new Error("connection lost");
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(ui.element("infoDialog").open, false);
  assert.equal(ui.element("savedTripNotice").hidden, false);
  assert.equal(ui.state.creatingTrip, true);
  ui.context.openNewTripDialog();
  assert.equal(ui.element("infoDialog").open, false);
  assert.equal(posts, 1);
  const acknowledge = ui.element("savedTripNotice").children[2].listeners.get("click");
  acknowledge(); // The default confirmation is declined.
  assert.equal(ui.state.creatingTrip, true);
  assert.equal(ui.storage.size, 1);
  ui.context.window.confirm = () => true;
  acknowledge();
  assert.equal(ui.state.creatingTrip, false);
  assert.equal(ui.state.createdTripId, 0);
  assert.equal(ui.storage.size, 0);
  assert.equal(ui.timers.size, 0);
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(posts, 1);
});

test("late saved-job detail cannot clear a newer job receipt or cancel its polling", async () => {
  const ui = jobsBrowser();
  const oldDetail = deferred();
  const detailStarted = deferred();
  const newKey = "33333333-3333-4333-8333-333333333333";
  const newId = "44444444-4444-4444-8444-444444444444";
  let posts = 0;
  let newPolls = 0;
  ui.context.crypto.randomUUID = () => posts === 0 ? JOB_KEY : newKey;
  ui.context.api = async (url, options) => {
    if (url === "/api/generation-jobs" && options?.method === "POST") {
      posts += 1;
      return posts === 1 ? jobStatus("succeeded", 101) : { ...jobStatus(), job_id: newId };
    }
    if (url === "/api/trips/101") { detailStarted.resolve(); return oldDetail.promise; }
    if (url === "/api/trips?limit=100") return { trips: [] };
    if (url === `/api/generation-jobs/${newId}`) {
      newPolls += 1;
      return { ...jobStatus(), job_id: newId };
    }
    assert.fail(`unexpected ${url}`);
  };
  const oldSubmission = ui.context.submitNewTrip({ preventDefault() {} });
  await detailStarted.promise;
  assert.equal(ui.state.createdTripId, 101);
  assert.equal(ui.state.creatingTrip, false);
  ui.context.openNewTripDialog();
  await ui.context.submitNewTrip({ preventDefault() {} });
  const newerReceipt = ui.state.generationReceipt;
  const newerTimer = ui.state.generationTimer;
  assert.equal(newerReceipt.key, newKey);
  assert.equal(ui.state.creatingTrip, true);
  oldDetail.resolve(detailTrip());
  await oldSubmission;
  assert.equal(ui.state.generationReceipt, newerReceipt);
  assert.equal(ui.state.generationTimer, newerTimer);
  assert.equal(ui.state.creatingTrip, true);
  assert.equal(ui.timers.size, 1);
  assert.equal(JSON.parse(ui.storage.get(ui.storageKey)).key, newKey);
  assert.equal(ui.element("savedTripNotice").hidden, false);
  assert.match(elementText(ui.element("savedTripNotice")), /running/);
  await ui.context.pollGenerationReceipt(ui.state.me, newerReceipt);
  assert.equal(newPolls, 1);
  assert.equal(posts, 2, "only the two explicitly submitted operations exist");
});

function guestAccess(remaining = 5, limit = 5) {
  return { owner: false, allowed: true, expires_at: "2099-11-01T10:30:00+00:00", route_remaining: remaining, route_limit: limit, grant_id: "synthetic-grant" };
}

test("invitation totals come from the server for five-attempt and legacy grants", () => {
  const ui = browser();
  for (const [left, total] of [[5, 5], [4, 5], [2, 3], [1, 1]]) {
    ui.state.me.access = guestAccess(left, total);
    ui.context.renderAccessStatus();
    assert.ok(elementText(ui.element("accessStatus")).includes(`Осталось ${left} из ${total} попыток`));
    assert.equal(ui.element("newTripButton").disabled, false);
  }
  ui.state.me.access = guestAccess(4);
  delete ui.state.me.access.route_limit;
  ui.context.renderAccessStatus();
  assert.match(elementText(ui.element("accessStatus")), /Осталось 4 попыток/);
  assert.doesNotMatch(elementText(ui.element("accessStatus")), /из 3|из 5/);
  ui.state.me.access = guestAccess(6, 5);
  ui.context.renderAccessStatus();
  assert.match(elementText(ui.element("accessStatus")), /Остаток попыток уточняется/);
});

function jsonResponse(payload, status = 200) {
  return { ok: status < 400, status, headers: new Headers({ "content-type": "application/json" }), json: async () => payload };
}

test("invitation status displays expiry, lifetime attempt allowance and free reads; owner/open stay distinct", () => {
  const ui = browser();
  ui.state.me.access = guestAccess(2);
  ui.context.renderAccessStatus();
  assert.equal(ui.element("accessStatus").hidden, false);
  assert.match(elementText(ui.element("accessStatus")), /Доступ по приглашению до.*2099.*Осталось 2 из 5 попыток/);
  assert.match(elementText(ui.element("accessStatus")), /за весь срок.*не ежедневно.*PDF не расходуют/);
  assert.equal(ui.element("newTripButton").disabled, false);
  ui.state.me.access = { owner: true, allowed: true, grant_id: "owner", route_remaining: null };
  ui.context.renderAccessStatus();
  assert.equal(elementText(ui.element("accessStatus")).trim(), "Доступ владельца");
  ui.state.me.access = { owner: false, allowed: true, grant_id: "open", route_remaining: null };
  ui.context.renderAccessStatus();
  assert.equal(ui.element("accessStatus").hidden, true);
  ui.state.me.access = guestAccess();
  ui.state.me.access.expires_at = "<img src=x onerror=alert(1)>";
  ui.state.me.access.route_remaining = -1;
  ui.context.renderAccessStatus();
  assert.match(elementText(ui.element("accessStatus")), /срока, указанного в приглашении.*Остаток попыток уточняется/);
  assert.ok(!elementText(ui.element("accessStatus")).includes("<img"));
});

test("zero remaining attempts block only new route operations, never trip opening or PDF links", async () => {
  const ui = browser();
  ui.state.me.access = guestAccess(0);
  ui.context.renderAccessStatus();
  ui.context.openNewTripDialog();
  assert.equal(ui.element("infoDialog").open, false);
  assert.equal(ui.element("newTripButton").disabled, true);
  ui.state.trip = trip(101);
  ui.context.openEditDialog();
  assert.equal(ui.element("editDialog").open, false);
  ui.context.api = () => assert.fail("exhausted route allowance cannot submit a paid request");
  await ui.context.submitNewTrip({ preventDefault() {} });
  await ui.context.submitEdit({ preventDefault() {} });
  assert.match(ui.element("newTripMessage").textContent, /Просмотр поездок и скачивание PDF остаются доступны/);
  const card = ui.context.createTripCard({ id: 101, destination: "Байкал", start_date: "2026-10-28", end_date: "2026-11-03", status: "upcoming",
    days_count: 7, planned_total: 52000, currency: "RUB", checklist: { done: 0, total: 30 } });
  card.children[1].children.at(-1).children[1].dispatch("click");
  assert.equal(ui.location.href, "/api/trips/101/pdf", "PDF remains an authenticated read even with zero route attempts");
  ui.context.api = async () => detailTrip(101);
  assert.equal(await ui.context.openTrip(101), true);
});

test("opening the trip list refreshes access with GET and preserves in-flight account identity", async () => {
  const ui = browser();
  const account = ui.state.me;
  account.access = guestAccess(3);
  const calls = [];
  ui.context.api = async (url, options) => {
    calls.push(url);
    assert.equal(options, undefined);
    assert.equal(url, "/api/me");
    return { name: "Should not replace account", access: guestAccess(1) };
  };
  ui.context.showTrips();
  await ui.state.accessRefresh.promise;
  assert.deepEqual(calls, ["/api/me"]);
  assert.equal(ui.state.me, account);
  assert.equal(account.access.route_remaining, 1);
  assert.match(elementText(ui.element("accessStatus")), /Осталось 1 из 5/);
});

test("late access refresh cannot populate another login or undo a newer access-denied response", async () => {
  for (const outcome of ["new-login", "denied"]) {
    const ui = browser();
    ui.state.me.access = guestAccess(1);
    const response = deferred();
    ui.context.api = () => response.promise;
    const refresh = ui.context.refreshAccountAccess();
    if (outcome === "new-login") ui.state.me = { name: "Other", access: guestAccess(2) };
    else ui.context.markAccessDenied();
    response.resolve({ access: guestAccess(3) });
    await refresh;
    assert.equal(ui.state.me.access.route_remaining, outcome === "new-login" ? 2 : 1);
    if (outcome === "denied") assert.equal(ui.state.me.access.allowed, false);
  }
});

test("a post-operation access refresh supersedes an older in-flight allowance read", async () => {
  const ui = browser();
  const account = ui.state.me;
  account.access = guestAccess(3);
  const oldRead = deferred();
  let count = 0;
  ui.context.api = () => ++count === 1 ? oldRead.promise : Promise.resolve({ access: guestAccess(2) });
  const earlier = ui.context.refreshAccountAccess(account);
  await ui.context.refreshAccountAccess(account, true);
  oldRead.resolve({ access: guestAccess(3) });
  await earlier;
  assert.equal(account.access.route_remaining, 2);
  assert.equal(count, 2);
});

test("failed status refresh labels last-known allowance without retries or fabricating fresh quota", async () => {
  const ui = browser();
  ui.state.me.access = guestAccess(2);
  let calls = 0;
  ui.context.api = async () => { calls += 1; throw new Error("offline"); };
  await ui.context.refreshAccountAccess();
  assert.equal(calls, 1);
  assert.equal(ui.state.me.access.route_remaining, 2);
  assert.match(elementText(ui.element("accessStatus")), /последние полученные данные/);
  ui.context.api = async () => ({ access: guestAccess(1) });
  await ui.context.refreshAccountAccess();
  assert.ok(!elementText(ui.element("accessStatus")).includes("последние полученные данные"));
});

test("real API recognizes access_denied separately from quota and never logs out", async () => {
  const ui = browser();
  const account = ui.state.me;
  account.access = guestAccess(0);
  const calls = [];
  ui.context.fetch = async (url) => { calls.push(url); return jsonResponse({ code: "access_quota", detail: "Все попытки использованы" }, 403); };
  await assert.rejects(ui.context.api("/api/trips"), (error) => error.code === "access_quota");
  assert.equal(account.access.allowed, true);
  ui.context.fetch = async (url) => { calls.push(url); return jsonResponse({ code: "access_denied", detail: "Доступ закрыт" }, 403); };
  await assert.rejects(ui.context.api("/api/trips"), (error) => error.code === "access_denied" && /новое приглашение/.test(error.message));
  assert.equal(ui.state.me, account);
  assert.equal(account.access.allowed, false);
  assert.equal(ui.element("accessDeniedNotice").hidden, false);
  assert.ok(!calls.includes("/api/logout"));
});

test("late forbidden response from an earlier account cannot close a newer account", async () => {
  const ui = browser();
  ui.state.me.access = guestAccess();
  const response = deferred();
  ui.context.fetch = () => response.promise;
  const pending = ui.context.api("/api/trips");
  const current = { name: "New login", access: guestAccess() };
  ui.state.me = current;
  response.resolve(jsonResponse({ code: "access_denied", detail: "old grant" }, 403));
  await assert.rejects(pending);
  assert.equal(ui.state.me, current);
  assert.equal(current.access.allowed, true);
});

test("initial access denial shows invitation guidance, not expired-login instructions or automatic retry", async () => {
  const ui = browser();
  ui.state.me = null;
  const calls = [];
  ui.context.fetch = async (url) => { calls.push(url); return jsonResponse({ code: "access_denied", detail: "expired" }, 403); };
  await ui.context.init();
  assert.deepEqual(calls, ["/api/me"]);
  assert.equal(ui.element("accessGate").hidden, false);
  assert.equal(ui.element("loginGate").hidden, true);
  assert.equal(ui.element("appShell").hidden, true);
  assert.match(ui.element("accessGateMessage").textContent, /новое приглашение.*\/web/);
});

test("revoked access stops durable polling, retains receipt, and never creates a replacement job", async () => {
  const ui = jobsBrowser();
  ui.state.me.access = guestAccess();
  ui.storage.set(ui.storageKey, JSON.stringify({ key: JOB_KEY, job_id: JOB_ID }));
  const calls = [];
  ui.context.fetch = async (url, options) => {
    calls.push([url, options.method]);
    return jsonResponse({ code: "access_denied", detail: "revoked" }, 403);
  };
  await ui.context.resumeGenerationReceipt();
  assert.equal(ui.timers.size, 0);
  assert.equal(ui.state.creatingTrip, false);
  assert.equal(ui.storage.size, 1);
  assert.match(elementText(ui.element("savedTripNotice")), /Проверка заявки остановлена/);
  ui.context.openNewTripDialog();
  await ui.context.submitNewTrip({ preventDefault() {} });
  assert.equal(calls.length, 1);
  assert.ok(calls.every(([, method]) => method === "GET"));
});

test("accepted durable generation refreshes its consumed allowance without another POST", async () => {
  const ui = jobsBrowser();
  ui.state.me.access = guestAccess(3);
  const calls = [];
  ui.context.api = async (url, options) => {
    calls.push([url, options?.method || "GET"]);
    return url === "/api/me" ? { access: guestAccess(2) } : jobStatus();
  };
  await ui.context.submitNewTrip({ preventDefault() {} });
  if (ui.state.accessRefresh) await ui.state.accessRefresh.promise;
  assert.equal(ui.state.me.access.route_remaining, 2);
  assert.deepEqual(calls, [["/api/generation-jobs", "POST"], ["/api/me", "GET"]]);
});

test("successful and failed edits refresh allowance without replacing the guarded account", async () => {
  for (const failed of [false, true]) {
    const ui = browser();
    const account = ui.state.me;
    account.access = guestAccess(3);
    const calls = [];
    ui.context.api = async (url, options) => {
      calls.push([url, options?.method || "GET"]);
      if (url === "/api/me") return { access: guestAccess(2) };
      if (failed) throw Object.assign(new Error("Проверка не пройдена"), { status: 422 });
      return trip(101, 2);
    };
    await startEdit(ui, 101, 2);
    if (ui.state.accessRefresh) await ui.state.accessRefresh.promise;
    assert.equal(ui.state.me, account);
    assert.equal(account.access.route_remaining, 2);
    assert.equal(calls.filter(([url]) => url === "/api/me").length, 1);
  }
});
