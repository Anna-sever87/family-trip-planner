"use strict";

const state = {
  me: null,
  trips: [],
  trip: null,
  selectedDay: 1,
  filter: "all",
  search: "",
  pageSize: 8,
  yearPages: new Map(),
  openYears: new Set(),
  closedYears: new Set(),
  creatingTrip: false,
  createdTripId: null,
  generationReceipt: null,
  generationTimer: null,
  cityClarification: null,
  newTripFormRevision: 0,
  editingTrip: false,
  viewRevision: 0,
  stopEditWaiting: null,
  map: null,
  coverCache: new Map(),
  coverView: null,
  coverCardMounts: [],
  coverListRevision: 0,
  accessRefresh: null,
  accessRevision: 0,
  accessUnconfirmed: false,
  deletingTrips: new Map(),
  deletedTrips: new Set(),
};

const $ = (id) => document.getElementById(id);

const DESTINATION_KIND_COPY = Object.freeze({
  city: {
    label: "Город и страна",
    placeholder: "Например: Казань, Россия",
    hint: "Укажите город и страну. Для одноимённых городов добавьте регион: Москва, Москва, Россия. Для городов-государств: Сингапур, Сингапур или Ватикан, Ватикан. Маршрут планируется в городском радиусе.",
    goal: "Познакомиться с городом",
    goalPlaceholder: "Например: познакомиться с городом и отдохнуть",
  },
  region: {
    label: "Регион, озеро или территория",
    placeholder: "Например: Байкал, Россия",
    hint: "Для озёр и водных территорий используется сам контур и ближайшая береговая зона до 5 км; для остальных регионов — допуск до 40 км. Это технический допуск, не официальная граница. Маршрут может включать несколько зон выбранной территории. Базы дней и переезды ориентировочные: время, сезонную доступность и стоимость нужно проверить отдельно — маршрут не подтверждает бронирование.",
    goal: "Познакомиться с территорией",
    goalPlaceholder: "Например: увидеть природу Байкала и отдохнуть",
  },
});

function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== "") element.textContent = String(text);
  return element;
}

// Reviewed local illustrations only. A destination string never becomes a URL.
// These covers depict autumn, not live weather or proof of a place's appearance.
const DESTINATION_COVERS = Object.freeze([
  Object.freeze({ file: "moscow", name: "Москвы", aliases: ["москва", "москва,россия", "moscow", "moscow,russia"] }),
  Object.freeze({ file: "sochi", name: "Сочи", aliases: ["сочи", "сочи,россия", "sochi", "sochi,russia"] }),
  Object.freeze({ file: "baikal", name: "Байкала", aliases: ["байкал", "байкал,россия", "озеро байкал", "озеро байкал,россия", "baikal", "baikal,russia", "lake baikal", "lake baikal,russia"] }),
  Object.freeze({ file: "nha-trang", name: "Нячанга", aliases: ["нячанг", "нячанг,вьетнам", "nha trang", "nha trang,vietnam"] }),
]);

function destinationCover(trip) {
  const destination = String(trip.destination || "").normalize("NFKC")
    .trim().toLocaleLowerCase("ru-RU").replace(/\s*,\s*/g, ",").replace(/\s+/g, " ");
  const dates = [trip.start_date, trip.end_date].map((date) => String(date || ""));
  // Do not show an autumn scene for a winter or summer itinerary.
  if (!dates.every((date) => /^\d{4}-(10|11)-\d{2}$/.test(date))) return null;
  const cover = DESTINATION_COVERS.find((item) => item.aliases.includes(destination));
  return cover ? { src: `/static/images/${cover.file}.png`,
    alt: `Художественная ИИ-иллюстрация ${cover.name}; не фотография`,
    caption: "ИИ-иллюстрация · не фотография" } : null;
}

function savedCoverAsset(trip, status) {
  const id = Number(trip.id);
  if (!Number.isSafeInteger(id) || id <= 0 || status?.status !== "ready" || status.hidden === true
      || status.image_url !== `/api/trips/${id}/cover/image`) return null;
  return { src: status.image_url, alt: `ИИ-иллюстрация направления ${trip.destination}; не фотография`,
    caption: "ИИ-иллюстрация · не фотография" };
}

function createTripCover(trip, detail = false, savedStatus = null) {
  const cover = node("div", detail ? "trip-detail-cover" : "trip-card-cover");
  const asset = savedCoverAsset(trip, savedStatus) || destinationCover(trip);
  const placeholder = node("div", "trip-cover-placeholder");
  placeholder.setAttribute("aria-hidden", "true");
  placeholder.append(node("span", "atlas-symbol", "⌖"), node("span", "atlas-route", "● · · · · · ✦"));
  const caption = node("span", "cover-caption", asset?.caption || "Иллюстрация путешествия");
  placeholder.hidden = Boolean(asset);
  cover.append(placeholder);
  if (asset) {
    const photo = node("img", "trip-cover-image");
    photo.src = asset.src;
    photo.alt = asset.alt;
    photo.loading = detail ? "eager" : "lazy";
    photo.decoding = "async";
    photo.width = 1536;
    photo.height = 1024;
    photo.addEventListener("error", () => {
      photo.hidden = true;
      placeholder.hidden = false;
      cover.classList.add("cover-is-generic");
      caption.textContent = "Иллюстрация путешествия";
    }, { once: true });
    cover.append(photo);
  } else cover.classList.add("cover-is-generic");
  cover.append(caption);
  return cover;
}

const COVER_STATES = new Set(["disabled", "empty", "queued", "running", "ready", "failed", "interrupted"]);
const COVER_POLL_LIMIT = 60;
const COVER_COPY = Object.freeze({
  empty: "Для этой поездки можно создать отдельную ИИ-иллюстрацию направления.",
  queued: "Обложка ожидает создания. Маршрут уже сохранён и доступен.",
  running: "ИИ создаёт обложку. Можно закрыть страницу — маршрут доступен без неё.",
  ready: "Обложка сохранена. Просмотр не запускает повторную генерацию.",
  failed: "Обложку создать не удалось. Маршрут сохранён; платный запрос автоматически не повторяется.",
  interrupted: "Создание обложки прервано или результат не подтверждён. Новый платный запрос не отправляется.",
});

function coverFeatureAvailable() {
  return state.me?.access?.allowed !== false && state.me?.capabilities?.cover_images === true;
}

function coverStatusValue(payload, tripId) {
  if (!payload || !COVER_STATES.has(payload.status) || typeof payload.available !== "boolean"
      || (payload.hidden !== undefined && typeof payload.hidden !== "boolean")
      || (payload.status === "ready" && (payload.hidden === true ? payload.image_url !== null
        : !savedCoverAsset({ id: tripId }, payload)))) {
    throw new Error("Не удалось проверить состояние обложки.");
  }
  return { status: payload.status, available: payload.available, hidden: payload.hidden === true,
    reason: payload.reason === "no_verified_subject" ? "no_verified_subject" : "",
    image_url: payload.status === "ready" && payload.hidden !== true ? payload.image_url : null };
}

function cachedCover(tripId) {
  const item = state.coverCache.get(Number(tripId));
  return item?.account === state.me ? item : null;
}

function rememberCover(tripId, status, account) {
  if (state.me === account) state.coverCache.set(Number(tripId), { account, value: status, checkedAt: Date.now() });
}

function stopCoverUpdates() {
  if (state.coverView?.timer !== null && state.coverView?.timer !== undefined) clearTimeout(state.coverView.timer);
  state.coverView = null;
  state.coverListRevision += 1;
}

function clearCoverState() {
  stopCoverUpdates();
  state.coverCache.clear();
  state.coverCardMounts = [];
  $("tripCoverControls").replaceChildren();
  $("tripCoverControls").hidden = true;
}

function currentCoverView(session) {
  return state.coverView === session && state.me === session.account
    && state.trip?.id === session.tripId && state.viewRevision === session.viewRevision;
}

function renderCoverControls(session) {
  if (!currentCoverView(session)) return;
  const panel = $("tripCoverControls");
  panel.replaceChildren();
  const status = session.status;
  panel.hidden = status?.status === "disabled";
  if (panel.hidden) return;
  const text = session.error || (session.visibilityBusy ? "Сохраняю видимость обложки. Новая генерация не запускается…"
    : session.posting ? "Отправляю один запрос на создание обложки…"
    : status?.status === "ready" && status.hidden ? "ИИ-обложка скрыта. Её можно вернуть без новой генерации."
    : status?.reason === "no_verified_subject" ? "Пока нет подтверждённой достопримечательности для обложки. Используется нейтральное оформление; платная генерация не запускается."
    : status ? COVER_COPY[status.status] : "Проверяю сохранённую обложку…");
  panel.append(node("p", "cover-status", text));
  if (status?.status === "ready") {
    const visibility = button(status.hidden ? "Вернуть ИИ-обложку" : "Скрыть ИИ-обложку",
      "button button-ghost button-small", () => { void setTripCoverHidden(session, !status.hidden); });
    visibility.disabled = session.visibilityBusy || session.reading || session.posting;
    panel.append(visibility);
  }
  if (status?.status === "empty" && status.available && !session.submitted && !session.error) {
    const note = node("p", "cover-billing-note", "Создание — отдельный платный запрос к ИИ по тарифу сервиса. Готовая обложка сохраняется; её просмотр не запускает генерацию. Изображение будет художественной иллюстрацией, не фотографией.");
    note.id = "coverBillingNote";
    panel.append(note);
    if (session.confirming) {
      const consent = node("div", "cover-consent");
      consent.setAttribute("role", "group");
      consent.setAttribute("aria-labelledby", "coverConsentQuestion");
      consent.setAttribute("aria-describedby", "coverBillingNote");
      const question = node("p", "cover-consent-question", `Создать обложку для «${state.trip.destination}»? Будет отправлен один платный запрос. Маршрут не изменится.`);
      question.id = "coverConsentQuestion";
      question.tabIndex = -1;
      const actions = node("div", "cover-consent-actions");
      const confirm = button("Подтвердить генерацию", "button button-primary button-small", () => { void submitTripCover(session); });
      confirm.disabled = session.posting || session.reading;
      const cancel = button("Отмена", "button button-ghost button-small", () => cancelTripCoverConfirmation(session));
      actions.append(confirm, cancel);
      consent.append(question, actions);
      consent.addEventListener("keydown", (event) => {
        if (event.key === "Escape") { event.preventDefault(); cancelTripCoverConfirmation(session); }
      });
      panel.append(consent);
    } else {
      const control = button("Создать ИИ-обложку", "button button-ghost button-small", () => requestTripCoverConfirmation(session));
      control.id = "createCoverButton";
      control.disabled = session.posting || session.reading;
      control.setAttribute("aria-describedby", "coverBillingNote");
      panel.append(control);
    }
  }
  if (session.paused || session.error) {
    panel.append(button("Проверить статус обложки", "button button-ghost button-small", () => {
      if (!currentCoverView(session) || session.reading || session.posting || session.visibilityBusy) return;
      session.polls = 0;
      session.paused = false;
      session.error = "";
      void refreshTripCover(session);
    }));
  }
}

function showSavedCover(session) {
  if (!currentCoverView(session) || session.status?.status !== "ready") return;
  $("tripHero").replaceChildren(createTripCover(state.trip, true, session.status));
}

function scheduleCoverRead(session) {
  if (!currentCoverView(session)) return;
  if (session.polls >= COVER_POLL_LIMIT) {
    session.paused = true;
    session.error = "Автоматическая проверка обложки завершена. Можно проверить её статус вручную — это не запускает новую генерацию.";
    renderCoverControls(session);
    return;
  }
  session.timer = setTimeout(() => {
    session.timer = null;
    void refreshTripCover(session);
  }, 3000);
}

async function refreshTripCover(session) {
  if (!currentCoverView(session) || session.reading || session.posting || session.visibilityBusy) return;
  session.reading = true;
  session.polls += 1;
  renderCoverControls(session);
  try {
    const status = coverStatusValue(await api(`/api/trips/${session.tripId}/cover`), session.tripId);
    if (!currentCoverView(session)) return;
    session.status = status;
    session.error = "";
    rememberCover(session.tripId, status, session.account);
    showSavedCover(session);
    renderCoverControls(session);
    if (["queued", "running"].includes(status.status)) scheduleCoverRead(session);
  } catch (_) {
    if (!currentCoverView(session)) return;
    session.error = "Состояние обложки временно недоступно. Маршрут сохранён. Повторный запрос генерации не отправлялся.";
    session.paused = true;
    renderCoverControls(session);
  } finally {
    session.reading = false;
    if (currentCoverView(session)) renderCoverControls(session);
  }
}

function startTripCoverView() {
  if (!coverFeatureAvailable() || !state.trip) {
    $("tripCoverControls").hidden = true;
    return;
  }
  if (state.coverView && currentCoverView(state.coverView)) {
    renderCoverControls(state.coverView);
    return;
  }
  const session = { tripId: state.trip.id, account: state.me, viewRevision: state.viewRevision,
    status: cachedCover(state.trip.id)?.value || null, timer: null, polls: 0,
    posting: false, reading: false, visibilityBusy: false, confirming: false, submitted: false, paused: false, error: "" };
  state.coverView = session;
  renderCoverControls(session);
  void refreshTripCover(session);
}

function requestTripCoverConfirmation(session) {
  if (!currentCoverView(session) || session.posting || session.reading || session.submitted
      || session.confirming || session.status?.status !== "empty" || session.status.available !== true) return;
  session.confirming = true;
  renderCoverControls(session);
  $("coverConsentQuestion").focus();
}

function cancelTripCoverConfirmation(session) {
  if (!currentCoverView(session) || session.posting || session.submitted) return;
  session.confirming = false;
  renderCoverControls(session);
  $("createCoverButton").focus();
}

async function submitTripCover(session) {
  if (state.deletingTrips.has(session.tripId) || state.deletedTrips.has(session.tripId)) return;
  if (!currentCoverView(session) || !session.confirming || session.posting || session.reading || session.submitted
      || session.status?.status !== "empty" || session.status.available !== true) return;
  session.confirming = false;
  session.submitted = true;
  session.posting = true;
  session.error = "";
  renderCoverControls(session);
  try {
    const status = coverStatusValue(await api(`/api/trips/${session.tripId}/cover`, { method: "POST" }), session.tripId);
    if (!currentCoverView(session)) return;
    session.status = status;
    rememberCover(session.tripId, status, session.account);
    showSavedCover(session);
    if (["queued", "running"].includes(status.status)) scheduleCoverRead(session);
  } catch (error) {
    if (!currentCoverView(session)) return;
    session.error = error.status ? error.message
      : "Ответ на запрос обложки не получен. Проверьте статус: запрос мог быть принят. Повторная генерация не отправляется.";
    session.paused = true;
  } finally {
    session.posting = false;
    if (currentCoverView(session)) renderCoverControls(session);
  }
}

async function setTripCoverHidden(session, hidden) {
  if (state.deletingTrips.has(session.tripId) || state.deletedTrips.has(session.tripId)) return;
  if (!currentCoverView(session) || session.reading || session.posting || session.visibilityBusy
      || session.status?.status !== "ready" || typeof hidden !== "boolean"
      || session.status.hidden === hidden) return;
  session.visibilityBusy = true;
  session.error = "";
  state.coverCache.delete(Number(session.tripId));
  renderCoverControls(session);
  try {
    const status = coverStatusValue(await api(`/api/trips/${session.tripId}/cover`, {
      method: "PATCH", body: JSON.stringify({ hidden }),
    }), session.tripId);
    if (!currentCoverView(session)) return;
    if (status.status !== "ready" || status.hidden !== hidden) throw new Error("Видимость не подтверждена.");
    session.status = status;
    session.paused = false;
    rememberCover(session.tripId, status, session.account);
    showSavedCover(session);
  } catch (error) {
    if (!currentCoverView(session)) return;
    session.error = error.status ? error.message
      : "Не удалось подтвердить изменение видимости. Проверьте статус обложки; новая генерация не запускалась.";
    session.paused = true;
  } finally {
    session.visibilityBusy = false;
    if (currentCoverView(session)) renderCoverControls(session);
  }
}

async function hydrateTripCardCovers(mounts, revision) {
  if (!coverFeatureAvailable() || $("tripsView").hidden) return;
  const account = state.me;
  const valid = () => state.me === account && state.coverListRevision === revision;
  const queue = mounts.slice(0, 24);
  let next = 0;
  const worker = async () => {
    while (valid() && next < queue.length) {
      const { trip, mount } = queue[next++];
      try {
        const cache = cachedCover(trip.id);
        const status = cache && Date.now() - cache.checkedAt < 60000 ? cache.value
          : coverStatusValue(await api(`/api/trips/${trip.id}/cover`), trip.id);
        if (!valid()) return;
        rememberCover(trip.id, status, account);
        if (status.status === "ready") {
          const replacement = createTripCover(trip, false, status);
          mount.className = replacement.className;
          mount.replaceChildren(...replacement.children);
        }
      } catch (_) { /* The trip card and its neutral/static cover remain usable. Never POST here. */ }
    }
  };
  await Promise.all(Array.from({ length: Math.min(3, queue.length) }, worker));
}

function mapSourceLink(point) {
  if (point.verified !== true || typeof point.source_url !== "string") return null;
  const is2gis = point.provider === "2gis" && /^https:\/\/2gis\.ru\/[a-z-]+\/firm\/[1-9][0-9]{0,31}$/.test(point.source_url);
  const isOsm = point.provider === "osm" && /^https:\/\/www\.openstreetmap\.org\/\?mlat=-?[0-9.]+&mlon=-?[0-9.]+#map=17\/-?[0-9.]+\/-?[0-9.]+$/.test(point.source_url);
  if (!is2gis && !isOsm) return null;
  const link = node("a", "map-source", is2gis ? "Источник координат: 2ГИС" : "Открыть на OpenStreetMap");
  link.href = point.source_url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function button(text, className, action) {
  const element = node("button", className, text);
  element.type = "button";
  element.addEventListener("click", action);
  return element;
}

function getCookie(name) {
  const prefix = `${name}=`;
  const part = document.cookie.split("; ").find((item) => item.startsWith(prefix));
  return part ? decodeURIComponent(part.slice(prefix.length)) : "";
}

function apiErrorMessage(payload, fallback) {
  const detail = payload && payload.detail;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail.message === "string") {
    const problems = Array.isArray(detail.problems) ? detail.problems : [];
    return [detail.message, ...problems.map((item) => `• ${item}`)].join("\n");
  }
  if (Array.isArray(detail)) return detail.map((item) => item.msg || String(item)).join("\n");
  return fallback;
}

const ACCESS_DENIED_MESSAGE = "Доступ по приглашению закончился или был отозван. Попросите владельца выдать новое приглашение, затем откройте новую ссылку /web из Telegram-бота. Повторный вход сам по себе не продлевает доступ.";

function isAccessDenied(error) {
  return error?.status === 403 && error?.code === "access_denied";
}

function accessExpiryLabel(value) {
  const date = typeof value === "string" ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return "срока, указанного в приглашении";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit",
    timeZone: "UTC", timeZoneName: "short",
  }).format(date);
}

function routeAccessProblem() {
  const access = state.me?.access;
  if (access?.allowed === false) return ACCESS_DENIED_MESSAGE;
  if (access?.owner !== true && access?.grant_id !== "open" && access?.route_remaining === 0) {
    return "Все попытки создания и ИИ-изменения поездки по приглашению использованы. Попросите владельца выдать новое приглашение. Просмотр поездок и скачивание PDF остаются доступны до окончания доступа.";
  }
  return "";
}

function renderAccessStatus() {
  const access = state.me?.access;
  const panel = $("accessStatus");
  panel.replaceChildren();
  panel.hidden = !access || (access.grant_id === "open" && access.allowed !== false);
  const denied = access?.allowed === false;
  $("accessDeniedNotice").hidden = !denied;
  $("accessDeniedNotice").textContent = denied ? ACCESS_DENIED_MESSAGE : "";
  $("newTripButton").disabled = Boolean(routeAccessProblem());
  if (panel.hidden) return;
  if (denied) {
    panel.append(node("strong", "", "Доступ по приглашению завершён"));
    return;
  }
  if (access.owner === true) {
    panel.append(node("strong", "", "Доступ владельца"));
    return;
  }
  const limit = Number.isInteger(access.route_limit) && access.route_limit >= 1 && access.route_limit <= 5
    ? access.route_limit : null;
  const validRemaining = Number.isInteger(access.route_remaining) && access.route_remaining >= 0
    && access.route_remaining <= (limit ?? 5);
  const remaining = validRemaining
    ? (limit === null ? `Осталось ${access.route_remaining} попыток` : `Осталось ${access.route_remaining} из ${limit} попыток`)
    : "Остаток попыток уточняется";
  panel.append(
    node("strong", "", `Доступ по приглашению до ${accessExpiryLabel(access.expires_at)} • ${remaining}`),
    node("p", "", "Попытки создания и изменения маршрутов считаются за весь срок доступа, не ежедневно. Просмотр поездок и PDF не расходуют попытки."),
  );
  if (state.accessUnconfirmed) panel.append(node("p", "access-status-warning", "Показаны последние полученные данные: сейчас обновить статус доступа не удалось."));
}

function markAccessDenied(account = state.me) {
  if (!account || state.me !== account) return;
  state.accessRevision += 1;
  state.accessUnconfirmed = false;
  account.access = { ...account.access, allowed: false };
  if (state.generationTimer !== null) clearTimeout(state.generationTimer);
  state.generationTimer = null;
  stopCoverUpdates();
  $("tripCoverControls").hidden = true;
  renderAccessStatus();
}

async function refreshAccountAccess(account = state.me, force = false) {
  if (!account?.access || state.me !== account) return false;
  if (!force && state.accessRefresh?.account === account) return state.accessRefresh.promise;
  const pending = { account, revision: state.accessRevision, promise: null };
  state.accessRefresh = pending;
  pending.promise = (async () => {
    try {
      const profile = await api("/api/me");
      if (state.me !== account || state.accessRevision !== pending.revision || state.accessRefresh !== pending) return false;
      if (!profile?.access || typeof profile.access.allowed !== "boolean") throw new Error("Access status unavailable");
      // Keep the account object: in-flight job and cover guards rely on its identity.
      account.access = profile.access;
      state.accessUnconfirmed = false;
      renderAccessStatus();
      return true;
    } catch (error) {
      if (state.me !== account || state.accessRevision !== pending.revision || state.accessRefresh !== pending) return false;
      if (isAccessDenied(error)) markAccessDenied(account);
      else { state.accessUnconfirmed = true; renderAccessStatus(); }
      return false;
    } finally {
      if (state.accessRefresh === pending) state.accessRefresh = null;
    }
  })();
  return pending.promise;
}

async function api(path, options = {}) {
  const account = state.me;
  const method = (options.method || "GET").toUpperCase();
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
    headers.set("X-CSRF-Token", getCookie("family_trip_csrf"));
  }

  const response = await fetch(path, { ...options, method, headers, credentials: "same-origin" });
  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) payload = await response.json();
  if (!response.ok) {
    const error = new Error(apiErrorMessage(payload, `Ошибка ${response.status}`));
    error.status = response.status;
    error.code = typeof payload?.code === "string" ? payload.code : payload?.detail?.code;
    if (error.status === 422) error.clarification = payload?.detail?.clarification;
    if (isAccessDenied(error)) {
      error.message = ACCESS_DENIED_MESSAGE;
      markAccessDenied(account);
    }
    if (payload?.detail?.saved === true && Number.isInteger(payload.detail.trip_id) && payload.detail.trip_id > 0) {
      error.savedTripId = payload.detail.trip_id;
    }
    throw error;
  }
  return payload;
}

function setLoading(active, text = "Загружаю…") {
  $("loadingText").textContent = text;
  $("loading").hidden = !active;
}

function startAIWait(messageElement, initialText, delayedText) {
  const startedAt = performance.now();
  const message = node("p", "", initialText);
  const elapsed = node("p", "generation-elapsed");
  elapsed.setAttribute("aria-hidden", "true");
  messageElement.replaceChildren(message, elapsed);
  messageElement.classList.add("is-waiting");
  let delayed = false;
  const update = () => {
    const seconds = Math.floor((performance.now() - startedAt) / 1000);
    const time = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    elapsed.textContent = `Время ожидания: ${time}`;
    if (seconds >= 60 && !delayed) {
      message.textContent = delayedText;
      delayed = true;
    }
    $("loadingText").textContent = `${delayed ? "Проверка продолжается…" : initialText} · ${time}`;
  };
  setLoading(true, initialText);
  update();
  const timer = setInterval(update, 1000);
  return () => {
    clearInterval(timer);
    messageElement.classList.remove("is-waiting");
  };
}

let toastTimer = null;
function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4200);
}

function formatDate(value, withYear = true) {
  const date = new Date(`${value}T12:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("ru-RU", {
    day: "numeric",
    month: "long",
    ...(withYear ? { year: "numeric" } : {}),
  }).format(date);
}

function formatMoney(value, currency) {
  try {
    return new Intl.NumberFormat("ru-RU", {
      maximumFractionDigits: 2,
      minimumFractionDigits: 0,
      style: "currency",
      currency,
    }).format(Number(value));
  } catch (_) {
    return `${Number(value).toLocaleString("ru-RU")} ${currency}`;
  }
}

function statusLabel(status) {
  return {
    upcoming: "Предстоит",
    active: "Сейчас в поездке",
    past: "Завершена",
  }[status] || "Поездка";
}

function tripYear(trip) {
  const match = String(trip.start_date || "").match(/^(\d{4})-/);
  return match ? match[1] : "Без даты";
}

function tripCountLabel(count) {
  const value = Math.abs(Number(count));
  const lastTwo = value % 100;
  const last = value % 10;
  if (lastTwo >= 11 && lastTwo <= 14) return `${value} поездок`;
  if (last === 1) return `${value} поездка`;
  if (last >= 2 && last <= 4) return `${value} поездки`;
  return `${value} поездок`;
}

function groupTripsByYear(trips) {
  const groups = new Map();
  trips.forEach((trip) => {
    const year = tripYear(trip);
    if (!groups.has(year)) groups.set(year, []);
    groups.get(year).push(trip);
  });
  return [...groups.entries()];
}

async function loadTrips() {
  const account = state.me;
  setLoading(true, "Загружаю поездки…");
  try {
    const payload = await api("/api/trips?limit=100");
    if (state.me !== account) return;
    state.trips = visibleTrips(payload.trips);
    renderTrips();
  } finally {
    if (state.me === account) setLoading(false);
  }
}

function visibleTrips(trips) {
  return Array.isArray(trips) ? trips.filter((trip) => !state.deletedTrips.has(Number(trip.id))) : [];
}

function createDeleteTripButton(trip) {
  const account = state.me;
  const control = button("Удалить поездку", "button button-delete", () => deleteTrip(trip, account, control));
  control.disabled = state.deletingTrips.has(Number(trip.id));
  control.setAttribute("aria-label", `Удалить поездку №${trip.id}: ${trip.destination}`);
  return control;
}

async function deleteTrip(trip, account = state.me, control = null) {
  const id = Number(trip?.id);
  if (!account || state.me !== account || !Number.isSafeInteger(id) || id <= 0
      || state.deletingTrips.has(id) || state.deletedTrips.has(id)) return;
  if (state.me.access?.allowed === false) { toast(ACCESS_DENIED_MESSAGE); return; }
  if (state.creatingTrip || state.editingTrip) { toast("Дождитесь завершения создания или изменения маршрута."); return; }
  const question = `Удалить поездку №${id} «${trip.destination}»?\n`
    + `${formatDate(trip.start_date)} — ${formatDate(trip.end_date)}.\n\n`
    + "Маршрут, чек-лист, серверная обложка и PDF будут удалены. Отменить удаление в кабинете нельзя. "
    + "Общая память семьи и скачанные вами файлы останутся. Потраченные попытки не возвращаются.";
  if (!window.confirm(question) || state.me !== account) return;
  const operation = { account };
  state.deletingTrips.set(id, operation);
  if (control) { control.disabled = true; control.textContent = "Удаляю…"; }
  try {
    let result;
    try {
      result = await api(`/api/trips/${id}`, { method: "DELETE" });
    } catch (error) {
      if (state.me === account) toast(error.status ? error.message
        : "Не удалось подтвердить удаление. Обновите список поездок перед повторной попыткой.");
      return;
    }
    if (state.me !== account || state.deletingTrips.get(id) !== operation) return;
    if (result?.deleted !== true || result.trip_id !== id) {
      toast("Ответ об удалении не подтверждён. Обновите список поездок.");
      return;
    }
    // A late list/detail response must not resurrect a confirmed deleted trip.
    // These account-local tombstones are never persisted outside this tab.
    state.deletedTrips.add(id);
    state.trips = visibleTrips(state.trips);
    state.coverCache.delete(id);
    if (state.createdTripId === id) {
      state.createdTripId = 0;
      clearGenerationReceipt(account, state.generationReceipt);
      $("savedTripNotice").hidden = true;
    }
    try {
      renderTrips();
      if (Number(state.trip?.id) === id) showTrips();
      toast(result.cleanup_pending === true
        ? "Поездка удалена из кабинета. Очистка её файлов ещё не завершена; сервис повторит её при следующем запуске."
        : "Поездка удалена. Потраченные попытки не возвращаются.");
    } catch (_) {
      toast("Поездка удалена. Обновите страницу, чтобы обновить список.");
    }
  } finally {
    if (state.deletingTrips.get(id) === operation) state.deletingTrips.delete(id);
    if (control) { control.disabled = false; control.textContent = "Удалить поездку"; }
  }
}

function renderTrips() {
  renderAccessStatus();
  state.coverListRevision += 1;
  state.coverCardMounts = [];
  const grid = $("tripsGrid");
  grid.replaceChildren();
  const query = state.search.trim().toLocaleLowerCase("ru-RU");
  const filtered = state.trips.filter((trip) => {
    const matchesFilter = state.filter === "all"
      || (state.filter === "upcoming" && trip.status !== "past")
      || trip.status === state.filter;
    if (!matchesFilter) return false;
    if (!query) return true;
    return [trip.destination, trip.title]
      .some((value) => String(value || "").toLocaleLowerCase("ru-RU").includes(query));
  });
  $("tripResultsStatus").textContent = tripCountLabel(filtered.length);

  const sorted = [...filtered].sort((left, right) => (
    String(right.start_date || "").localeCompare(String(left.start_date || ""))
    || Number(right.id || 0) - Number(left.id || 0)
  ));
  const groups = groupTripsByYear(sorted);
  const years = groups.map(([year]) => year);
  const currentYear = String(new Date().getFullYear());
  const fallbackYear = years.includes(currentYear) ? currentYear : years[0];
  const expandMatchingGroups = Boolean(query);
  groups.forEach(([year, trips]) => {
    const totalPages = Math.max(1, Math.ceil(trips.length / state.pageSize));
    const requestedPage = Number(state.yearPages.get(year) || 1);
    const page = Math.min(Math.max(1, requestedPage), totalPages);
    state.yearPages.set(year, page);
    const start = (page - 1) * state.pageSize;
    grid.append(createTripYearSection(
      year,
      trips.slice(start, start + state.pageSize),
      trips.length,
      expandMatchingGroups
        || state.openYears.has(year)
        || (!state.closedYears.has(year) && year === fallbackYear),
      page,
      totalPages,
      !expandMatchingGroups,
    ));
  });

  const empty = $("emptyTrips");
  empty.hidden = filtered.length !== 0;
  if (!state.trips.length) {
    $("emptyTripsTitle").textContent = "Здесь появятся ваши маршруты";
    $("emptyTripsText").textContent = "Создайте первую поездку — она сразу появится в кабинете.";
  } else if (!filtered.length) {
    $("emptyTripsTitle").textContent = "Ничего не найдено";
    $("emptyTripsText").textContent = "Измените поиск или выберите другой фильтр.";
  }
  void hydrateTripCardCovers(state.coverCardMounts, state.coverListRevision);
}

function createTripYearSection(year, trips, totalTrips, isOpen, page, totalPages, rememberToggle) {
  const section = node("details", "trip-year-group");
  section.dataset.year = year;
  section.open = isOpen;
  const summary = node("summary", "trip-year-summary");
  summary.append(
    node("h2", "", year === "Без даты" ? year : `${year} год`),
    node("span", "trip-year-count", tripCountLabel(totalTrips)),
  );
  const cards = node("div", "trips-grid trip-year-grid");
  trips.forEach((trip) => cards.append(createTripCard(trip)));
  section.append(summary, cards);
  if (totalPages > 1) section.append(createTripYearPagination(year, page, totalPages));
  section.addEventListener("toggle", (event) => {
    if (!event.isTrusted || !rememberToggle) return;
    if (section.open) {
      state.openYears.add(year);
      state.closedYears.delete(year);
    } else {
      state.openYears.delete(year);
      state.closedYears.add(year);
    }
  });
  return section;
}

function createTripYearPagination(year, page, totalPages) {
  const pagination = node("nav", "pagination trip-year-pagination");
  pagination.setAttribute(
    "aria-label",
    year === "Без даты" ? "Страницы поездок без даты" : `Страницы поездок за ${year} год`,
  );
  const showPage = (nextPage) => {
    state.yearPages.set(year, nextPage);
    renderTrips();
    const section = [...document.querySelectorAll(".trip-year-group")]
      .find((item) => item.dataset.year === year);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    section?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  };
  const previous = button("← Назад", "button button-ghost pagination-button", () => {
    if (page <= 1) return;
    showPage(page - 1);
  });
  previous.disabled = page === 1;

  const label = node("span", "pagination-label", `Страница ${page} из ${totalPages}`);

  const next = button("Вперёд →", "button button-ghost pagination-button", () => {
    if (page >= totalPages) return;
    showPage(page + 1);
  });
  next.disabled = page === totalPages;
  pagination.append(previous, label, next);
  return pagination;
}

function createTripCard(trip) {
  const card = node("article", "trip-card");
  const content = node("div", "trip-card-content");
  const top = node("div", "trip-card-top");
  const copy = node("div");
  const destinationType = trip.destination_kind === "region" ? "Территория" : "Город";
  copy.append(node("p", "eyebrow", `Поездка №${trip.id} · ${destinationType}`));
  copy.append(node("h3", "", trip.destination));
  copy.append(node("p", "", `${formatDate(trip.start_date)} — ${formatDate(trip.end_date)}`));
  const badge = node("span", `status-pill ${trip.status}`, statusLabel(trip.status));
  top.append(copy, badge);

  const title = node("p", "trip-card-title", trip.title);
  const stats = node("div", "trip-stats");
  const values = [
    [`${trip.days_count}`, trip.days_count === 1 ? "день" : "дней"],
    [formatMoney(trip.planned_total ?? trip.program_total, trip.currency), "план поездки"],
    [`${trip.checklist.done}/${trip.checklist.total}`, "собрано"],
  ];
  values.forEach(([value, label]) => {
    const stat = node("div", "trip-stat");
    stat.append(node("strong", "", value), node("span", "", label));
    stats.append(stat);
  });

  const actions = node("div", "trip-card-actions");
  actions.append(
    button("Открыть маршрут", "button button-primary", () => openTrip(trip.id)),
    button("PDF", "button button-ghost", () => { window.location.href = `/api/trips/${trip.id}/pdf`; }),
    createDeleteTripButton(trip),
  );
  content.append(top, title, stats, actions);
  const cover = createTripCover(trip, false, cachedCover(trip.id)?.value);
  state.coverCardMounts.push({ trip, mount: cover });
  card.append(cover, content);
  return card;
}

function beginViewChange() {
  stopCoverUpdates();
  state.cityClarification = null;
  state.viewRevision += 1;
  state.stopEditWaiting?.();
  $("editDialog").close();
  return state.viewRevision;
}

async function openTrip(tripId, updateHash = true) {
  if (state.deletedTrips.has(Number(tripId))) { toast("Эта поездка удалена."); return false; }
  const viewRevision = beginViewChange();
  setLoading(true, "Открываю маршрут…");
  try {
    const trip = await api(`/api/trips/${tripId}`);
    if (viewRevision !== state.viewRevision) return false;
    if (state.deletedTrips.has(Number(trip.id))) return false;
    displayTrip(trip, updateHash);
    return true;
  } catch (error) {
    if (viewRevision !== state.viewRevision) return false;
    toast(error.message);
    if (error.status === 404) showTrips();
    return false;
  } finally {
    if (viewRevision === state.viewRevision) setLoading(false);
  }
}

function displayTrip(trip, updateHash = true) {
  if (state.deletedTrips.has(Number(trip.id))) return false;
  state.trip = trip;
  state.selectedDay = trip.itinerary.days[0]?.day || 1;
  $("tripsView").hidden = true;
  $("tripView").hidden = false;
  renderTrip();
  if (updateHash) history.pushState({}, "", `#trip/${trip.id}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderTrip() {
  renderTripHeader();
  renderTripNotes();
  renderDayTabs();
  renderSelectedDay();
  renderBudget();
  renderChecklist();
  $("remindersText").textContent = state.trip.reminders_text || "Напоминаний пока нет.";
  startTripCoverView();
}

function renderTripNotes() {
  const panel = $("tripNotes");
  panel.replaceChildren();
  const itinerary = state.trip.itinerary;
  const summary = typeof itinerary.summary === "string" ? itinerary.summary.trim() : "";
  const tips = Array.isArray(itinerary.tips)
    ? itinerary.tips.filter((tip) => typeof tip === "string" && tip.trim()) : [];
  const coverage = programCoverageText(itinerary);
  const unpricedNote = cityUnpricedVisitNote(itinerary);
  const quality = state.trip.generation_quality;
  const hasQuality = quality && ["complete", "partial", "insufficient_data"].includes(quality.quality_status);
  panel.hidden = !summary && !tips.length && !coverage && !unpricedNote && !hasQuality;
  if (hasQuality) {
    const disclosure = node("section", "program-coverage");
    const labels = { complete: "Программа составлена на все дни", partial: "Программа составлена частично",
      insufficient_data: "Недостаточно подтверждённых данных" };
    disclosure.append(node("h2", "", labels[quality.quality_status]));
    if ([quality.program_days, quality.requested_days].every(Number.isInteger)) {
      disclosure.append(node("p", "", `Дней с программой: ${quality.program_days} из ${quality.requested_days}.`));
    }
    for (const value of [...(Array.isArray(quality.reason_texts) ? quality.reason_texts : []),
      ...(Array.isArray(quality.interest_texts) ? quality.interest_texts : [])].slice(0, 20)) {
      if (typeof value === "string") disclosure.append(node("p", "", value));
    }
    if (typeof quality.memory_note === "string" && quality.memory_note) {
      disclosure.append(node("p", "budget-note", quality.memory_note));
    }
    panel.append(disclosure);
  }
  if (coverage) {
    const disclosure = node("div", "program-coverage");
    disclosure.append(node("h2", "", "Программа и свободные дни"), node("p", "", coverage));
    panel.append(disclosure);
  }
  if (unpricedNote) panel.append(node("p", "budget-note", unpricedNote));
  if (summary || tips.length) {
    const explanation = node("details", "trip-explainer");
    explanation.append(node("summary", "", "Описание, советы и условия поездки"));
    const explanationContent = node("div", "trip-explainer-content");
    if (summary) explanationContent.append(node("p", "", summary));
    if (tips.length) {
      const list = node("ul");
      tips.forEach((tip) => list.append(node("li", "", tip)));
      explanationContent.append(list);
    }
    explanation.append(explanationContent);
    panel.append(explanation);
  }
}

function isUnpricedCityVisit(itinerary, item) {
  return itinerary.city_schedule === true && itinerary.source_schedule !== true
    && item.kind === "attraction" && item.cost === 0;
}

function cityUnpricedVisitNote(itinerary) {
  const count = (itinerary.days || []).reduce((total, day) =>
    total + day.items.filter((item) => isUnpricedCityVisit(itinerary, item)).length, 0);
  if (!count) return "";
  if (typeof itinerary.unpriced_visit_note === "string" && itinerary.unpriced_visit_note.trim()) {
    return itinerary.unpriced_visit_note;
  }
  return `Посещений с неуточнённой ценой: ${count}. Итог предварительный и не включает `
    + "неизвестную стоимость входа; нулевой резерв не означает бесплатный вход.";
}

function programCoverageText(itinerary) {
  if (typeof itinerary.program_report_text === "string") return itinerary.program_report_text;
  const report = itinerary.program_report;
  const sourceReport = itinerary.source_schedule === true
    && (itinerary.city_schedule === undefined || itinerary.city_schedule === false)
    && (report?.coverage_basis === undefined || report.coverage_basis === "source");
  const mapReport = itinerary.city_schedule === true && itinerary.source_schedule === false
    && report?.coverage_basis === "map";
  if (!report || report.schema_version !== 1 || report.reason !== "limited_verified_program"
      || (!sourceReport && !mapReport)) return "";
  const values = [report.requested_days, report.program_days, report.free_days,
    report.eligible_unique_visits, report.selected_unique_visits];
  if (!values.every((value) => Number.isInteger(value) && value >= 0)
      || report.program_days + report.free_days !== report.requested_days
      || report.selected_unique_visits > report.eligible_unique_visits) return "";
  if (mapReport) {
    return `Подтверждено мест на карте: ${report.eligible_unique_visits}. `
      + `Включено: ${report.selected_unique_visits}. Дней с программой: ${report.program_days}. `
      + `Дней свободного режима: ${report.free_days}. Даты сохранены. `
      + "На остальные даты не удалось составить программу в рамках условий из найденных кандидатов; "
      + "это не утверждение, что других мест нет. Карта подтверждает названия и местоположение, "
      + "но не условия посещения, часы работы, цены, длительность или наличие помещения. "
      + "Резерв на обед учтён во все дни; личные расходы не запланированы.";
  }
  return `Проверено подходящих мест: ${report.eligible_unique_visits}. Включено: ${report.selected_unique_visits}. `
    + `Дней с программой: ${report.program_days}. Дней свободного режима: ${report.free_days}. Даты сохранены. `
    + "На оставшиеся даты не удалось подобрать уникальную проверенную программу в рамках условий; "
    + "это не утверждение, что других мест нет. Резерв на обед учтён во все дни; личные расходы не запланированы.";
}

const FREE_DAY_NOTE = "Свободный день: посещения не запланированы. "
  + "Резерв на обед учтён в бюджете; дополнительные занятия и остальные расходы не включены.";

function tripEditUnavailableReason() {
  if (state.deletingTrips.has(Number(state.trip?.id))) return "Поездка удаляется. Дождитесь завершения операции.";
  if (state.trip?.edit_available !== false) return "";
  return state.trip.edit_unavailable_reason
    || "Редактирование этого городского маршрута пока недоступно. Маршрут сохранён без изменений.";
}

function renderTripHeader() {
  const hero = $("tripHero");
  hero.replaceChildren(createTripCover(state.trip, true, cachedCover(state.trip.id)?.value));
  // The outer node is a stable mount point, the inner one owns the image/fallback.
  hero.className = "trip-cover-mount";
  const header = $("tripHeader");
  header.replaceChildren();
  const copy = node("div");
  copy.append(node("p", "eyebrow", `Поездка №${state.trip.id}`));
  copy.append(node("h1", "", state.trip.destination));
  const request = state.trip.request;
  const meta = [
    request.destination_kind === "region" ? "территория" : "город",
    `${formatDate(state.trip.start_date)} — ${formatDate(state.trip.end_date)}`,
    `${request.adults} взрослых`,
  ];
  if (request.children_ages?.length) {
    meta.push(`дети: ${request.children_ages.join(", ")} лет`);
  }
  meta.push(
    `режим дня ${request.start_time || "09:00"}–${request.end_time || "20:00"}`,
    `темп ${request.pace}`,
  );
  copy.append(node("p", "trip-header-meta", meta.join(" · ")));
  const scopeNote = state.trip.destination_scope_note
    || (request.destination_kind === "region" ? DESTINATION_KIND_COPY.region.hint : "");
  if (scopeNote) copy.append(node("p", "trip-header-meta", scopeNote));

  const actions = node("div", "trip-header-actions");
  const pdf = node("a", "button button-primary", "Скачать PDF");
  pdf.href = `/api/trips/${state.trip.id}/pdf`;
  const edit = button("Изменить день", "button button-ghost", openEditDialog);
  const editReason = tripEditUnavailableReason();
  for (const control of [edit, $("editDayButton")]) {
    control.disabled = Boolean(editReason);
    control.title = editReason;
  }
  if (editReason) copy.append(node("p", "trip-header-meta", editReason));
  else if (state.trip.edit_scope_note) copy.append(node("p", "trip-header-meta", state.trip.edit_scope_note));
  actions.append(pdf, edit, createDeleteTripButton(state.trip));
  header.append(copy, actions);
}

function renderDayTabs() {
  const tabs = $("dayTabs");
  tabs.replaceChildren();
  state.trip.itinerary.days.forEach((day) => {
    const tab = node("button", `day-tab${day.day === state.selectedDay ? " is-active" : ""}`);
    tab.type = "button";
    tab.append(node("strong", "", `День ${day.day}`), node("span", "", formatDate(day.date, false)));
    if (day.day_mode === "free") tab.append(node("span", "day-mode-label", "Свободный режим"));
    tab.addEventListener("click", () => {
      state.selectedDay = day.day;
      renderDayTabs();
      renderSelectedDay();
      renderBudget();
    });
    tabs.append(tab);
  });
}

function selectedDay() {
  return state.trip.itinerary.days.find((day) => day.day === state.selectedDay);
}

function renderSelectedDay() {
  const day = selectedDay();
  if (!day) return;
  $("dayTitle").textContent = day.theme;
  const geography = $("dayGeography");
  geography.replaceChildren();
  if (day.base_area) {
    geography.append(node("p", "", `База дня: ${day.base_area}`));
  }
  if (day.transfer_note) {
    geography.append(node("p", "", `Переезд: ${day.transfer_note}`));
  }
  geography.hidden = !day.base_area && !day.transfer_note;
  const timeline = $("timeline");
  timeline.replaceChildren();

  if (day.day_mode === "free") {
    timeline.append(node("p", "free-day-note", FREE_DAY_NOTE));
  }

  day.items.forEach((item) => {
    const row = node("article", "timeline-item");
    const time = node("div", "timeline-time", item.time);
    const copy = node("div", "timeline-copy");
    copy.append(node("h3", "", item.title));
    if (item.description) copy.append(node("p", "", item.description));
    if (state.trip.request.children_ages?.length && item.kids_note) {
      copy.append(node("p", "kids-note", `Для детей: ${item.kids_note}`));
    }
    if (item.kind === "attraction") {
      copy.append(node("p", `place-state${item.verified ? "" : " unverified"}`, item.verified ? "✓ Место проверено на карте" : "Нужно проверить название места"));
      const source = mapSourceLink(item);
      if (source) copy.append(source);
    }
    const costText = isUnpricedCityVisit(state.trip.itinerary, item) ? "Цена не уточнена"
      : item.cost ? formatMoney(item.cost, state.trip.request.currency) : "";
    const cost = node("div", "timeline-cost", costText);
    row.append(time, copy, cost);
    timeline.append(row);
  });

  const rain = $("rainAlternative");
  rain.hidden = !day.rainy_alternative;
  rain.textContent = day.rainy_alternative ? `☔ Если дождь: ${day.rainy_alternative}` : "";
  renderMap(day.day);
}

function renderMap(dayNumber) {
  const mapElement = $("map");
  const empty = $("mapEmpty");
  const day = state.trip.itinerary.days.find((item) => item.day === dayNumber);
  const freeDay = day?.day_mode === "free";
  const points = freeDay ? [] : (state.trip.map_points || []).filter((point) => point.day === dayNumber && point.verified);
  empty.textContent = freeDay
    ? "Свободный день: посещения не запланированы, точки на карте не требуются."
    : "Для этого дня пока нет подтверждённых координат.";

  if (state.map) {
    state.map.remove();
    state.map = null;
  }
  mapElement.hidden = points.length === 0;
  empty.hidden = points.length !== 0;
  if (!points.length) return;
  if (!window.L) {
    mapElement.hidden = true;
    empty.hidden = false;
    empty.textContent = "Карта временно недоступна. Координаты мест сохранены.";
    return;
  }

  state.map = window.L.map(mapElement, { scrollWheelZoom: false });
  window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(state.map);

  const bounds = [];
  points.forEach((point, index) => {
    const coordinates = [point.lat, point.lon];
    bounds.push(coordinates);
    const tooltip = node("span", "", `${index + 1}. ${point.title}`);
    window.L.circleMarker(coordinates, {
      radius: 10,
      color: "#ffffff",
      weight: 3,
      fillColor: "#ef7e65",
      fillOpacity: 1,
    }).addTo(state.map).bindTooltip(tooltip).openTooltip();
  });
  const map = state.map;
  const fitVisibleMap = () => {
    if (state.map !== map || !state.trip || !mapElement.isConnected) return;
    if (!mapElement.clientWidth || !mapElement.clientHeight) {
      requestAnimationFrame(fitVisibleMap);
      return;
    }
    map.invalidateSize({ pan: false });
    if (bounds.length === 1) map.setView(bounds[0], 14, { animate: false });
    else map.fitBounds(bounds, { padding: [30, 30], maxZoom: 15, animate: false });
  };
  requestAnimationFrame(fitVisibleMap);
}

function renderBudget() {
  const panel = $("budgetPanel");
  panel.replaceChildren();
  const budget = state.trip.budget;
  const dayBudget = (budget.days || []).find(
    (item) => Number(item.day) === Number(state.selectedDay),
  ) || {
    day: state.selectedDay,
    day_total: budget.program_total,
    planned_through_day: budget.planned_total,
    remaining_after_day: budget.remaining,
  };
  const remaining = Number(budget.remaining);
  const overBudget = remaining < 0;
  const heading = node("div", "panel-heading compact");
  heading.append(node("h2", "", "Смета всей поездки"));
  panel.append(
    heading,
    node("p", "budget-note", overBudget ? "Превышение бюджета по смете (предварительно)" : "Не распределено по смете (предварительно)"),
    node(
      "div",
      `budget-total${overBudget ? " over" : ""}`,
      overBudget
        ? `−${formatMoney(Math.abs(remaining), budget.currency)}`
        : formatMoney(remaining, budget.currency),
    ),
  );
  const track = node("div", "budget-line");
  const fill = node("span", overBudget ? "over" : "");
  const percent = budget.limit > 0
    ? Math.min(100, (Number(budget.planned_total) / budget.limit) * 100)
    : 100;
  fill.style.width = `${percent}%`;
  track.append(fill);
  const meta = node("div", "budget-meta");
  meta.append(
    node("span", "", `Программа дня ${dayBudget.day}: ${formatMoney(dayBudget.day_total, budget.currency)}`),
    node("span", "", `Нарастающий итог к концу дня ${dayBudget.day}, включая дорогу и жильё: ${formatMoney(dayBudget.planned_through_day, budget.currency)}`),
    node("span", "", `Бюджет минус этот итог: ${formatMoney(dayBudget.remaining_after_day, budget.currency)}. Последующие дни здесь ещё не вычтены.`),
  );
  const breakdown = node("dl", "budget-breakdown");
  const fixedValue = (value) => Number(value) > 0
    ? formatMoney(value, budget.currency)
    : "Не указано / не учтено";
  [
    ["Дорога туда и обратно", fixedValue(budget.travel_cost)],
    ["Жильё за все ночи", fixedValue(budget.lodging_cost)],
    ["Вся программа", formatMoney(budget.program_total, budget.currency)],
    ["Итого план поездки", formatMoney(budget.planned_total, budget.currency)],
  ].forEach(([label, value], index) => {
    const row = node("div", index === 3 ? "budget-breakdown-total" : "");
    row.append(node("dt", "", label), node("dd", "", value));
    breakdown.append(row);
  });
  const unpricedNote = cityUnpricedVisitNote(state.trip.itinerary);
  const coverage = budget.coverage;
  const disclosedAdmission = coverage && Array.isArray(coverage.missing)
    && coverage.missing.some((entry) => entry && entry.message === unpricedNote);
  const scopeNote = disclosedAdmission ? (budget.scope_note || "").replace(unpricedNote, "").trim() : (budget.scope_note || "");
  const priceDisclosure = unpricedNote && !disclosedAdmission && !scopeNote.includes(unpricedNote) ? ` ${unpricedNote}` : "";
  if (coverage && Array.isArray(coverage.missing) && coverage.missing.length) {
    const warning = node("div", "budget-coverage");
    warning.append(node("strong", "", "Смета неполная: что ещё нужно учесть"));
    const list = node("ul");
    coverage.missing.forEach((entry) => {
      if (entry && typeof entry.message === "string") list.append(node("li", "", entry.message));
    });
    warning.append(list);
    panel.append(warning);
  }
  panel.append(
    track,
    meta,
    breakdown,
    node("p", "budget-note", `Общий бюджет: ${formatMoney(budget.limit, budget.currency)}. ${scopeNote}${priceDisclosure}`),
  );
  if (coverage && Array.isArray(coverage.notes)) {
    coverage.notes.forEach((note) => {
      if (typeof note === "string") panel.append(node("p", "budget-note", note));
    });
  }
}

function renderChecklist() {
  const items = state.trip.checklist || [];
  const done = items.filter((item) => Boolean(item.done)).length;
  $("checklistCount").textContent = `${done}/${items.length}`;
  $("checklistProgress").style.width = `${items.length ? (done / items.length) * 100 : 0}%`;

  const container = $("checklistGroups");
  container.replaceChildren();
  const categories = [...new Set(items.map((item) => item.category))];
  categories.forEach((category, categoryIndex) => {
    const details = node("details", "checklist-group");
    if (categoryIndex === 0) details.open = true;
    const categoryItems = items.filter((item) => item.category === category);
    const categoryDone = categoryItems.filter((item) => Boolean(item.done)).length;
    details.append(node("summary", "", `${category} · ${categoryDone}/${categoryItems.length}`));
    const list = node("div", "checklist-items");
    categoryItems.forEach((item) => {
      const label = node("label", `check-item${item.done ? " is-done" : ""}`);
      const checkbox = node("input");
      checkbox.type = "checkbox";
      checkbox.checked = Boolean(item.done);
      checkbox.addEventListener("change", () => updateChecklistItem(item, checkbox.checked, checkbox));
      label.append(checkbox, node("span", "", item.text));
      list.append(label);
    });
    details.append(list);
    container.append(details);
  });
}

async function updateChecklistItem(item, done, checkbox) {
  const trip = state.trip;
  if (!trip || !trip.checklist?.includes(item)) return;
  const account = state.me;
  const viewRevision = state.viewRevision;
  const isCurrentTrip = () => state.me === account && state.trip === trip && viewRevision === state.viewRevision;
  checkbox.disabled = true;
  try {
    await api(`/api/checklist-items/${item.id}`, {
      method: "PATCH",
      body: JSON.stringify({ done }),
    });
    item.done = done ? 1 : 0;
    if (isCurrentTrip()) renderChecklist();
    if (account && state.me === account) await refreshTripSummary(account);
  } catch (error) {
    checkbox.checked = !done;
    if (isCurrentTrip()) toast(error.message);
  } finally {
    checkbox.disabled = false;
  }
}

function localToday() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function syncDestinationKind() {
  const kind = $("tripDestinationKind").value === "region" ? "region" : "city";
  const copy = DESTINATION_KIND_COPY[kind];
  $("tripDestinationLabel").textContent = copy.label;
  $("tripDestination").placeholder = copy.placeholder;
  $("tripDestinationHint").textContent = copy.hint;
  syncCityMvpDisclosure();
  $("tripGoal").placeholder = copy.goalPlaceholder;
  const currentGoal = $("tripGoal").value.trim();
  const defaultGoals = Object.values(DESTINATION_KIND_COPY).map((item) => item.goal);
  if (!currentGoal || defaultGoals.includes(currentGoal)) {
    $("tripGoal").value = copy.goal;
  }
}

function syncCityMvpDisclosure() {
  const disclosure = $("cityMvpDisclosure");
  const capabilities = state.me?.capabilities;
  const visible = $("tripDestinationKind").value === "city"
    && capabilities?.city_candidates_first === true;
  disclosure.hidden = !visible;
  disclosure.textContent = visible
    ? (typeof capabilities.city_mvp_disclosure === "string" && capabilities.city_mvp_disclosure.trim()
      ? capabilities.city_mvp_disclosure
      : "Предварительная городская программа: часть дат может остаться свободной. Карта подтверждает название и местоположение, но не условия посещения, цены или наличие помещения. Проезд к местам и транспортные расходы отдельно не рассчитаны. Редактирование новых городских маршрутов этого режима пока недоступно.")
    : "";
}

function openNewTripDialog() {
  const accessProblem = routeAccessProblem();
  if (accessProblem) { toast(accessProblem); return; }
  if (state.creatingTrip) return;
  const pendingReceipt = readGenerationReceipt();
  if (pendingReceipt && (state.createdTripId === null || !state.generationReceipt
      || pendingReceipt.key !== state.generationReceipt.key)) {
    void resumeGenerationReceipt();
    return;
  }
  state.createdTripId = null;
  state.cityClarification = null;
  state.newTripFormRevision += 1;
  clearGenerationReceipt(state.me, state.generationReceipt);
  $("savedTripNotice").hidden = true;
  const form = $("newTripForm");
  form.reset();
  $("newTripMessage").textContent = "";

  const today = localToday();
  $("tripStartDate").min = today;
  $("tripEndDate").min = today;
  $("tripAdults").value = String(state.me?.adults ?? 2);
  $("tripChildrenAges").value = Array.isArray(state.me?.children_ages)
    ? state.me.children_ages.join(", ")
    : "";
  $("tripNoChildren").checked = !Array.isArray(state.me?.children_ages)
    || state.me.children_ages.length === 0;
  $("tripCurrency").value = ["EUR", "USD", "RUB"].includes(state.me?.currency)
    ? state.me.currency
    : "EUR";
  $("tripPace").value = ["спокойный", "средний", "насыщенный"].includes(state.me?.pace)
    ? state.me.pace
    : "средний";
  $("tripStartTime").value = "09:00";
  $("tripEndTime").value = "20:00";
  $("tripDestinationKind").value = "city";
  $("tripGoal").value = "Познакомиться с городом";
  $("tripLocationPreference").value = "anywhere";
  // Wishes belong to this trip; a previous destination's restrictions are not reusable defaults.
  $("tripWishes").value = "";
  syncChildrenMode();
  syncDestinationKind();
  syncLocationMode();
  syncBudgetPreview();

  $("infoDialog").showModal();
  $("tripDestination").focus();
}

function closeNewTripDialog() {
  if (!state.creatingTrip) $("infoDialog").close();
}

function parseChildrenAges(value) {
  const text = value.trim();
  if (!text) return [];
  if (!/^\d{1,2}(?:\s*,\s*\d{1,2})*$/.test(text)) {
    throw new Error("Укажите возраст детей целыми числами через запятую.");
  }
  const ages = text.split(",").map((part) => Number(part.trim()));
  if (ages.length > 6) throw new Error("Можно указать не больше 6 детей.");
  if (ages.some((age) => age < 0 || age > 17)) {
    throw new Error("Возраст каждого ребёнка должен быть от 0 до 17 лет.");
  }
  return ages;
}

function syncChildrenMode() {
  const withoutChildren = $("tripNoChildren").checked;
  $("tripChildrenAges").disabled = withoutChildren || state.creatingTrip;
  $("tripChildrenAges").required = !withoutChildren;
  if (withoutChildren) $("tripChildrenAges").value = "";
}

function syncLocationMode() {
  const nearby = $("tripLocationPreference").value === "nearby";
  $("tripStayAreaField").hidden = !nearby;
  $("tripStayArea").required = nearby;
  $("tripStayArea").disabled = !nearby || state.creatingTrip;
  if (!nearby) $("tripStayArea").value = "";
}

function timeToMinutes(value) {
  if (!/^\d{2}:\d{2}$/.test(value)) {
    throw new Error("Укажите время начала и окончания дня.");
  }
  const [hour, minute] = value.split(":").map(Number);
  if (hour > 23 || minute > 59) {
    throw new Error("Укажите корректное время.");
  }
  return hour * 60 + minute;
}

function syncBudgetPreview() {
  const budget = Number($("tripBudget").value);
  const travelCost = Number($("tripTravelCost").value);
  const lodgingCost = Number($("tripLodgingCost").value);
  const preview = $("tripProgramBudget");
  const available = Math.round((budget - travelCost - lodgingCost) * 100) / 100;
  const valid = [budget, travelCost, lodgingCost].every(Number.isFinite)
    && budget > 0 && travelCost >= 0 && lodgingCost >= 0;
  preview.classList.toggle("over", valid && available <= 0);
  preview.textContent = valid
    ? `На программу после дороги и жилья: ${formatMoney(available, $("tripCurrency").value)}`
    : "Укажите общий бюджет, чтобы увидеть сумму на программу.";
}

function newTripPayload() {
  const destination = $("tripDestination").value.trim();
  const destinationKind = $("tripDestinationKind").value;
  const startDate = $("tripStartDate").value;
  const endDate = $("tripEndDate").value;
  const adults = Number($("tripAdults").value);
  const budget = Number($("tripBudget").value);
  const travelCost = Number($("tripTravelCost").value);
  const lodgingCost = Number($("tripLodgingCost").value);
  const startTime = $("tripStartTime").value;
  const endTime = $("tripEndTime").value;
  const startMinutes = timeToMinutes(startTime);
  const endMinutes = timeToMinutes(endTime);
  const tripGoal = $("tripGoal").value.trim();
  const interestCategories = [...document.querySelectorAll(
    'input[name="interest_categories"]:checked',
  )].map((item) => item.value);
  const locationPreference = $("tripLocationPreference").value;
  const stayArea = $("tripStayArea").value.trim();
  const wishes = $("tripWishes").value.trim();

  if (!["city", "region"].includes(destinationKind)) {
    throw new Error("Выберите тип направления.");
  }
  if (destination.length < 2) {
    throw new Error(destinationKind === "region" ? "Укажите территорию поездки." : "Укажите город поездки.");
  }
  if (!startDate || !endDate) throw new Error("Укажите даты начала и окончания.");
  if (endDate < startDate) {
    throw new Error("Дата окончания не может быть раньше даты начала.");
  }
  if (!Number.isInteger(adults) || adults < 1 || adults > 10) {
    throw new Error("Число взрослых должно быть от 1 до 10.");
  }
  if (!Number.isFinite(budget) || budget <= 0) {
    throw new Error("Укажите бюджет больше нуля.");
  }
  if ([travelCost, lodgingCost].some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("Стоимость дороги и жилья должна быть неотрицательной.");
  }
  if (Math.round((budget - travelCost - lodgingCost) * 100) / 100 <= 0) {
    throw new Error("После дороги и жилья должен остаться бюджет на программу.");
  }
  if (
    startMinutes < 6 * 60
    || startMinutes > 23 * 60 + 30
    || endMinutes < 6 * 60
    || endMinutes > 23 * 60 + 30
  ) {
    throw new Error("Время маршрута должно быть между 06:00 и 23:30.");
  }
  if (startMinutes >= endMinutes) {
    throw new Error("Окончание дня должно быть позже начала.");
  }
  if (endMinutes - startMinutes < 4 * 60) {
    throw new Error("Для маршрута нужно выделить не меньше 4 часов в день.");
  }
  if (tripGoal.length < 2) {
    throw new Error("Укажите цель поездки.");
  }
  if (!interestCategories.length) {
    throw new Error("Выберите хотя бы один вариант отдыха.");
  }
  if (locationPreference === "nearby" && stayArea.length < 2) {
    throw new Error("Укажите район проживания или ближайшую станцию.");
  }
  if (wishes.length > 500) {
    throw new Error("Пожелания должны быть не длиннее 500 символов.");
  }

  return {
    destination,
    destination_kind: destinationKind,
    start_date: startDate,
    end_date: endDate,
    adults,
    children_ages: $("tripNoChildren").checked
      ? []
      : parseChildrenAges($("tripChildrenAges").value),
    budget,
    travel_cost: travelCost,
    lodging_cost: lodgingCost,
    currency: $("tripCurrency").value,
    pace: $("tripPace").value,
    start_time: startTime,
    end_time: endTime,
    trip_goal: tripGoal,
    interest_categories: interestCategories,
    location_preference: locationPreference,
    stay_area: locationPreference === "nearby" ? stayArea : "",
    wishes,
  };
}

function setNewTripBusy(active) {
  state.creatingTrip = active;
  $("newTripForm").querySelectorAll("input, select, textarea, button")
    .forEach((control) => { control.disabled = active; });
  if (!active) {
    syncChildrenMode();
    syncDestinationKind();
    syncLocationMode();
  }
}

function showSavedTripNotice(tripId, message, retryOpen = false) {
  const notice = $("savedTripNotice");
  notice.replaceChildren(node("p", "", message));
  notice.hidden = false;
  if (retryOpen && tripId > 0) {
    notice.append(button("Открыть сохранённый маршрут", "button button-ghost", async () => {
      if (await openTrip(tripId)) notice.hidden = true;
    }));
  }
}

async function completeTripCreation(result, account, viewRevision, receipt = null) {
  const isCurrent = () => state.me === account && (!receipt || state.generationReceipt === receipt);
  if (!isCurrent()) return;
  const savedTrip = result?.trip ?? result;
  const tripId = Number(savedTrip?.id);
  const hasId = Number.isInteger(tripId) && tripId > 0;
  if (hasId && state.deletedTrips.has(tripId)) {
    state.createdTripId = 0;
    showSavedTripNotice(0, "Эта поездка уже удалена. Повторная генерация не запускалась.");
    return true;
  }
  // A successful POST must never become a second POST when subsequent reads fail.
  state.createdTripId = hasId ? tripId : 0;
  $("infoDialog").close();
  const savedMessage = hasId ? `Маршрут №${tripId} готов и сохранён.` : "Маршрут сохранён.";
  let opened = false;
  if (state.viewRevision === viewRevision && hasId) {
    if (savedTrip?.request && savedTrip?.budget && Array.isArray(savedTrip?.itinerary?.days)) {
      beginViewChange();
      try { displayTrip(savedTrip); opened = true; } catch (_) { /* Retry the saved detail below. */ }
    }
    if (!opened) {
      const detailRevision = state.viewRevision + 1;
      opened = await openTrip(tripId);
      if (isCurrent() && state.deletedTrips.has(tripId)) return true;
      if (!opened && isCurrent() && state.viewRevision === detailRevision) showTrips();
    }
  }
  if (!isCurrent()) return;

  let listUpdated = false;
  try {
    const list = await api("/api/trips?limit=100");
    if (!isCurrent()) return;
    if (hasId && state.deletedTrips.has(tripId)) return true;
    state.trips = visibleTrips(list.trips);
    state.yearPages.clear();
    renderTrips();
    listUpdated = true;
  } catch (_) { /* Saving has already succeeded; list refresh is independent. */ }
  if (!isCurrent()) return;
  if (hasId && state.deletedTrips.has(tripId)) return true;
  if (opened && listUpdated) {
    $("savedTripNotice").hidden = true;
    toast(savedMessage);
  } else {
    const details = [];
    if (!opened) details.push("Карточка пока не открыта.");
    if (!listUpdated) details.push("Список поездок временно не обновился.");
    showSavedTripNotice(hasId ? tripId : 0, `${savedMessage} ${details.join(" ")} Создавать маршрут повторно не нужно.`, !opened);
  }
  return opened && listUpdated;
}

// Only opaque receipt identifiers are kept locally, separately for each account.
// In particular, never persist the personal questionnaire or repeat POST on reload.
function generationStorageKey(account = state.me) {
  const capability = account?.generation_jobs;
  return capability?.enabled === true && /^[a-f0-9]{64}$/.test(capability.owner || "")
    ? `family-trip-generation:${capability.owner}` : null;
}

function clearGenerationReceipt(account = state.me, expectedReceipt = null) {
  if (expectedReceipt && state.generationReceipt !== expectedReceipt) return;
  if (state.generationTimer !== null) clearTimeout(state.generationTimer);
  state.generationTimer = null;
  state.generationReceipt = null;
  const key = generationStorageKey(account);
  if (key) {
    try {
      const stored = JSON.parse(localStorage.getItem(key) || "null");
      if (!expectedReceipt || stored?.key === expectedReceipt.key) localStorage.removeItem(key);
    } catch (_) { /* No paid action here. */ }
  }
}

function storeGenerationReceipt(receipt, account = state.me) {
  const key = generationStorageKey(account);
  if (!key) throw new Error("Не удалось подготовить восстановление заявки.");
  localStorage.setItem(key, JSON.stringify({ key: receipt.key, job_id: receipt.job_id || null }));
  state.generationReceipt = receipt;
}

function readGenerationReceipt(account = state.me) {
  const key = generationStorageKey(account);
  if (!key) return null;
  let receipt;
  try { receipt = JSON.parse(localStorage.getItem(key) || "null"); } catch (_) { return null; }
  return receipt && /^[a-f0-9-]{36}$/.test(receipt.key || "")
    && (receipt.job_id === null || /^[a-f0-9-]{36}$/.test(receipt.job_id || "")) ? receipt : null;
}

function scheduleGenerationPoll(account, receipt) {
  if (state.me !== account || state.generationReceipt !== receipt || account?.access?.allowed === false) return;
  if (state.generationTimer !== null) clearTimeout(state.generationTimer);
  state.generationTimer = setTimeout(() => {
    state.generationTimer = null;
    void pollGenerationReceipt(account, receipt);
  }, 3000);
}

function showGenerationUncertain(account, receipt) {
  if (state.me !== account || state.generationReceipt !== receipt) return;
  $("infoDialog").close();
  const message = "Не удалось получить состояние заявки. Повторная генерация не запускалась. Проверяю сохранённую заявку; также можно открыть список поездок.";
  $("newTripMessage").textContent = message;
  showSavedTripNotice(0, message);
  $("savedTripNotice").append(button("Открыть список поездок", "button button-ghost", () => {
    $("infoDialog").close();
    showTrips();
  }));
  // An uncertain POST may never have reached the server. Do not lock the user
  // forever, but require an explicit acknowledgement before a NEW operation.
  $("savedTripNotice").append(button("Завершить проверку заявки", "button button-ghost", () => {
    if (state.me !== account || state.generationReceipt !== receipt) return;
    if (!window.confirm("Сначала проверьте список поездок: маршрут мог сохраниться. Завершить только проверку статуса, без новой генерации?")) return;
    clearGenerationReceipt(account, receipt);
    setNewTripBusy(false);
    state.createdTripId = 0;
    $("infoDialog").close();
    showSavedTripNotice(0, "Проверка остановлена. Повторная генерация не запускалась. Проверьте список поездок перед созданием новой заявки.");
  }));
}

async function applyGenerationStatus(result, account, receipt) {
  if (state.me !== account || state.generationReceipt !== receipt) return;
  if (!result || !/^[a-f0-9-]{36}$/.test(result.job_id || "")
      || !["queued", "running", "succeeded", "failed", "interrupted"].includes(result.status)) {
    throw new Error("Не удалось проверить состояние заявки.");
  }
  receipt.job_id = result.job_id;
  try { storeGenerationReceipt(receipt, account); } catch (_) { /* Original key remains sufficient for GET recovery. */ }
  const message = typeof result.message === "string" ? result.message : "Состояние заявки обновлено.";
  $("newTripMessage").textContent = message;
  if (["queued", "running"].includes(result.status)) {
    $("infoDialog").close();
    showSavedTripNotice(0, message);
    scheduleGenerationPoll(account, receipt);
    return;
  }
  if (state.generationTimer !== null) clearTimeout(state.generationTimer);
  state.generationTimer = null;
  setNewTripBusy(false);
  void refreshAccountAccess(account, true);
  if (result.status === "succeeded" && Number.isInteger(result.trip_id) && result.trip_id > 0) {
    const displayed = await completeTripCreation({ id: result.trip_id }, account, receipt.viewRevision, receipt);
    if (state.me === account && state.generationReceipt === receipt && displayed) {
      clearGenerationReceipt(account, receipt);
    }
    return;
  }
  clearGenerationReceipt(account, receipt);
  state.createdTripId = 0; // Re-submit requires the explicit opening of a NEW form.
  $("infoDialog").close();
  showSavedTripNotice(0, message);
  if (result.status === "failed") showCityClarification(result.clarification, account, receipt);
}

// These buttons only restore/open a form. A new paid request always requires
// the user's separate submit; the questionnaire never goes into localStorage.
function showCityClarification(clarification, account, draft) {
  if (state.me !== account || clarification?.kind !== "city" || clarification.attempt_used !== false) return;
  const options = Array.isArray(clarification.options) ? [...new Set(clarification.options.slice(0, 10)
    .filter((value) => typeof value === "string" && value.length <= 100
      && value.split(",").length === 3
      && value.split(",").every((part) => /[\p{L}\p{N}]/u.test(part))
      && /^[\p{L}\p{M}\p{N} ,\-'’().–—]+$/u.test(value)))].slice(0, 5) : [];
  const token = { account, formRevision: state.newTripFormRevision, viewRevision: state.viewRevision };
  state.cityClarification = token;
  const choose = (destination) => {
    if (state.me !== account || state.cityClarification !== token || state.creatingTrip
        || token.formRevision !== state.newTripFormRevision || token.viewRevision !== state.viewRevision) return;
    const problem = routeAccessProblem();
    if (problem) { toast(problem); return; }
    const hasDraft = draft?.draftAvailable === true && draft.formRevision === state.newTripFormRevision;
    state.cityClarification = null;
    if (!hasDraft) openNewTripDialog();
    else {
      state.createdTripId = null;
      $("infoDialog").showModal();
    }
    if (!$("infoDialog").open) return;
    $("tripDestinationKind").value = "city";
    syncDestinationKind();
    if (destination) $("tripDestination").value = destination;
    $("savedTripNotice").hidden = true;
    $("newTripMessage").textContent = hasDraft
      ? "Направление можно уточнить. Остальные поля анкеты сохранены. Проверьте их и нажмите кнопку создания, когда будете готовы."
      : "Заполните остальные поля анкеты и нажмите кнопку создания, когда будете готовы. Автоматическая генерация не запускалась.";
    $("tripDestination").focus();
  };
  for (const option of options) $("savedTripNotice").append(button(option, "button button-ghost", () => choose(option)));
  $("savedTripNotice").append(button("Уточнить направление вручную", "button button-ghost", () => choose(null)));
}

async function pollGenerationReceipt(account, receipt) {
  if (state.me !== account || state.generationReceipt !== receipt) return;
  try {
    const path = receipt.job_id ? `/api/generation-jobs/${receipt.job_id}`
      : `/api/generation-jobs?key=${encodeURIComponent(receipt.key)}`;
    await applyGenerationStatus(await api(path), account, receipt);
  } catch (error) {
    if (isAccessDenied(error)) {
      if (state.me !== account || state.generationReceipt !== receipt) return;
      markAccessDenied(account);
      setNewTripBusy(false);
      $("infoDialog").close();
      $("newTripMessage").textContent = ACCESS_DENIED_MESSAGE;
      showSavedTripNotice(0, `${ACCESS_DENIED_MESSAGE} Проверка заявки остановлена; повторная генерация не запускалась.`);
      return;
    }
    showGenerationUncertain(account, receipt);
    // Expired login needs a new login, not endless unauthorized requests.
    if (error.status !== 401 && error.status !== 403) scheduleGenerationPoll(account, receipt);
  }
}

async function resumeGenerationReceipt() {
  const account = state.me;
  const receipt = readGenerationReceipt(account);
  if (!receipt) return;
  state.generationReceipt = { key: receipt.key, job_id: receipt.job_id, viewRevision: state.viewRevision };
  setNewTripBusy(true);
  await pollGenerationReceipt(account, state.generationReceipt);
}

async function submitGenerationJob(payload, account) {
  if (readGenerationReceipt(account)) {
    await resumeGenerationReceipt();
    return;
  }
  let receipt;
  try {
    state.cityClarification = null;
    receipt = { key: crypto.randomUUID(), job_id: null, viewRevision: state.viewRevision,
      draftAvailable: true, formRevision: state.newTripFormRevision };
    // If storage is denied, fail BEFORE sending a paid request.
    storeGenerationReceipt(receipt, account);
  } catch (_) {
    $("newTripMessage").textContent = "Браузер не разрешил сохранить ключ заявки. Разрешите локальное хранилище; генерация не запускалась.";
    return;
  }
  setNewTripBusy(true);
  $("newTripMessage").textContent = "Отправляю заявку. После принятия можно закрыть страницу и вернуться позже.";
  try {
    const result = await api("/api/generation-jobs", {
      method: "POST", headers: { "Idempotency-Key": receipt.key }, body: JSON.stringify(payload),
    });
    await applyGenerationStatus(result, account, receipt);
  } catch (error) {
    if (state.me !== account || state.generationReceipt !== receipt) return;
    if ([401, 403, 404, 409, 422, 429].includes(error.status)) {
      clearGenerationReceipt(account, receipt);
      setNewTripBusy(false);
      $("newTripMessage").textContent = error.message;
    } else {
      // Never POST again, even if the initial response was lost.
      await pollGenerationReceipt(account, receipt);
    }
  } finally {
    void refreshAccountAccess(account, true);
  }
}

async function submitNewTrip(event) {
  event.preventDefault();
  const accessProblem = routeAccessProblem();
  if (accessProblem) { $("newTripMessage").textContent = accessProblem; return; }
  if (state.creatingTrip || state.createdTripId !== null) return;
  const form = $("newTripForm");
  if (!form.reportValidity()) return;

  let payload;
  try {
    payload = newTripPayload();
  } catch (error) {
    $("newTripMessage").textContent = error.message;
    return;
  }

  $("newTripMessage").textContent = "";
  const account = state.me;
  const viewRevision = state.viewRevision;
  const draft = { draftAvailable: true, formRevision: state.newTripFormRevision };
  if (generationStorageKey(account)) {
    await submitGenerationJob(payload, account);
    return;
  }
  setNewTripBusy(true);
  const stopWaiting = startAIWait(
    $("newTripMessage"),
    "Составляю маршрут и проверяю места…",
    "Маршрут всё ещё формируется и проверяется. Это может занять несколько минут. Оставьте вкладку открытой.",
  );
  try {
    const result = await api("/api/trips", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    stopWaiting();
    await completeTripCreation(result, account, viewRevision);
  } catch (error) {
    if (state.me !== account) return;
    if (error.savedTripId) {
      stopWaiting();
      await completeTripCreation({ id: error.savedTripId }, account, viewRevision);
    } else if (state.createdTripId !== null) {
      showSavedTripNotice(state.createdTripId, `Маршрут${state.createdTripId ? ` №${state.createdTripId}` : ""} сохранён, но не удалось показать его. Создавать маршрут повторно не нужно.`, true);
    } else {
      $("newTripMessage").textContent = error.status
        ? error.message
        : "Не удалось получить ответ. Проверьте список поездок перед повторным запуском: маршрут мог сохраниться на сервере.";
      if (error.status === 422 && error.clarification?.kind === "city") {
        $("infoDialog").close();
        state.createdTripId = 0;
        showSavedTripNotice(0, error.message);
        showCityClarification(error.clarification, account, draft);
      }
    }
  } finally {
    stopWaiting();
    setLoading(false);
    setNewTripBusy(false);
    void refreshAccountAccess(account, true);
  }
}

function showTrips(updateHash = true) {
  beginViewChange();
  state.trip = null;
  if (state.map) { state.map.remove(); state.map = null; }
  $("tripView").hidden = true;
  $("tripsView").hidden = false;
  if (!state.creatingTrip) setLoading(false);
  if (updateHash) history.pushState({}, "", "#trips");
  window.scrollTo({ top: 0, behavior: "smooth" });
  void hydrateTripCardCovers(state.coverCardMounts, state.coverListRevision);
  renderAccessStatus();
  void refreshAccountAccess();
}

function openEditDialog() {
  const accessProblem = routeAccessProblem();
  if (accessProblem) { toast(accessProblem); return; }
  if (!state.trip || state.editingTrip) return;
  const unavailable = tripEditUnavailableReason();
  if (unavailable) { toast(unavailable); return; }
  $("editDialogTitle").textContent = `Изменить день ${state.selectedDay}`;
  $("editInstruction").value = "";
  $("editScopeNote").textContent = state.trip.edit_scope_note || "Остальные дни сохранятся без изменений.";
  $("editInstruction").placeholder = state.trip.itinerary.city_schedule
    ? "Например: перенеси обед на 13:30, сохрани посещения и отдых"
    : "Например: измени программу этого дня, сохрани обед и отдых";
  $("editMessage").textContent = "";
  $("editDialog").showModal();
  $("editInstruction").focus();
}

function closeEditDialog() {
  if (!state.editingTrip) $("editDialog").close();
}

function setEditBusy(active) {
  state.editingTrip = active;
  $("editForm").querySelectorAll("input, select, textarea, button")
    .forEach((control) => { control.disabled = active; });
}

async function submitEdit(event) {
  event.preventDefault();
  const accessProblem = routeAccessProblem();
  if (accessProblem) { $("editMessage").textContent = accessProblem; return; }
  if (state.editingTrip || !state.trip) return;
  const unavailable = tripEditUnavailableReason();
  if (unavailable) { toast(unavailable); return; }
  const instruction = $("editInstruction").value.trim();
  if (instruction.length < 3) return;
  const tripId = state.trip.id;
  const account = state.me;
  const dayNumber = state.selectedDay;
  const viewRevision = state.viewRevision;
  const isCurrentTrip = () => state.me === account && state.trip?.id === tripId && viewRevision === state.viewRevision;
  setEditBusy(true);
  const stopWaiting = startAIWait(
    $("editMessage"),
    "Изменяю день и проверяю места…",
    "Проверка дня продолжается и может занять несколько минут. Старый вариант пока остаётся сохранённым.",
  );
  state.stopEditWaiting = stopWaiting;
  try {
    const trip = await api(`/api/trips/${tripId}/days/${dayNumber}/edit`, {
      method: "POST",
      body: JSON.stringify({ instruction }),
    });
    stopWaiting();
    if (isCurrentTrip()) {
      state.trip = trip;
      state.selectedDay = dayNumber;
      $("editDialog").close();
      renderTrip();
      toast(`День ${dayNumber} обновлён. Остальные дни сохранены.`);
    }
    if (account && state.me === account) await refreshTripSummary(account);
  } catch (error) {
    if (isCurrentTrip()) {
      $("editMessage").textContent = error.status
        ? error.message
        : "Не удалось получить ответ. Обновите маршрут перед повторным изменением: сервер мог сохранить новый вариант.";
    }
  } finally {
    stopWaiting();
    state.stopEditWaiting = null;
    if (isCurrentTrip()) setLoading(false);
    setEditBusy(false);
    void refreshAccountAccess(account, true);
  }
}

async function refreshTripSummary(account = state.me) {
  if (!account || state.me !== account) return;
  try {
    const payload = await api("/api/trips?limit=100");
    if (state.me !== account) return;
    state.trips = visibleTrips(payload.trips);
    renderTrips();
  } catch (_) {
    // Детальная страница уже обновлена; список обновится при следующем открытии.
  }
}

async function logout() {
  beginViewChange();
  clearCoverState();
  clearGenerationReceipt();
  state.deletingTrips.clear();
  state.deletedTrips.clear();
  setNewTripBusy(false);
  state.me = null;
  state.accessRevision += 1;
  state.accessRefresh = null;
  state.accessUnconfirmed = false;
  state.createdTripId = null;
  $("savedTripNotice").hidden = true;
  try { await api("/api/logout", { method: "POST" }); } catch (_) { /* очищаем экран в любом случае */ }
  state.trips = [];
  state.trip = null;
  state.search = "";
  state.yearPages.clear();
  state.openYears.clear();
  state.closedYears.clear();
  $("appShell").hidden = true;
  $("loginGate").hidden = false;
  $("accessGate").hidden = true;
  setLoading(false);
  history.replaceState({}, "", "#login");
}

function bindEvents() {
  $("brandHome").addEventListener("click", () => showTrips());
  $("backButton").addEventListener("click", () => showTrips());
  $("editDayButton").addEventListener("click", openEditDialog);
  $("logoutButton").addEventListener("click", logout);
  $("retryLogin").addEventListener("click", init);
  $("retryAccess").addEventListener("click", init);
  $("newTripButton").addEventListener("click", openNewTripDialog);
  $("closeInfoDialog").addEventListener("click", closeNewTripDialog);
  $("cancelNewTrip").addEventListener("click", closeNewTripDialog);
  $("newTripForm").addEventListener("submit", submitNewTrip);
  $("infoDialog").addEventListener("cancel", (event) => {
    if (state.creatingTrip) event.preventDefault();
  });
  $("tripStartDate").addEventListener("change", () => {
    $("tripEndDate").min = $("tripStartDate").value || localToday();
    if ($("tripEndDate").value && $("tripEndDate").value < $("tripStartDate").value) {
      $("tripEndDate").value = $("tripStartDate").value;
    }
  });
  $("tripNoChildren").addEventListener("change", syncChildrenMode);
  $("tripDestinationKind").addEventListener("change", syncDestinationKind);
  $("tripLocationPreference").addEventListener("change", syncLocationMode);
  ["tripBudget", "tripTravelCost", "tripLodgingCost", "tripCurrency"].forEach((id) => {
    $(id).addEventListener("input", syncBudgetPreview);
    $(id).addEventListener("change", syncBudgetPreview);
  });
  $("tripSearch").addEventListener("input", () => {
    state.search = $("tripSearch").value;
    state.yearPages.clear();
    renderTrips();
  });
  $("closeEditDialog").addEventListener("click", closeEditDialog);
  $("cancelEdit").addEventListener("click", closeEditDialog);
  $("editDialog").addEventListener("cancel", (event) => {
    if (state.editingTrip) event.preventDefault();
  });
  $("editForm").addEventListener("submit", submitEdit);
  document.querySelectorAll(".filter-button").forEach((item) => {
    item.addEventListener("click", () => {
      state.filter = item.dataset.filter;
      state.yearPages.clear();
      document.querySelectorAll(".filter-button").forEach((other) => {
        const isActive = other === item;
        other.classList.toggle("is-active", isActive);
        other.setAttribute("aria-pressed", String(isActive));
      });
      renderTrips();
    });
  });
  window.addEventListener("popstate", routeFromHash);
}

async function routeFromHash() {
  const match = window.location.hash.match(/^#trip\/(\d+)$/);
  if (match) await openTrip(Number(match[1]), false);
  else showTrips(false);
}

async function exchangeLoginFromHash() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const token = params.get("login");
  if (!token) return;

  history.replaceState({}, "", "#login");
  try {
    await api("/api/login", {
      method: "POST",
      body: JSON.stringify({ token }),
    });
    history.replaceState({}, "", "#trips");
  } catch (error) {
    toast(error.message);
    throw error;
  }
}

async function init() {
  setLoading(true, "Проверяю вход…");
  try {
    await exchangeLoginFromHash();
    state.me = await api("/api/me");
    state.accessRevision += 1;
    state.accessUnconfirmed = false;
    $("accessGate").hidden = true;
    clearCoverState();
    $("userName").textContent = state.me.name;
    $("loginGate").hidden = true;
    $("appShell").hidden = false;
    try { await loadTrips(); } catch (error) {
      if (error.status === 401 || error.status === 403) throw error;
      toast("Список поездок временно не обновился. Проверяю сохранённую заявку отдельно.");
    }
    await routeFromHash();
    await resumeGenerationReceipt();
  } catch (error) {
    $("appShell").hidden = true;
    $("accessGate").hidden = !isAccessDenied(error);
    $("loginGate").hidden = isAccessDenied(error);
    if (isAccessDenied(error)) $("accessGateMessage").textContent = ACCESS_DENIED_MESSAGE;
    else if (error.status && error.status !== 401) toast(error.message);
  } finally {
    setLoading(false);
  }
}

bindEvents();
init();
