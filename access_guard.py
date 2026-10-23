"""Bind paid work to the access grant captured when it was accepted.

Scopes are thread/task local. Every subsequent provider call rechecks the same
grant; a new invitation cannot revive work accepted under an old grant.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import access_control

_scope = ContextVar("paid_access_scope", default=None)


def snapshot(tg_id, expected=None):
    current = access_control.require_access(tg_id)["grant_id"]
    inherited = expected_grant(tg_id)
    if (expected is not None and current != expected) or (inherited is not None and current != inherited):
        raise access_control.AccessDeniedError("Доступ к этой операции изменился.")
    return current


def expected_grant(tg_id):
    current = _scope.get()
    if current is not None and current[0] != tg_id:
        raise access_control.AccessDeniedError("Доступ к операции закрыт.")
    return current[1] if current is not None else None


@contextmanager
def operation(tg_id, expected_grant=None):
    inherited = _scope.get()
    if inherited is not None:
        if inherited[0] != tg_id:
            raise access_control.AccessDeniedError("Доступ к операции закрыт.")
        if expected_grant is not None and expected_grant != inherited[1]:
            raise access_control.AccessDeniedError("Доступ к этой операции изменился.")
        expected_grant = inherited[1]
    grant = snapshot(tg_id)
    if expected_grant is not None and expected_grant != grant:
        raise access_control.AccessDeniedError("Доступ изменился. Начните новую операцию.")
    token = _scope.set((tg_id, grant))
    try:
        yield
    finally:
        _scope.reset(token)


def check():
    current = _scope.get()
    if current is not None and snapshot(current[0]) != current[1]:
        raise access_control.AccessDeniedError("Доступ к этой операции завершён.")


def guard_user_operation(function):
    @wraps(function)
    def guarded(tg_id, *args, **kwargs):
        with operation(tg_id, expected_grant=kwargs.pop("_access_grant", None)):
            return function(tg_id, *args, **kwargs)
    return guarded
