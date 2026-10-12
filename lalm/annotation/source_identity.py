"""Recognize explicit production session identities; never infer from filenames."""
import re
import uuid


def production_conversation(key):
    if not key.startswith('staging/bitmanager-sessions/'):
        return None
    match = re.search(r'/sessions/\d{4}/\d{2}/\d{2}/([0-9a-fA-F-]{36})/'
                      r'(?:agent|customer)\.opus$', key)
    if match is None:
        return None
    try:
        return str(uuid.UUID(match.group(1)))
    except ValueError:
        return None
