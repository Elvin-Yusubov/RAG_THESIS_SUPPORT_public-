"""Per-user authentication for the web app (salted PBKDF2 password hashes).

Accounts are configured in ``APP_USERS`` as ``name:salt$hash`` entries
separated by semicolons. Generate an entry with::

    python -m thesis_assistant.auth

Passwords are never stored or compared in plain text, and verification is
constant-time.
"""

import getpass
import hashlib
import hmac
import secrets
from typing import Dict, Optional

PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Return ``salt$hash`` (hex) for a password, generating a salt if needed."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a ``salt$hash`` string in constant time."""
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)


def parse_users(raw: str) -> Dict[str, str]:
    """Parse ``name:salt$hash;name2:salt$hash`` into ``{name: salt$hash}``."""
    users = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name, _, stored = entry.partition(":")
        if name and stored:
            users[name.strip()] = stored.strip()
    return users


def authenticate(username: str, password: str, users: Dict[str, str]) -> bool:
    """Verify credentials. Unknown users still cost one hash to blunt timing probes."""
    stored = users.get(username)
    if stored is None:
        hash_password(password, "0" * 32)
        return False
    return verify_password(password, stored)


def main() -> None:
    """Print an APP_USERS entry for a username and password read from the terminal."""
    name = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    print(f"\nAdd to APP_USERS (separate several users with ';'):\n{name}:{hash_password(password)}")


if __name__ == "__main__":
    main()
