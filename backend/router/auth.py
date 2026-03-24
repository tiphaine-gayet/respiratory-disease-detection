"""
backend/router/auth.py
Authentication and user registration against APP.USERS in Snowflake.
"""

from __future__ import annotations

import hmac
import os
import re
import hashlib
import random
from datetime import date
from typing import Optional

from backend.utils.snowflake_client import SnowflakeClient

_DB = os.getenv("SNOWFLAKE_DATABASE")
_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_APP") or "APP"
_USERS_TABLE = f"{_DB}.{_SCHEMA}.USERS"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_RNG = random.SystemRandom()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    """Return (password_hash_hex, salt_hex) using PBKDF2-HMAC-SHA256."""
    if salt_hex is None:
        salt = os.urandom(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)

    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return pwd_hash.hex(), salt_hex


def _validate_registration_input(full_name: str, email: str, password: str) -> Optional[str]:
    if len(full_name.strip()) < 2:
        return "Le nom complet doit contenir au moins 2 caracteres."

    email_norm = _normalize_email(email)
    if not _EMAIL_RE.match(email_norm):
        return "Format d'email invalide."

    if len(password) < 8:
        return "Le mot de passe doit contenir au moins 8 caracteres."

    return None


def _generate_nir_like_uid() -> str:
    """
    Generate a French NIR-like identifier (15 digits):
    [1 digit sex][2 year][2 month][2 dept][3 commune][3 order][2 key].
    """
    sex = str(_RNG.choice([1, 2]))

    today = date.today()
    year = f"{_RNG.randint(1950, today.year) % 100:02d}"
    month = f"{_RNG.randint(1, 12):02d}"
    dept = f"{_RNG.randint(1, 95):02d}"
    commune = f"{_RNG.randint(1, 999):03d}"
    order = f"{_RNG.randint(1, 999):03d}"

    base13 = f"{sex}{year}{month}{dept}{commune}{order}"
    key = 97 - (int(base13) % 97)
    return f"{base13}{key:02d}"


def create_user(full_name: str, email: str, password: str, is_doctor: bool) -> tuple[bool, str, Optional[dict]]:
    """Create a user in APP.USERS. Returns (ok, message, user_payload)."""
    validation_error = _validate_registration_input(full_name, email, password)
    if validation_error:
        return False, validation_error, None

    email_norm = _normalize_email(email)
    pwd_hash, pwd_salt = _hash_password(password)
    with SnowflakeClient() as client:
        cur = client.cursor()
        try:
            cur.execute(f"SELECT USER_ID FROM {_USERS_TABLE} WHERE EMAIL = %s LIMIT 1", (email_norm,))
            if cur.fetchone() is not None:
                return False, "Un compte existe deja avec cet email.", None

            # Generate a NIR-like UID and ensure no collision in USERS.
            user_id = None
            for _ in range(8):
                candidate = _generate_nir_like_uid()
                cur.execute(f"SELECT 1 FROM {_USERS_TABLE} WHERE USER_ID = %s LIMIT 1", (candidate,))
                if cur.fetchone() is None:
                    user_id = candidate
                    break

            if user_id is None:
                return False, "Impossible de generer un identifiant utilisateur unique.", None

            cur.execute(
                f"""
                INSERT INTO {_USERS_TABLE}
                    (USER_ID, FULL_NAME, EMAIL, PASSWORD_HASH, PASSWORD_SALT, IS_DOCTOR)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, full_name.strip(), email_norm, pwd_hash, pwd_salt, bool(is_doctor)),
            )

            cur.execute(
                f"""
                SELECT USER_ID, FULL_NAME, EMAIL, IS_DOCTOR
                FROM {_USERS_TABLE}
                WHERE EMAIL = %s
                LIMIT 1
                """,
                (email_norm,),
            )
            row = cur.fetchone()
        finally:
            cur.close()

    if row is None:
        return False, "Compte cree mais impossible de recuperer l'utilisateur.", None

    user = {
        "user_id": row[0],
        "full_name": row[1],
        "email": row[2],
        "is_doctor": bool(row[3]),
    }
    return True, "Compte cree avec succes.", user


def authenticate_user(email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """Authenticate a user by email and password."""
    email_norm = _normalize_email(email)

    with SnowflakeClient() as client:
        cur = client.cursor()
        try:
            cur.execute(
                f"""
                SELECT USER_ID, FULL_NAME, EMAIL, PASSWORD_HASH, PASSWORD_SALT, IS_DOCTOR
                FROM {_USERS_TABLE}
                WHERE EMAIL = %s
                  AND IS_ACTIVE = TRUE
                LIMIT 1
                """,
                (email_norm,),
            )
            row = cur.fetchone()
            if row is None:
                return False, "Email ou mot de passe invalide.", None

            stored_hash = row[3]
            stored_salt = row[4]
            candidate_hash, _ = _hash_password(password, stored_salt)
            if not hmac.compare_digest(stored_hash, candidate_hash):
                return False, "Email ou mot de passe invalide.", None

            cur.execute(
                f"UPDATE {_USERS_TABLE} SET LAST_LOGIN_AT = CURRENT_TIMESTAMP() WHERE USER_ID = %s",
                (row[0],),
            )
        finally:
            cur.close()

    user = {
        "user_id": row[0],
        "full_name": row[1],
        "email": row[2],
        "is_doctor": bool(row[5]),
    }
    return True, "Connexion reussie.", user
