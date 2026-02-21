"""
AIFit - 유저 인증 (bcrypt)
"""
import bcrypt
from .database import get_connection


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def register_user(username: str, password: str) -> tuple[bool, str]:
    """회원가입. (성공 여부, 메시지) 반환"""
    if len(username) < 2:
        return False, "아이디는 2자 이상이어야 합니다."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            return False, "이미 존재하는 아이디입니다."

        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, hash_password(password)),
        )
        conn.commit()
        return True, "회원가입이 완료되었습니다!"
    finally:
        conn.close()


def login_user(username: str, password: str) -> tuple[bool, str, int | None]:
    """로그인. (성공 여부, 메시지, user_id) 반환"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return False, "존재하지 않는 아이디입니다.", None
        if not verify_password(password, row["password_hash"]):
            return False, "비밀번호가 일치하지 않습니다.", None
        return True, f"{username}님 환영합니다!", row["id"]
    finally:
        conn.close()
