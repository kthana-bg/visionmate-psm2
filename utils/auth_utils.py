import hashlib
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database.db_manager import (
    create_user,
    get_user_by_username,
    get_all_users,
    save_face_embedding,
    get_face_embedding,
    get_all_face_embeddings,
)

FACE_MATCH_THRESHOLD = 0.50


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    return hash_password(password) == stored_hash


def _embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def find_matching_user(new_embedding: np.ndarray) -> dict | None:
    all_stored = get_all_face_embeddings()
    best_distance = float("inf")
    best_user_id = None

    for entry in all_stored:
        dist = _embedding_distance(new_embedding, entry["embedding"])
        if dist < best_distance:
            best_distance = dist
            best_user_id = entry["user_id"]

    if best_distance < FACE_MATCH_THRESHOLD and best_user_id is not None:
        all_users = get_all_users()
        for u in all_users:
            if u["id"] == best_user_id:
                return u
    return None


def is_duplicate_face(new_embedding: np.ndarray) -> bool:
    return find_matching_user(new_embedding) is not None


def register_user_face(full_name: str, face_embedding: np.ndarray) -> dict:
    if face_embedding is None:
        return {"success": False, "message": "No face detected. Please capture your face first."}

    existing = find_matching_user(face_embedding)
    if existing is not None:
        name = existing.get("full_name", existing.get("username", "another account"))
        return {
            "success": False,
            "message": (
                f"This face is already registered as '{name}'. "
                "Please use the Login tab to sign in with your face."
            ),
        }

    import re
    base = re.sub(r"[^a-z0-9]", "", full_name.lower().replace(" ", "_"))
    base = base[:20] or "user"
    username = base
    suffix = 1
    while get_user_by_username(username):
        username = f"{base}{suffix}"
        suffix += 1

    placeholder_hash = hash_password(f"face_only_{username}")

    user_id = create_user(username, placeholder_hash, full_name)
    if user_id == -1:
        return {"success": False, "message": "Failed to create account. Please try again."}

    save_face_embedding(user_id, face_embedding)
    return {
        "success": True,
        "message": "Account created successfully.",
        "username": username,
    }


def login_by_face(face_embedding: np.ndarray) -> dict:
    if face_embedding is None:
        return {"success": False, "message": "No face detected. Please try again."}

    matched_user = find_matching_user(face_embedding)

    if matched_user is None:
        return {
            "success": False,
            "message": (
                "Face not recognised. "
                "If you are new, please register first."
            ),
        }

    return {
        "success": True,
        "message": f"Welcome back, {matched_user.get('full_name', matched_user['username'])}!",
        "user": matched_user,
    }


def verify_face_identity(user_id: int, live_embedding: np.ndarray) -> bool:
    stored = get_face_embedding(user_id)
    if stored is None:
        return True
    return _embedding_distance(live_embedding, stored) < FACE_MATCH_THRESHOLD
