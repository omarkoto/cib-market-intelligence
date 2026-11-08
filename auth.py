import hashlib
import json

USERS_FILE = "users.json"

# Which data sources each role is allowed to read.
ROLE_SOURCES = {
    "admin": ["r8-train", "r8-test"],
    "member": ["r8-train"],
}


# Turns a password into a SHA-256 hash (a fixed-length scrambled text).
# WHY: we never store or compare plain passwords, only their hashes.
def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# Checks a username and password against users.json; returns the role or None.
# WHY: this is the single place that decides who may log in and what role they get.
def check_login(username, password):
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)

    if username not in users:
        return None
    if users[username]["password_hash"] == hash_password(password):
        return users[username]["role"]
    return None
