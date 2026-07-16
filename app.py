# -*- coding: utf-8 -*-
"""
Hajj & Umrah Sentiment Analysis System — Flask backend API.

Run in VS Code:
    1) python -m venv venv
    2) venv\\Scripts\\activate      (Windows)   or   source venv/bin/activate   (Mac/Linux)
    3) pip install -r requirements.txt
    4) python app.py
    -> API runs on http://localhost:5000

Endpoints:
    GET    /api/health
    POST   /api/analyze                body: {"text": "..."}
    GET    /api/comments               query: search, sentiment, category, sort, page, per_page
    POST   /api/comments               body: {"text": "...", "category": "..."}
    PUT    /api/comments/<id>          body: {"text": "..."}
    DELETE /api/comments/<id>
    GET    /api/comments/export        query: format=csv
    GET    /api/dashboard/stats
    POST   /api/auth/login             body: {"email": "...", "password": "..."}
    POST   /api/auth/signup            body: {"name": "...", "email": "...", "password": "..."}
    POST   /api/auth/guest             (Continue as Guest — no credentials)
    POST   /api/auth/forgot-password   body: {"email": "..."}  (simulated — no email is actually sent)
    GET    /api/users
    POST   /api/users                  body: {"name","email","role","password"}
    PUT    /api/users/<id>             body: {"name","email","role"}
    DELETE /api/users/<id>

Roles: admin (full access) / guest (view + analyze only).
Fixed admin (cannot be deleted or demoted): abdullah1222@gmail.com / Ab1231234
Admin-only endpoints (require "Authorization: Bearer <token>" from login):
    /api/users (all methods), PUT/DELETE /api/comments/<id>
"""
import os
import io
import csv
import sqlite3
from datetime import datetime, timedelta, timezone

from functools import wraps

from flask import Flask, request, jsonify, g, Response, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import joblib

from lexicon import find_keywords
from train_model import train_and_save, MODEL_PATH
from dataset import TRAIN_DATA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hajj_umrah.db")
CATEGORIES = ["Services", "Crowd Management", "Transportation", "Food",
              "Staff Behavior", "Accommodation", "General"]

# ---- Roles & fixed admin account ----
# The system has exactly two roles:
#   admin -> full access (users page, delete/edit operations)
#   guest -> read/analyze only
VALID_ROLES = ("admin", "guest")
ADMIN_EMAIL = "abdullah1222@gmail.com"  # fixed admin — cannot be deleted or demoted
ADMIN_PASSWORD = "Ab1231234"
LEGACY_ADMIN_EMAIL = "admin@hajj.sa"    # old seeded admin — migrated automatically
SECRET_KEY = os.environ.get("SECRET_KEY", "hajj-umrah-dev-secret-change-in-production")
TOKEN_MAX_AGE = 60 * 60 * 12  # 12 hours
_serializer = URLSafeTimedSerializer(SECRET_KEY, salt="auth-token")

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# ---- CORS (manual, no extra dependency required) ----
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def options_handler(_any):
    return "", 204


# ---------------------------------------------------------------- #
# Database helpers (plain sqlite3 — no ORM dependency needed)
# ---------------------------------------------------------------- #
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            category TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            keywords TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'guest',
            created_at TEXT NOT NULL
        )
    """)
    # Always make sure the fixed admin account exists with role=admin,
    # regardless of what else is in the users table.
    admin_row = conn.execute("SELECT id, role FROM users WHERE lower(email)=?",
                             (ADMIN_EMAIL,)).fetchone()
    if admin_row is None:
        # Migrate the old seeded admin (if this DB predates the email change).
        legacy = conn.execute("SELECT id FROM users WHERE lower(email)=?",
                              (LEGACY_ADMIN_EMAIL,)).fetchone()
        if legacy:
            conn.execute(
                "UPDATE users SET email=?, password_hash=?, role='admin' WHERE id=?",
                (ADMIN_EMAIL, generate_password_hash(ADMIN_PASSWORD), legacy[0]),
            )
            print(f"Migrated fixed admin -> {ADMIN_EMAIL}")
        else:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
                ("Admin User", ADMIN_EMAIL, generate_password_hash(ADMIN_PASSWORD), "admin",
                 datetime.now(timezone.utc).isoformat()),
            )
            print(f"Seeded fixed admin -> email: {ADMIN_EMAIL}")
    elif admin_row[1] != "admin":
        conn.execute("UPDATE users SET role='admin' WHERE id=?", (admin_row[0],))
    # Migrate any legacy roles (viewer/manager/...) to 'guest'.
    conn.execute("UPDATE users SET role='guest' WHERE role NOT IN ('admin','guest')")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    if count == 0:
        # Seed the database with the labeled training examples so the
        # dashboard/comments pages have real, non-empty data on first run.
        for i, (text, label) in enumerate(TRAIN_DATA):
            category = CATEGORIES[i % len(CATEGORIES)]
            pos_hits, neg_hits = find_keywords(text)
            keywords = ",".join(pos_hits + neg_hits)
            confidence = 78.0 + (i % 15)
            created_at = (datetime.now(timezone.utc) - timedelta(hours=i * 6)).isoformat()
            conn.execute(
                "INSERT INTO comments (text, category, sentiment, confidence, keywords, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (text, category, label, confidence, keywords, created_at),
            )
        conn.commit()
    conn.close()


# ---------------------------------------------------------------- #
# ML model loading (trains automatically the first time it's needed)
# ---------------------------------------------------------------- #
def load_model():
    model_path = os.path.join(BASE_DIR, MODEL_PATH)
    if not os.path.exists(model_path):
        print("No trained model found — training a fresh one now...")
        return train_and_save()
    return joblib.load(model_path)


MODEL = load_model()


def run_sentiment_analysis(text: str):
    probs = MODEL.predict_proba([text])[0]
    classes = list(MODEL.classes_)
    scores = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
    label = classes[int(probs.argmax())]
    confidence = scores[label]
    pos_hits, neg_hits = find_keywords(text)
    return {
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "positive_keywords": pos_hits,
        "negative_keywords": neg_hits,
        "keywords": pos_hits + neg_hits,
    }


# ---------------------------------------------------------------- #
# Auth helpers (token-based, stateless)
# ---------------------------------------------------------------- #
def make_token(user_id: int) -> str:
    return _serializer.dumps({"uid": user_id})


def current_user():
    """Return the users row for the Bearer token in this request, or None."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        data = _serializer.loads(auth[7:].strip(), max_age=TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    return get_db().execute("SELECT * FROM users WHERE id=?", (data.get("uid"),)).fetchone()


def admin_required(fn):
    """Protect admin-only endpoints (users page + destructive operations)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None or user["role"] != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return fn(*args, **kwargs)
    return wrapper


def is_fixed_admin(row) -> bool:
    return (row["email"] or "").lower() == ADMIN_EMAIL


# ---------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------- #
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_classes": list(MODEL.classes_)})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    result = run_sentiment_analysis(text)
    return jsonify(result)


@app.route("/api/comments", methods=["GET"])
def list_comments():
    db = get_db()
    search = request.args.get("search", "").strip()
    sentiment = request.args.get("sentiment", "all")
    category = request.args.get("category", "all")
    sort = request.args.get("sort", "date")
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, int(request.args.get("per_page", 10)))

    query = "SELECT * FROM comments WHERE 1=1"
    params = []
    if search:
        query += " AND (text LIKE ? OR keywords LIKE ?)"
        params += [f"%{search}%", f"%{search}%"]
    if sentiment != "all":
        query += " AND sentiment = ?"
        params.append(sentiment)
    if category != "all":
        query += " AND category = ?"
        params.append(category)

    order = "created_at DESC" if sort == "date" else "confidence DESC"
    rows = db.execute(query + f" ORDER BY {order}", params).fetchall()
    total = len(rows)
    start = (page - 1) * per_page
    page_rows = rows[start:start + per_page]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "items": [dict(r) for r in page_rows],
    })


@app.route("/api/comments", methods=["POST"])
def add_comment():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    category = data.get("category") or "General"
    if not text:
        return jsonify({"error": "text is required"}), 400

    result = run_sentiment_analysis(text)
    db = get_db()
    cur = db.execute(
        "INSERT INTO comments (text, category, sentiment, confidence, keywords, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (text, category, result["label"], result["confidence"],
         ",".join(result["keywords"]), datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    new_row = db.execute("SELECT * FROM comments WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify(dict(new_row)), 201


@app.route("/api/comments/<int:comment_id>", methods=["PUT"])
@admin_required
def update_comment(comment_id):
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    result = run_sentiment_analysis(text)
    db = get_db()
    db.execute(
        "UPDATE comments SET text=?, sentiment=?, confidence=?, keywords=? WHERE id=?",
        (text, result["label"], result["confidence"], ",".join(result["keywords"]), comment_id),
    )
    db.commit()
    row = db.execute("SELECT * FROM comments WHERE id=?", (comment_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.route("/api/comments/<int:comment_id>", methods=["DELETE"])
@admin_required
def delete_comment(comment_id):
    db = get_db()
    db.execute("DELETE FROM comments WHERE id=?", (comment_id,))
    db.commit()
    return jsonify({"deleted": comment_id})


@app.route("/api/comments/export")
def export_comments():
    fmt = request.args.get("format", "csv")
    db = get_db()
    rows = db.execute("SELECT * FROM comments ORDER BY created_at DESC").fetchall()

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "text", "category", "sentiment", "confidence", "created_at"])
        for r in rows:
            writer.writerow([r["id"], r["text"], r["category"], r["sentiment"], r["confidence"], r["created_at"]])
        return Response(
            output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=comments.csv"},
        )
    return jsonify([dict(r) for r in rows])


def user_public(row):
    d = dict(row)
    d.pop("password_hash", None)
    return d


# ---- Authentication (simple, session-less: returns the user object) ---- #
@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE lower(email)=?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"user": user_public(row), "token": make_token(row["id"])})


@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not name or not email or len(password) < 6:
        return jsonify({"error": "name, email and a password of 6+ characters are required"}), 400
    db = get_db()
    exists = db.execute("SELECT id FROM users WHERE lower(email)=?", (email,)).fetchone()
    if exists:
        return jsonify({"error": "An account with this email already exists"}), 409
    cur = db.execute(
        "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
        (name, email, generate_password_hash(password), "guest", datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify({"user": user_public(row), "token": make_token(row["id"])}), 201


@app.route("/api/auth/guest", methods=["POST"])
def guest_login():
    """'Continue as Guest' — returns a guest identity with no token.

    Guests can browse the dashboard, comments, analytics and reports and can
    analyze/add comments, but every admin-only endpoint (users management,
    deleting/editing comments) rejects them because they carry no admin token.
    """
    return jsonify({
        "user": {"id": None, "name": "Guest", "email": None, "role": "guest", "guest": True},
        "token": None,
    })


@app.route("/api/auth/forgot-password", methods=["POST"])
def forgot_password():
    # NOTE: no email server is configured, so this only confirms whether the
    # flow ran — it does not actually send an email. Wire up an SMTP/email
    # provider (e.g. Flask-Mail) here for real password-reset emails.
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    db = get_db()
    row = db.execute("SELECT id FROM users WHERE lower(email)=?", (email,)).fetchone()
    return jsonify({
        "message": "If this email is registered, a reset link would be sent to it."
        if row else "If this email is registered, a reset link would be sent to it."
    })


# ---- Users management ---- #
@app.route("/api/users", methods=["GET"])
@admin_required
def list_users():
    db = get_db()
    rows = db.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
    return jsonify([user_public(r) for r in rows])


@app.route("/api/users", methods=["POST"])
@admin_required
def add_user():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    role = data.get("role") or "guest"
    password = data.get("password") or "changeme123"
    if not name or not email:
        return jsonify({"error": "name and email are required"}), 400
    if role not in VALID_ROLES:
        return jsonify({"error": "role must be 'admin' or 'guest'"}), 400
    db = get_db()
    if db.execute("SELECT id FROM users WHERE lower(email)=?", (email,)).fetchone():
        return jsonify({"error": "An account with this email already exists"}), 409
    cur = db.execute(
        "INSERT INTO users (name, email, password_hash, role, created_at) VALUES (?,?,?,?,?)",
        (name, email, generate_password_hash(password), role, datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify(user_public(row)), 201


@app.route("/api/users/<int:user_id>", methods=["PUT"])
@admin_required
def update_user(user_id):
    data = request.get_json(force=True, silent=True) or {}
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    name = data.get("name", row["name"])
    email = (data.get("email") or row["email"]).strip().lower()
    role = data.get("role", row["role"])
    if role not in VALID_ROLES:
        return jsonify({"error": "role must be 'admin' or 'guest'"}), 400
    if is_fixed_admin(row):
        # The fixed admin account cannot be demoted or have its email changed.
        if role != "admin":
            return jsonify({"error": "The primary admin account role cannot be changed"}), 403
        if email != ADMIN_EMAIL:
            return jsonify({"error": "The primary admin account email cannot be changed"}), 403
    db.execute("UPDATE users SET name=?, email=?, role=? WHERE id=?", (name, email, role, user_id))
    db.commit()
    row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return jsonify(user_public(row))


@app.route("/api/users/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if row and is_fixed_admin(row):
        return jsonify({"error": "The primary admin account cannot be deleted"}), 403
    db.execute("DELETE FROM users WHERE id=?", (user_id,))
    db.commit()
    return jsonify({"deleted": user_id})


@app.route("/api/dashboard/stats")
def dashboard_stats():
    db = get_db()
    rows = db.execute("SELECT * FROM comments").fetchall()
    total = len(rows)
    pos = sum(1 for r in rows if r["sentiment"] == "positive")
    neg = sum(1 for r in rows if r["sentiment"] == "negative")
    neu = total - pos - neg

    by_category = {}
    for r in rows:
        c = by_category.setdefault(r["category"], {"total": 0, "positive": 0, "negative": 0, "neutral": 0})
        c["total"] += 1
        c[r["sentiment"]] = c.get(r["sentiment"], 0) + 1

    keyword_freq = {}
    for r in rows:
        if r["keywords"]:
            for kw in r["keywords"].split(","):
                kw = kw.strip()
                if kw:
                    keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
    top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:12]

    return jsonify({
        "total": total, "positive": pos, "negative": neg, "neutral": neu,
        "positive_pct": round(pos / total * 100, 1) if total else 0,
        "negative_pct": round(neg / total * 100, 1) if total else 0,
        "neutral_pct": round(neu / total * 100, 1) if total else 0,
        "by_category": by_category,
        "top_keywords": top_keywords,
    })


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))
