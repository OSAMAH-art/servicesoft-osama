import os
import sqlite3
from functools import wraps

from flask import Flask, request, redirect, url_for, session, render_template, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

import search_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "users.db")
INDEX_DIR = os.path.join(BASE_DIR, "data", "index")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )"""
    )
    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT id, username, password_hash FROM users WHERE username=?", (username,)
    )
    row = cur.fetchone()
    conn.close()
    return row


def create_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return wrapper


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("الرجاء تعبئة كل الحقول")
        elif create_user(username, password):
            flash("تم إنشاء الحساب، سجل دخولك الآن")
            return redirect(url_for("login"))
        else:
            flash("اسم المستخدم موجود مسبقاً")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            return redirect(url_for("dashboard"))
        flash("اسم المستخدم أو كلمة المرور غير صحيحة")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    index_ready = os.path.exists(os.path.join(INDEX_DIR, "embeddings.npy"))
    return render_template(
        "dashboard.html", username=session.get("username"), index_ready=index_ready
    )


@app.route("/api/search", methods=["POST"])
@login_required
def api_search():
    if "image" not in request.files:
        return jsonify({"error": "لم يتم إرسال صورة"}), 400
    file = request.files["image"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "ملف الصورة غير صالح"}), 400

    if not os.path.exists(os.path.join(INDEX_DIR, "embeddings.npy")):
        return (
            jsonify(
                {
                    "error": "قاعدة بيانات الصور غير جاهزة بعد. شغّل build_index.py أولاً (راجع README)"
                }
            ),
            500,
        )

    results = search_engine.search(img, INDEX_DIR, top_k=5)
    return jsonify({"results": results})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
