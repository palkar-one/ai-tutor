import streamlit as st
import sqlite3
import hashlib
from auth import init_roadmap_table, init_user_roadmaps_table



DB_NAME = "users.db"

# ----- DB Functions -----
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    return True

def validate_user(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (username, password_hash))
    result = c.fetchone()
    conn.close()
    return result is not None
def login_required():
    if not st.session_state.get("logged_in", False):
        st.warning("You must log in to access this page.")
        st.stop()

# ----- Page Functions -----
def login_page():
    st.title("Login to Streamlit App")

    username = st.text_input("Username", key="username_input")
    password = st.text_input("Password", type="password", key="password_input")   

    if st.button("Login"):
        if validate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.subheader("New user? Create an account")
    new_username = st.text_input("New Username", key="new_username_input")
    new_password = st.text_input("New Password", type="password", key="new_password_input")

    if st.button("Sign Up"):
        if new_username and new_password:
            success = add_user(new_username, new_password)
            if success:
                st.success("Account created successfully. You can now log in.")
            else:
                st.warning("Username already exists.")
        else:
            st.warning("Please enter both a username and password.")

def authenticated_page():
    st.sidebar.success(f"Logged in as {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.title("Welcome to Your Streamlit App!")
    st.write("This is protected content visible only to authenticated users.")

def login_required(func):
    def wrapper(*args, **kwargs):
        if not st.session_state.get("logged_in", False):
            st.warning("Please login to access this page.")
            login_page()
            st.stop()
        return func(*args, **kwargs)
    return wrapper

# ----- Main -----
def main():
    

    if st.session_state.get("logged_in"):
        authenticated_page()
    else:
        login_page()


if __name__ == "__main__":
    init_roadmap_table()
    init_user_roadmaps_table()
    main()
