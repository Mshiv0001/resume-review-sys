from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
import pandas as pd
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super-secret-key')

USERS_FILE = 'users.json'

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump({}, f)

def load_users():
    with open(USERS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)

# Load model
model = joblib.load("insurance_model.pkl")

@app.route('/')
def root():
    if session.get('username'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('username'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        users = load_users()
        user = users.get(username)

        if not username or not password:
            flash('Enter both username and password.', 'error')
        elif not user or not check_password_hash(user['password'], password):
            flash('Invalid username or password.', 'error')
        else:
            session['username'] = username
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session.get('username'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')

        users = load_users()

        if not username or not password or not confirm:
            flash('Please fill in all fields.', 'error')
        elif password != confirm:
            flash('Passwords do not match.', 'error')
        elif username in users:
            flash('That username is already taken.', 'error')
        else:
            users[username] = {'password': generate_password_hash(password)}
            save_users(users)
            session['username'] = username
            flash('Account created successfully!', 'success')
            return redirect(url_for('dashboard'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('username'):
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('username'):
        return redirect(url_for('login'))

    age = float(request.form['age'])
    sex = request.form.get('sex')
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    # Build a DataFrame with the same columns the model was trained on
    input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    prediction = model.predict(input_df)

    # Format prediction to two decimal places
    pred_value = float(prediction[0])
    formatted = f"{pred_value:.2f}"

    return render_template(
        'index.html',
        username=session.get('username'),
        prediction_text=f'Predicted Insurance Cost: {formatted} Rs'
    )

if __name__ == "__main__":
    app.run(debug=True)