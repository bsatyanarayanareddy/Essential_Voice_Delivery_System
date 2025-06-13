from flask import Flask, render_template, request, redirect, url_for
import json
import os
from voice_pipeline import VoiceGrocerySystem

app = Flask(__name__)
system = VoiceGrocerySystem("Daily_Grocery_Prices.csv")

# Combined login and signup route
@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'login':
            username = request.form['username']
            password = request.form['password']
            # 🔒 Dummy check (replace with real DB check)
            if username == 'admin' and password == '1234':
                return render_template('app.html')  # or return success JSON
            return "Invalid credentials", 401

        elif action == 'signup':
            # Extract signup details from form
            username = request.form['username']
            password = request.form['password']
            # TODO: Add logic to save user to DB or file
            return "Signup successful! Please log in."  # Or redirect to login

        else:
            return "Invalid action", 400

    # GET request renders combined login/signup page
    return render_template('auth.html')

@app.route('/')
def home():
    latest_bill = system.get_latest_bill()
    order_history = system.get_order_history()
    return render_template(
        'main.html',
        bill=latest_bill,
        history=order_history
    )

@app.route('/process_order', methods=['POST'])
def process_order():
    success = system.run_voice_order_pipeline()
    return redirect(url_for('home'))

if __name__ == '__main__':
    if not os.path.exists("order_history.json"):
        with open("order_history.json", "w") as f:
            json.dump([], f)
    app.run(debug=True)
