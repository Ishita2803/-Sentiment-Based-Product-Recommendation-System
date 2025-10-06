from flask import Flask, render_template, request
from model import product_recommendations_user

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    username = ''
    if request.method == 'POST':
        username = request.form.get('username')
        results = product_recommendations_user(username)
        if results is None:
            results = "User not found. Please try another username."
    return render_template('index.html', recommendations=results, username=username)

if __name__ == '__main__':
    app.run(debug=True)