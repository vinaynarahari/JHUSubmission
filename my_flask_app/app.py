# app.py

from flask import Flask
from routes.predict import predict_blueprint

app = Flask(__name__)

# Register the predict route blueprint
app.register_blueprint(predict_blueprint)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
