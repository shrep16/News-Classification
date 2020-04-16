from flask import Flask
from flask_restplus import Api, Resource

flask_app = Flask(__name__)
app = Api(app=flask_app, title = "News Classification", description="This application classifies the news into different categories")

ns = app.namespace('main', description = 'Main APIs')

@ns.route("/")
class MainClass(Resource):
    def get(self):
        return {"status": "Got new data"}

    def post(self):
        return {"status": "Posted new data"}

if __name__ == "__main__":
    flask_app.run()
