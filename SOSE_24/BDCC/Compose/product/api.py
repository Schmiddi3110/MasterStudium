# Product Service

# Import framework
from flask import Flask
from flask_restful import Resource, Api
import os

# Instantiate the app
app = Flask(__name__)
api = Api(app)

class Product(Resource):
    def get(self):
        new_item = os.environ['NEW_ITEM']
        return {
            'products': ['Ice cream', 'Chocolate', 'Fruit', 'Eggs', new_item]
        }

# Create routes
api.add_resource(Product, '/')

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)