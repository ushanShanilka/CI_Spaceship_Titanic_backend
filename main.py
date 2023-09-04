from flask import Flask, request
import pickle
from flask_cors import CORS
import pandas as pd 

app = Flask(__name__)
CORS(app)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    print(data['data'])

    data_df = pd.DataFrame([data['data']], columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])

    prediction = model.predict(data_df)
    return str(prediction[0])


if __name__ == "__main__":
    app.run(debug=True)
