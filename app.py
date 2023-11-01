from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
#
CORS(app)

api = Api(app)


class prediction(Resource):
    def get(self, bid_tendency,bid_ratio,succ_outbid,win_ratio):
        try:
            # print(bid_tendency,bid_ratio,succ_outbid,win_ratio)

            values = {'Bidder_Tendency':bid_tendency,'Bidding_Ratio':bid_ratio,'Successive_Outbidding':succ_outbid,'Winning_Ratio':win_ratio}

            df = pd.DataFrame(values, index=([0]))
            model_loaded = load_model('DNN.h5')
            predict_s = model_loaded.predict(df)
            print(predict_s)
            return str(np.round(predict_s[0]))


        except Exception as e:
            print(e)
            # return e



api.add_resource(prediction,'/predict/<float:bid_tendency>&<float:bid_ratio>&<float:succ_outbid>&<float:win_ratio>')



if __name__ == '__main__':
    app.run(debug=True)


    #http://127.0.0.1:5000/predict/0.5&0.5&0.5&0.5