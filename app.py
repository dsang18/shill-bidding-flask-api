from flask import Flask, request
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS

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
            xgb_model_loaded = pickle.load(open('xgboost_model.pkl','rb'))
            predict_s = xgb_model_loaded.predict(df)
            # print(predict_s)
            return str(predict_s[0])


        except Exception as e:
            print(e)
            # return e



api.add_resource(prediction,'/predict/<float:bid_tendency>&<float:bid_ratio>&<float:succ_outbid>&<float:win_ratio>')



if __name__ == '__main__':
    app.run(debug=True)


    #http://127.0.0.1:5000/predict0.5&0.5&0.5&0.5