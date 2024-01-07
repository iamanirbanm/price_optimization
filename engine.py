from flask import Flask,request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima.arima.utils import ndiffs

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/engine',methods=['POST'])
def getPrices():
    try:
        product_name = request.form.get('product_name').upper()
        base_price = float(request.form.get('base_price'))
        profit_margin = float(request.form.get('profit_margin'))
        csv_file = request.files.get('csvFile')

        data = pd.read_csv(csv_file,parse_dates=['date'])
        data.columns=['date','sales','inventory']
        data = data.dropna()
        data['date'] = data.date.dt.to_period('D')
        data = data.set_index(['date']).sort_index()

        values = data['sales']

        dates = [str(date) for date in values.index]
        train = values.values

        inventory = [data['inventory'].iloc[0]]
        for i in range(1,len(data)):
            inventory.append(data['inventory'].iloc[i]-(data['inventory'].iloc[i-1]-data['sales'].iloc[i-1]))
        data['inventory']=inventory


        #-------------------------------------Sales------------------------------------
        d = ndiffs(values.values)
        model = pm.auto_arima(train, d=d, seasonal=False, suppress_warnings=True)

        ARIMAmodel_sales = ARIMA(train, order = model.order)
        ARIMAmodel_sales = ARIMAmodel_sales.fit()

        training_pred_sales = ARIMAmodel_sales.predict(start=0, end=len(train)-1)

        confidence_sales = training_pred_sales.sum()/train.sum()*100
        confidence_sales = confidence_sales if confidence_sales<100 else 200-confidence_sales


        sales_pred = ARIMAmodel_sales.get_forecast(steps = 30)
        sales_pred_df = pd.DataFrame(sales_pred.conf_int(alpha = 0.8))
        sales_pred_df.columns=['Lower','Upper']
        sales_preds = np.array(sales_pred_df['Upper'])
        #-------------------------------------Sales------------------------------------

        #-------------------------------------Inventory------------------------------------
        d = ndiffs(data['inventory'].values)
        model = pm.auto_arima(data['inventory'].values, d=d, seasonal=False, suppress_warnings=True)

        ARIMAmodel_inventory = ARIMA(data['inventory'].values, order=model.order)
        ARIMAmodel_inventory = ARIMAmodel_inventory.fit()

        training_pred_inventory = ARIMAmodel_inventory.predict(start=0, end=len(train)-1)

        confidence_inventory = training_pred_inventory.sum()/data['inventory'].values.sum()*100
        confidence_inventory = confidence_inventory if confidence_inventory<100 else 200-confidence_inventory

        inventory_pred = ARIMAmodel_inventory.get_forecast(steps = 30)
        inventory_pred_df = pd.DataFrame(inventory_pred.conf_int(alpha = 0.8))
        inventory_pred_df.columns=['Lower','Upper']
        inventory_preds = np.array(inventory_pred_df['Upper'])
        #-------------------------------------Inventory------------------------------------

        confidence = confidence_inventory*confidence_sales/100
        selling_price = round(base_price*(1 + min(1, sales_preds.sum()/inventory_preds.sum())*profit_margin/100),0)

        return jsonify({
            "Success" : True,
            "Product Name" : product_name,
            "Base Price" : base_price,
            "Profit Margin" : profit_margin,
            "Monthly Sales Forecast" : list(sales_preds),
            "Monthly Inventory Forecast" : list(inventory_preds),
            "Expected Accuracy" : confidence,
            "Recommended Price" : selling_price
        })
    except:
        return jsonify({
            "Success":False,
            "Error": "An unknown error occurred"
        })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
