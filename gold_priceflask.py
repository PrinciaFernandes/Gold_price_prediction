from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import yfinance as yf

app = Flask(__name__)


# Load the model
with open('gold_price_prediction.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load your dataset for preprocessing
df1 = pd.read_csv('C:/Users/Princia/colab/Gold  Historical Data.csv')
df2 = pd.read_csv('C:/Users/Princia/colab/Gold Historical Data 2.csv')
df2.dropna(inplace = True)
df = pd.concat([df1,df2]).reset_index(drop = True)
df = df.drop(['Vol.','Change %'],axis=1)
numcols=['Price','Open','High','Low']
df[numcols]=df[numcols].replace({',':''},regex=True)
df[numcols]=df[numcols].astype('float64')

df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y')



# Function to fetch recent gold price data
def fetch_recent_data():
    gold_data = yf.download('GC=F', start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    gold_data.reset_index(inplace = True)
    gold_data['Price'] = gold_data['Close']
    
    return gold_data

# Update your dataset 
def update_dataset(existing_data):
    recent_data = fetch_recent_data()
    updated_data = pd.concat([existing_data[['Date', 'Price']], recent_data[['Date', 'Price']]]).reset_index(drop = True)
    updated_data['Date'] = pd.to_datetime(updated_data['Date'],format='%d/%m/%Y')
    return updated_data
    
    
    
window_size = 60
def create_sequences(data, window_size):
    sequences = []
    sequences.append(data[0:0 + window_size])
    return np.array(sequences)

@app.route('/')
def home():
    return render_template('goldwebpage.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    data = request.form['date']
    if data != str(datetime.today().date()):
        date = pd.to_datetime(data, format='%Y-%m-%d')
        date = date.date()
        df_sorted = df.sort_index()
        df_sorted['Date'] = pd.to_datetime(df_sorted['Date'],format='%Y-%m-%d')
        df_sorted['Date'] = [i.date() for i in df_sorted['Date']]
      
        while(date not in df_sorted['Date'].values):          
            date = date - timedelta(days=1)
         
        
        
        end_index = df_sorted[df_sorted['Date'] == date].index[0]        
        start_index = max(0, end_index - 59)  
        
        data = df_sorted.iloc[start_index:end_index + 1]['Price'].values
        
        if len(data) < 60:
            return render_template('goldwebpage.html',error= 'Not enough data to make a prediction. Ensure you have at least 60 days of data.')
        
        # Normalize the input data
        data = scaler.transform(data.reshape(-1, 1))
        
        # Create the sequence
        sequence = create_sequences(data, window_size)[-1]
        sequence = sequence.reshape(1, window_size, 1)
          
        # Make the prediction
        prediction = model.predict(sequence)
        prediction = scaler.inverse_transform(prediction)
        
        return render_template('goldwebpage.html',prediction = prediction[0][0])
        
    else :
        date = pd.to_datetime(data, format='%Y-%m-%d')
        
        print(date)
        updated_data = df
        
        
        # Ensure the DataFrame is sorted by date
        df_sorted = updated_data.sort_index()
             
        
        if (date not in df_sorted['Date'].values):
            date = df_sorted.iloc[len(df_sorted)-1]['Date']
        
        end_index = df_sorted[df_sorted['Date'] == date].index[-1]
        
        
        # Get the last 60 rows up to the given date
        start_index = max(0, end_index - 59)  # Ensure start_index is not negative
        
        
        if df_sorted.iloc[start_index:end_index + 1].empty:
            return "No data available for the specified date range"
        data = df_sorted.iloc[start_index:end_index + 1]['Price'].values
       
        # Scale the data
        data = scaler.transform(data.reshape(-1, 1))
        
        # Create the sequence
        sequence = create_sequences(data, window_size)[-1]
        sequence = sequence.reshape(1, window_size, 1)
        
        # Make the prediction
        prediction = model.predict(sequence)
        prediction = scaler.inverse_transform(prediction)
        
        return render_template('goldwebpage.html', prediction = prediction[0][0]) 

if __name__ == '__main__':
    app.run(debug = True)
