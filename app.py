# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'mobileprice.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        battery_power = float(request.form['battery_power'])
        blue = float(request.form['blue'])
        clock_speed = float(request.form['clock_speed'])
        dual_sim = float(request.form['dual_sim'])
        fc = float(request.form['fc'])
        four_g = float(request.form['four_g'])
        int_memory = float(request.form['int_memory'])
        m_dep = float(request.form['m_dep'])
        mobile_wt = float(request.form['mobile_wt'])
        n_cores = float(request.form['n_cores'])
        pc = float(request.form['pc'])
        px_height = float(request.form['px_height'])
        px_width = float(request.form['px_width'])
        ram = float(request.form['ram'])
        sc_h = float(request.form['sc_h'])
        sc_w = float(request.form['sc_w'])
        talk_time = float(request.form['talk_time'])
        three_g = float(request.form['three_g'])
        touch_screen = float(request.form['touch_screen'])
        wifi = float(request.form['wifi'])
        
        
        
        
        
        
        data = np.array([[battery_power,blue,clock_speed,dual_sim,fc,four_g, int_memory,m_dep,
                          mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,three_g, touch_screen,wifi]])
        my_prediction = classifier.predict(data)
        
        return render_template('results.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)