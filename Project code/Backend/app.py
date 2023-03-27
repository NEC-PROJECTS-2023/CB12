from flask import Flask,render_template
from flask import request
import pickle
import numpy as np 

filename='gold.pkl'
classifier=pickle.load(open(filename,'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
        spx=request.form['SPX']
        gld=request.form['GLD']
        uso=request.form['USO']
        slv=request.form['SLV']
        data=np.array([(spx,gld,uso,slv)])
        my_prediction=classifier.predict(data)

        return render_template('index.html',prediction=my_prediction)
        
if __name__=='__main__':
    app.run(debug=False)