from flask import Flask,request,render_template
import pickle
import  numpy as np

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app=Flask(__name__)

@app.route('/',methods=["GET","POST"])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def model():
    if request.method == 'POST':
        preg=int(request.form['pregnancies'])
        gluc=int(request.form['glucose'])
        bp=int(request.form['bp'])
        skint=int(request.form['skint'])
        insulin=int(request.form['insulin'])
        bmi=float(request.form['bmi'])
        dpf=float(request.form['dpf'])
        age=int(request.form['age'])

        data = np.array([[preg, gluc, bp, skint, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        return render_template('result.html',pred=my_prediction)





if __name__=="__main__":
    app.run(debug=True)