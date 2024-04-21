from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('hep_fe.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[x for x in request.form.values()]
    final=[np.array(int_features)]
    final[0][0] = (final[0][0].astype(float)-4.120000e+01)/12.565878
    final[0][1] = -0.33927557 if final[0][1] == 'Male' else 2.94745653
    final[0][2] = 0.98082889 if final[0][2] == 'Yes' else -1.01954582
    final[0][3] = 0.42802583 if final[0][3] == 'Yes' else -2.33630763
    final[0][4] = 1.36761485 if final[0][4] == 'Yes' else -0.73120002
    final[0][5] = 0.80556575 if final[0][5] == 'Yes' else -1.24136361
    final[0][6] = 0.51006137 if final[0][6] == 'Yes' else -1.96054839
    final[0][7] = 0.43852901 if final[0][7] == 'Yes' else -2.28035085
    final[0][8] = 0.79471941 if final[0][8] == 'Yes' else -1.25830574
    final[0][9] = 0.48989795 if final[0][9] == 'Yes' else -2.04124145
    final[0][10] = 0.70027467 if final[0][10] == 'Yes' else -1.42801109
    final[0][11] = 0.38490018 if final[0][11] == 'Yes' else -2.59807621
    final[0][12] = 0.36247326 if final[0][12] == 'Yes' else -2.75882423
    final[0][13] = (final[0][13].astype(float)-1.015226e+02)/47.082042
    final[0][14] = (final[0][14].astype(float)-8.419355e+01)/89.097648
    final[0][15] = (final[0][15].astype(float)-3.836129e+00)/0.619267
    final[0][16] = 1.10194633 if final[0][16] == 'Yes' else -0.90748521
    final = final[0].astype(np.float)
    #final = final.reshape(1,17)
    prediction=model.predict([final])
    print(prediction)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)

    if prediction[0] == 1:
        return render_template('hep_fe.html',pred='High Chances of Hepatitis B virus')
    else:
        return render_template('hep_fe.html',pred='Low chances of Hepatitis B virus')


if __name__ == '__main__':
    app.run(debug=True)
