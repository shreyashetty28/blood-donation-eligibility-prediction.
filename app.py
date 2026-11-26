from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

with open("blood_donation_rf_model.pkl","rb") as f:
    model=pickle.load(f)

with open("gender_encoder.pkl","rb") as f:
    gender_encoder=pickle.load(f)

with open("blood_group_encoder.pkl","rb") as f:
    blood_encoder=pickle.load(f)

with open("target_encoder.pkl","rb") as f:
    target_encoder=pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/methodology")
def methodology():
    return render_template("methodology.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")



@app.route("/predict",methods=["POST"])
def predict():

    gender=request.form["gender"]

    age=float(request.form["age"])

    blood_group=request.form["blood_group"]

    weight=float(request.form["weight"])

    hemoglobin=float(request.form["hemoglobin"])

    display_gender = gender
    display_age = age
    display_blood_group = blood_group
    display_weight = weight
    display_hemoglobin = hemoglobin

    gender=gender_encoder.transform([gender])[0]

    blood_group=blood_encoder.transform([blood_group])[0]

    prediction=model.predict([[gender,
                               age,
                               blood_group,
                               weight,
                               hemoglobin]])

    result=target_encoder.inverse_transform(prediction)[0]

    if result=="Yes":
        text="Eligible for Blood Donation"

    else:
        text="Not Eligible for Blood Donation"

    return render_template(
    "index.html",
    prediction_text=text,
    gender=request.form["gender"],
    age=request.form["age"],
    blood_group=request.form["blood_group"],
    weight=request.form["weight"],
    hemoglobin=request.form["hemoglobin"]
)


if __name__=="__main__":
    app.run(debug=True)