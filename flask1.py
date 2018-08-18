from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/getResult',methods=['GET','POST'])

def index():
    #get data
    age = request.json['age']
    sex = request.json['sex']
    result = "tuoi " + age
    return jsonify({"key" : result}) # tra ve json

    # return "ba"

if __name__ == "__main__":
    app.run(host='172.20.10.2', port=5000)
