from flask import Flask, render_template, request, redirect
import mysql.connector
from license_plate_model import *

sql_root = 'root'  # enter sql user here
sql_password = 'hellohello'  # enter sql password here


app = Flask(__name__)


app.app_context().push()


mydb=mysql.connector.connect(host="localhost",user=sql_root, password = sql_password,auth_plugin='mysql_native_password')
my_cursor = mydb.cursor()
qu =('set global max_allowed_packet=67108864')
my_cursor.execute(qu)



@app.route('/',methods=['GET', 'POST'])
def hello_world():
    return render_template('index.html' )

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)  
        name = os.getcwd()+'\\'+f.filename
        result = prediction(name)
        if len(result)>1:
            return render_template("Acknowledgement.html",result=result)  
        else:
            return render_template('Failed.html', result=result)

if __name__ == "__main__":
    app.run(debug=True,port=8000)