from flask import *
import pandas as pd
from res_rec import topres, resfoodmatch, search


app = Flask(__name__)



@app.route("/index")
def rest1():
	return render_template("index.html")

@app.route("/res2",methods = ['POST', 'GET'])
def recomm():
	recomm.option=''
	if request.method == 'POST':
		recomm.option = request.form['options']
	return render_template("res2.html", city = recomm.option)

@app.route("/search",methods = ['POST', 'GET'])
def khoj():
	if request.method == 'POST':
		keyword = request.form['research']
		if keyword=='':
			m ='Empty Field!'
			return render_template("res2.html",city=recomm.option,serror = m)
		else:
			try:
				df=search(recomm.option,keyword)
				return render_template("search.html", column_names=df.columns.values, row_data=list(df.values.tolist()),link_column="image", zip=zip)
			except:
				m ='No Restaurant Found!'
				return render_template("res2.html",city=recomm.option,serror = m)


@app.route("/topres",methods = ['POST', 'GET'])
def tresd():
	kitne=''
	if request.method == 'POST':
		kitne =request.form['toprec']
	if kitne=='':
		message = 'Field required!'
		return render_template("res2.html",city=recomm.option,khaali = message)
	else:
		df = topres(recomm.option,int(kitne))
		return render_template("res3.html", column_names=df.columns.values, row_data=list(df.values.tolist()),link_column="Res_url", zip=zip)

@app.route("/matchres",methods = ['POST', 'GET'])
def mresd():
	if request.method == 'POST':
		#kitne = int(request.form['toprec'])
		restomatch = request.form['resname']
	try:
		df = resfoodmatch(recomm.option,restomatch,20)
		return render_template("res3.html", column_names=df.columns.values, row_data=list(df.values.tolist()),link_column="Res_url", zip=zip)
	except:
		if restomatch=='':
			message='Field Required!'
			return render_template("res2.html",city = recomm.option,error=message)
		else:
			message='Restaurant Name not found!'
			return render_template("res2.html",city = recomm.option,error=message)


if __name__ == "__main__":
    app.run(debug=True)

#topres('delhi')