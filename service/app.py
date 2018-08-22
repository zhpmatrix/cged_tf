from flask import Flask,render_template
from flask_wtf import Form
from flask_bootstrap import Bootstrap
from wtforms import StringField,SubmitField,TextAreaField
from wtforms.validators import Required,Length

from test import *

class NameForm(Form):
    name = TextAreaField('Input',validators=[Required()],render_kw={"placeholder":"此处输入文本"})
    submit = SubmitField('Submit')

def inference(data):
    """
        data: 今天是七夕情人节呀，单身狗们快乐！
        seq = ['今','天','是','七','夕','情','人','节','呀','单','身','狗','们','快','乐']
        tag = ['O','S','S','S','M','W','W','W','S','S','R','R','O','M','O']
    """
    rst = []

    # Convert data to list
    seq = list(data)
    
	# Predict
    tag = predict(data)

    # Get last tag of tag list
    tag = [tag[i][-1] for i in range(len(tag))]  
	
	# Log
    print('Seq: ',seq)
    print('Tag: ', tag)
	
	# Merge data
    for i in range(len(seq)):
        elem = {}
        elem['seq'] = seq[i]
        elem['tag'] = tag[i]
        rst.append(elem)
    return rst

app = Flask(__name__)
app.config['SECRET_KEY'] = '666'
bootstrap = Bootstrap(app)

@app.route('/',methods=['GET','POST'])
def index():
    name = None
    nameForm = NameForm()
    rst = []

    if nameForm.validate_on_submit():
        name = nameForm.name.data
        nameForm.name.data = ''
        rst = inference(name)
    return render_template('index.html',form=nameForm,data=rst)

if __name__ == '__main__':
    app.run('172.18.147.43', debug=True)

