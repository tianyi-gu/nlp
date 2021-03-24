from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo

class inputData(FlaskForm):
	firstlink = StringField('Comparison Link', validators = [DataRequired()])
	secondlink = StringField('Self Link', validators = [DataRequired()])
	submit = SubmitField('Submit')
