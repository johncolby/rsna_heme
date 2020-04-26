import argparse
from flask_wtf import FlaskForm
import os
from urllib.parse import urljoin
from wtforms import BooleanField, FormField, IntegerField
from wtforms.validators import Optional

from rad_apps.appplugin import AppPlugin
from .process import HemeStudy

class Inputs(FlaskForm):
    ct = IntegerField('Axial CT', validators=[Optional()])
    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(Inputs, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs) 

class Options(FlaskForm):
    inputs = FormField(Inputs)
    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(Options, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs)

def wrapper_fun(app, form):
    HemeStudy(acc = form['acc'],
              download_url = app.config['AIR_URL'],
              cred_path = app.config['DOTENV_FILE'],
              process_url = urljoin(app.config['SEG_URL'], 'heme'),
              output_dir = os.path.join(app.config['OUTPUT_DIR_NODE'], 'heme')
              ).run()

app = AppPlugin(long_name = 'NCHCT hemorrhage',
                short_name = 'heme',
           form_opts = Options,
           wrapper_fun = wrapper_fun)
