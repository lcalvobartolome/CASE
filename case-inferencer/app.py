"""Main application entry point
"""
from src.apis import api
from flask import Flask
from pyfiglet import figlet_format
from termcolor import cprint

# Create Flask app
app = Flask(__name__)
# Deactivate the default mask parameter.
app.config["RESTX_MASK_SWAGGER"] = False
api.init_app(app)

if __name__ == '__main__':
    cprint(figlet_format("CASE's Inferencer API",
           font='big'), 'red', attrs=['bold'])
    print('\n')
    app.run(host='0.0.0.0', port=90, debug=True)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=90)
