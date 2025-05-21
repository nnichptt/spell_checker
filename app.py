from flask import Flask, request, render_template, jsonify
from spell_model import SpellChecker

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("gui.html",result=None)

@app.route('/correct', methods=['POST'])
def correct():
    input_text = request.form.get('input_text').strip().lower()
    
    # Process the input_text (example: reversing the text)
    if not input_text:
        # TODO: Warn User
        return

    spc = SpellChecker()
    suggestion = spc.brownSuggest(word=input_text)
    result = f"Suggestion:{suggestion}"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)