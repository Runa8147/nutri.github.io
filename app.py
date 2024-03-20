from flask import Flask, request, jsonify, render_template
import spacy

app = Flask(__name__)

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Dummy data for personalized nutrition recommendations
nutrition_recommendations = {
    "weight_loss": {
        "breakfast": "Oatmeal with berries and a glass of skim milk",
        "lunch": "Grilled chicken salad with mixed greens and vinaigrette dressing",
        "dinner": "Baked salmon with quinoa and steamed vegetables"
    },
    "muscle_gain": {
        "breakfast": "Egg white omelette with spinach and whole grain toast",
        "lunch": "Lean beef stir-fry with brown rice and broccoli",
        "dinner": "Grilled turkey breast with sweet potato and asparagus"
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    response = get_bot_response(message)
    return jsonify({"response": response})

def get_bot_response(message):
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return f"Hello {ent.text}, how can I assist you today?"
    return "I'm sorry, I didn't understand. Can you please rephrase?"

if __name__ == '__main__':
    app.run(debug=True)
