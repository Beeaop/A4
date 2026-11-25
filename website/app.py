from flask import Flask, render_template, request, abort
import pickle
from config import Config
from guardrails.g1_privacy import redact
from guardrails.g2_validation import validate


app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # Read the form outside the try, so a too-large request still gives a 413 page
        form = request.form

        # G2: checks the input first and refuses anything bad.
        # G2 off = the old A3 behaviour (accepts anything).
        if app.config["G2_ENABLED"]:
            try:
                review, chosen_model, error = validate(form)
            except Exception:
                abort(400)
            if error:
                return render_template('predict.html', result=None, error=error), 400
        else:
            review = form['review']
            chosen_model = form['model']

        # G1: remove personal data.
        # If redaction crashes, it refuses the request
        # (fail closed) so any unredacted text never gets through
        redactions = {}
        if app.config["G1_ENABLED"]:
            try:
                review, redactions = redact(review)
            except Exception:
                abort(500)

        # Choose the model
        if chosen_model == 'decision_tree':
            model_file = 'csc1052-decision-tree-model.pkl'
        else:
            model_file = 'csc1052-nb1000-model.pkl'

        # Load the selected model
        with open(model_file, 'rb') as file:
            model = pickle.load(file)

        # Load the vocabulary
        with open('csc1052-nb1000-features.pkl', 'rb') as file:
            features = pickle.load(file)

        # Convert the review into numerical features
        review_features = features.transform([review])

        # Predict sentiment
        prediction = str(model.predict(review_features)[0])

        if prediction.lower() in ('1', 'positive'):
            sentiment = 'Positive'
        elif prediction.lower() in ('0', 'negative'):
            sentiment = 'Negative'
        else:
            sentiment = prediction

        # Calculate prediction confidence
        probabilities = model.predict_proba(review_features)[0]

        confidence = {}

        for label, probability in zip(model.classes_, probabilities):

            if str(label).lower() in ('1', 'positive'):
                label_name = 'Positive'
            elif str(label).lower() in ('0', 'negative'):
                label_name = 'Negative'
            else:
                label_name = str(label)

            confidence[label_name] = round(probability * 100, 2)

        # Get the words recognised by the model
        words = features.get_feature_names_out()
        review_counts = review_features.toarray()[0]

        contributions = []

        # Find the positive and negative class positions
        positive_index = next(
            (i for i, label in enumerate(model.classes_)
             if str(label).lower() in ('1', 'positive')),
            None
        )

        if positive_index is None:
            raise ValueError("Could not identify the positive class.")

        negative_index = 1 - positive_index

        # NAIVE BAYES
        if chosen_model == 'naive_bayes':

            for i, count in enumerate(review_counts):

                # Only use words present in the review
                if count > 0:

                    score = count * (
                        model.feature_log_prob_[positive_index][i]
                        - model.feature_log_prob_[negative_index][i]
                    )

                    contributions.append({
                        'word': words[i],
                        'direction': 'Positive' if score > 0 else 'Negative',
                        'score': float(score)
                    })

        # DECISION TREE
        else:

            tree = model.tree_
            node_id = 0

            # Follow the path taken by the review
            while tree.feature[node_id] >= 0:

                feature_index = tree.feature[node_id]
                word = words[feature_index]
                count = review_counts[feature_index]

                # Positive probability before the decision
                parent_values = tree.value[node_id][0]

                parent_positive = (
                    parent_values[positive_index] / parent_values.sum()
                )

                # Follow the correct branch
                if count <= tree.threshold[node_id]:
                    child_id = tree.children_left[node_id]
                else:
                    child_id = tree.children_right[node_id]

                # Positive probability after the decision
                child_values = tree.value[child_id][0]

                child_positive = (
                    child_values[positive_index] / child_values.sum()
                )

                # Calculate contribution
                score = float(child_positive - parent_positive)

                # Only include words present in the review
                if count > 0:

                    contributions.append({
                        'word': word,
                        'direction': 'Positive' if score > 0 else 'Negative',
                        'score': score
                    })

                # Move to the next node
                node_id = child_id

        # Combine contributions for repeated words
        combined = {}

        for item in contributions:

            word = item['word']

            if word not in combined:
                combined[word] = 0

            combined[word] += item['score']

        # Keep the five strongest contributing words
        top_words = []

        for word, score in combined.items():

            if abs(score) > 0:

                top_words.append({
                    'word': word,
                    'direction': 'Positive' if score > 0 else 'Negative',
                    'score': round(score, 3)
                })

        top_words = sorted(
            top_words,
            key=lambda item: abs(item['score']),
            reverse=True
        )[:5]

        # G1 ON: logs basic info about the request, not what the user actually wrote
        # G1 OFF: logs words like the og A3 app
        if app.config["G1_ENABLED"]:
            print({"model": chosen_model, "prediction": sentiment, "review_chars": len(review), "redactions": redactions})
        else:
            print("Recognised words:", [w for w, c in zip(words, review_counts) if c > 0])
            print("Top contributing words:", top_words)

        # Display results
        return render_template(
            'predict.html',
            result={
                'text': review,
                'prediction': sentiment,
                'confidence': confidence,
                'model': chosen_model,
                'top_words': top_words
            }
        )

    return render_template('predict.html', result=None)
# Custom error pages: no stack traces, nothing the user typed is shown back.
ERRORS = {
    400: ("Bad request", "That request could not be processed."),
    404: ("Page not found", "That page does not exist."),
    413: ("Review too large", "That request was too large for the app to accept."),
    500: ("Something went wrong", "Your review was not processed or stored."),
}

@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
@app.errorhandler(500)

def error_page(error):
    title, message = ERRORS[error.code]
    return render_template("error.html", code=error.code, title=title, message=message), error.code

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
