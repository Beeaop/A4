from flask import Flask, render_template, request
import pickle
from config import Config
from guardrails.g1_privacy import redact

app = Flask(__name__)
app.config.from_object(Config)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        review = request.form['review']
        # G1: removing the personal data before the model, page or logs can see it
        redactions = {}
        if app.config["G1_ENABLED"]:
            review, redactions = redact(review)
        chosen_model = request.form['model']

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

        print("Recognised words:", [
            words[i] for i, count in enumerate(review_counts)
            if count > 0
        ])

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
