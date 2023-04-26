import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Load predictions and labels from the pickled files
with open("predictions_score_2023-04-14.pkl", "rb") as file:
    scores = pickle.load(file)

with open("predictions_label.pkl", "rb") as file:
    labels = pickle.load(file)

df = pd.read_csv('data_frame.csv')
true_labels = df['results'].values
# print(true_labels)
# Assume scores is a 2D array with shape (n_samples, 2) for binary classification
# Extract the probability estimates for the positive class
positive_class_scores = scores[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(true_labels, positive_class_scores)
roc_auc = auc(fpr, tpr)

# Create subplots
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        "Histogram of Prediction Scores",
        "Histogram of Prediction Labels",
        "ROC Curve " + str(roc_auc_score(true_labels, scores[:, 1]))
    )
)

# Add histogram of prediction scores
fig.add_trace(
    go.Histogram(x=positive_class_scores, nbinsx=20, name="Prediction Scores"),
    row=1, col=1
)

# Add histogram of prediction labels
fig.add_trace(
    go.Histogram(x=labels, nbinsx=2, name="Prediction Labels"),
    row=2, col=1
)

# Add ROC curve
fig.add_trace(
    go.Scatter(x=fpr, y=tpr, mode="lines",
               name="ROC curve (AUC = {:.2f})".format(roc_auc)),
    row=3, col=1
)

# Layout settings
fig.update_layout(height=1200, width=900)

# Render the dashboard
st.title("Dashboard")
st.plotly_chart(fig, use_container_width=True)
