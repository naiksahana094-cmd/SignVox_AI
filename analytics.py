
import plotly.express as px
import streamlit as st
import csv
import os
from collections import Counter
import pandas as pd
import plotly.express as px


DATASET_FILE = "dataset/isl_data.csv"


def analytics_dashboard():

    st.header("📊 AI Performance Analytics")

    st.write(
        "View dataset statistics and model performance "
        "for SignVox AI."
    )

    if not os.path.exists(DATASET_FILE):
        st.error("Dataset file not found!")
        return

    # Read dataset
    labels = []

    with open(DATASET_FILE, "r", newline="") as file:

        reader = csv.reader(file)
        next(reader)

        for row in reader:
            if len(row) == 127:
                labels.append(row[0])

    if not labels:
        st.warning("No dataset samples found.")
        return

    # Calculate statistics
    counts = Counter(labels)

    total_samples = len(labels)
    total_letters = len(counts)

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", total_samples)
    col2.metric("Letters Available", total_letters)
    col3.metric("Average Samples", round(total_samples / total_letters, 2))

    st.divider()

    # Dataset distribution
    st.subheader("📈 ISL Dataset Distribution")

    chart_data = pd.DataFrame({
        "Letter": sorted(counts.keys()),
        "Samples": [
            counts[letter]
            for letter in sorted(counts.keys())
        ]
    })

    fig = px.bar(
        chart_data,
        x="Letter",
        y="Samples",
        color="Samples",
        title="Number of Samples Per ISL Letter",
        text="Samples"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Dataset table
    st.subheader("📋 Sample Count Details")

    st.dataframe(
        chart_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Model information
    st.subheader("🤖 Model Information")

    st.write("Algorithm: Random Forest Classifier")
    st.write("Number of Estimators: 300")
    st.write("Classes: A to Z")
    st.write("Features: 126 hand landmark coordinates")

    st.info(
        "Note: Model accuracy should be evaluated using "
        "a separate test dataset. Dataset statistics alone "
        "do not measure live recognition accuracy."
    )
