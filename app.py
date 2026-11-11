
import streamlit as st
import pandas as pd
from pathlib import Path

from src.extract import extract_text
from src.preprocess import clean_text
from src.chunking import split_into_chunks
from src.retrieval import (
    build_reference_database,
    reference_database_exists,
    retrieve_sources
)
from src.similarity import filter_matches
from src.report import generate_report


REFERENCE_DIR = Path("data/reference_docs")
UPLOAD_DIR = Path("data/uploads")

REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="AI Content Copy Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("AI-Based Content Copy Detection System")

st.write(
    "Detect semantically similar content using AI embeddings and FAISS."
)

# SECTION 1: BUILD REFERENCE DATABASE

st.header("1. Build Reference Database")

reference_files = st.file_uploader(
    "Upload reference documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="reference_files"
)

if st.button("Build / Update Reference Database"):

    if not reference_files:
        st.warning("Please upload reference documents.")

    else:
        all_reference_chunks = []

        with st.spinner("Building reference database..."):

            for uploaded_file in reference_files:

                file_path = REFERENCE_DIR / uploaded_file.name

                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                text = extract_text(str(file_path))
                cleaned_text = clean_text(text)

                chunks = split_into_chunks(cleaned_text)

                for chunk in chunks:
                    all_reference_chunks.append({
                        "text": chunk,
                        "source": uploaded_file.name
                    })

            if all_reference_chunks:

                count = build_reference_database(
                    all_reference_chunks
                )

                st.success(
                    f"Database created successfully! {count} chunks saved."
                )

            else:
                st.error("No text found in uploaded documents.")


# SECTION 2: DATABASE STATUS

st.header("2. Database Status")

if reference_database_exists():
    st.success("Reference database is ready.")
else:
    st.warning("Reference database not found. Build it first.")


# SECTION 3: ANALYZE DOCUMENT

st.header("3. Analyze Document")

suspicious_file = st.file_uploader(
    "Upload document to check",
    type=["pdf", "docx", "txt"],
    key="suspicious_file"
)

threshold = st.slider(
    "Similarity threshold (%)",
    min_value=0,
    max_value=100,
    value=70
)

top_k = st.slider(
    "Number of matches",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Analyze Document"):

    if suspicious_file is None:
        st.warning("Please upload a document.")

    elif not reference_database_exists():
        st.error("Build the reference database first.")

    else:

        with st.spinner("Analyzing document..."):

            file_path = UPLOAD_DIR / suspicious_file.name

            with open(file_path, "wb") as file:
                file.write(suspicious_file.getbuffer())

            text = extract_text(str(file_path))
            cleaned_text = clean_text(text)

            query_chunks = split_into_chunks(cleaned_text)

            results = retrieve_sources(
                query_chunks,
                top_k=top_k
            )

            matches = filter_matches(results, threshold)

        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Matching Passages", len(matches))

        average_similarity = (
            sum(item["similarity"] for item in matches) / len(matches)
            if matches else 0
        )

        col2.metric(
            "Average Similarity",
            f"{average_similarity:.2f}%"
        )

        col3.metric(
            "Possible Source Documents",
            len(set(item["source_name"] for item in matches))
        )

        if matches:

            st.subheader("Matched Content")

            df = pd.DataFrame(matches)
            st.dataframe(df, use_container_width=True)

            for index, match in enumerate(matches):

                with st.expander(
                    f"Match {index + 1} - {match['similarity']}%"
                ):

                    st.write("**Uploaded Document:**")
                    st.write(match["query_chunk"])

                    st.write("**Reference Document:**")
                    st.write(match["source_chunk"])

                    st.write("**Source:**")
                    st.write(match["source_name"])

            report = generate_report(matches)

            st.download_button(
                "Download Report",
                data=report,
                file_name="content_copy_report.md",
                mime="text/markdown"
            )

            csv_data = df.to_csv(index=False)

            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="content_copy_results.csv",
                mime="text/csv"
            )

        else:
            st.success("No matches found above the selected threshold.")

        st.info(
            "Similarity indicates semantic resemblance, not proof of copying."
        )
