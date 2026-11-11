
import streamlit as st
import pandas as pd
from pathlib import Path
from uuid import uuid4

from src.extract import extract_text
from src.preprocess import clean_text
from src.chunking import split_into_chunks
from src.retrieval import retrieve_sources
from src.similarity import analyze_matches, get_summary
from src.report import generate_report


st.set_page_config(
    page_title="AI Content Copy Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Based Content Copy Detection System")

st.write(
    "Upload a suspicious document and reference documents "
    "to find semantically similar passages."
)

REFERENCE_DIR = Path("data/reference_docs")
UPLOAD_DIR = Path("data/uploads")

REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, folder):

    extension = Path(uploaded_file.name).suffix.lower()

    file_path = folder / f"{uuid4().hex}{extension}"

    file_path.write_bytes(uploaded_file.getvalue())

    return file_path


st.sidebar.header("Detection Settings")

threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0,
    max_value=100,
    value=50
)

top_k = st.sidebar.slider(
    "Top source passages",
    min_value=1,
    max_value=10,
    value=5
)

st.header("1. Upload Reference Documents")

reference_files = st.file_uploader(
    "Upload original/reference documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="references"
)

st.header("2. Upload Suspicious Document")

suspicious_file = st.file_uploader(
    "Upload document to analyze",
    type=["pdf", "docx", "txt"],
    key="suspicious"
)

if st.button("Analyze Document", type="primary"):

    if not reference_files:
        st.error("Please upload at least one reference document.")

    elif not suspicious_file:
        st.error("Please upload a suspicious document.")

    else:

        try:
            reference_chunks = []

            with st.spinner("Processing reference documents..."):

                for uploaded_file in reference_files:

                    file_path = save_uploaded_file(
                        uploaded_file,
                        REFERENCE_DIR
                    )

                    text = extract_text(file_path)
                    text = clean_text(text)

                    chunks = split_into_chunks(text)

                    for chunk in chunks:

                        reference_chunks.append({
                            "text": chunk,
                            "source": uploaded_file.name
                        })

            suspicious_path = save_uploaded_file(
                suspicious_file,
                UPLOAD_DIR
            )

            suspicious_text = extract_text(suspicious_path)
            suspicious_text = clean_text(suspicious_text)

            query_chunks = split_into_chunks(suspicious_text)

            if not reference_chunks:
                st.error("No usable text found in reference documents.")

            elif not query_chunks:
                st.error("No usable text found in suspicious document.")

            else:

                with st.spinner("Generating embeddings and searching sources..."):

                    results = retrieve_sources(
                        query_chunks,
                        reference_chunks,
                        top_k=top_k
                    )

                    df = analyze_matches(
                        results,
                        threshold=threshold
                    )

                summary = get_summary(df)

                st.success("Document analysis completed!")

                st.header("3. Analysis Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Matching passages",
                    summary["total_matches"]
                )

                col2.metric(
                    "Average similarity",
                    f'{summary["average_similarity"]}%'
                )

                col3.metric(
                    "Possible source documents",
                    summary["sources_found"]
                )

                st.header("4. Matching Source Documents")

                if df.empty:

                    st.info(
                        "No passages crossed the selected similarity threshold."
                    )

                else:

                    st.dataframe(
                        df[
                            [
                                "source_name",
                                "similarity",
                                "query_chunk",
                                "source_chunk"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                    st.header("5. Detailed Match Analysis")

                    for index, row in df.iterrows():

                        with st.expander(
                            f"Match {index + 1} - "
                            f"{row['source_name']} "
                            f"({row['similarity']}%)"
                        ):

                            st.write("**Suspicious passage:**")
                            st.write(row["query_chunk"])

                            st.write("**Possible source passage:**")
                            st.write(row["source_chunk"])

                st.header("6. Download Report")

                report = generate_report(df)

                st.download_button(
                    label="Download Analysis Report",
                    data=report,
                    file_name="content_detection_report.md",
                    mime="text/markdown"
                )

                csv_data = df.to_csv(index=False)

                st.download_button(
                    label="Download CSV Results",
                    data=csv_data,
                    file_name="matching_results.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Similarity scores indicate semantic resemblance. "
                    "They do not establish plagiarism or AI authorship."
                )

        except Exception as error:

            st.error(f"An error occurred: {error}")
            st.exception(error)


st.divider()

st.caption(
    "Built with Python, NLP, Sentence Transformers, FAISS and Streamlit."
)
