
def generate_report(df):

    if df.empty:
        return "No matching passages found."

    report = "# AI Content Copy Detection Report\n\n"

    report += f"Total matches: {len(df)}\n\n"

    for index, row in df.iterrows():

        report += f"## Match {index + 1}\n"

        report += f"Source: {row['source_name']}\n\n"

        report += f"Similarity: {row['similarity']}%\n\n"

        report += f"Suspicious passage:\n{row['query_chunk']}\n\n"

        report += f"Possible source passage:\n{row['source_chunk']}\n\n"

        report += "---\n"

    report += (
        "\nNote: Similarity scores indicate semantic resemblance, "
        "not proof of plagiarism or AI authorship.\n"
    )

    return report
