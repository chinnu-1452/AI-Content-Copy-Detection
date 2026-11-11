
import pandas as pd


def generate_report(matches):

    if not matches:
        return "No matching content found."

    df = pd.DataFrame(matches)

    report = "# AI Content Copy Detection Report\n\n"

    report += f"Total Matches: {len(df)}\n\n"

    report += "## Matching Results\n\n"

    for index, row in df.iterrows():

        report += f"### Match {index + 1}\n\n"

        report += f"**Source:** {row['source_name']}\n\n"

        report += f"**Similarity:** {row['similarity']}%\n\n"

        report += "**Uploaded Content:**\n\n"
        report += f"{row['query_chunk']}\n\n"

        report += "**Reference Content:**\n\n"
        report += f"{row['source_chunk']}\n\n"

        report += "---\n\n"

    return report
