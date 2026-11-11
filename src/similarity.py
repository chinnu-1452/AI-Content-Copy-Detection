
import pandas as pd


def analyze_matches(results, threshold=50):

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    df = df[df["similarity"] >= threshold]

    df = df.sort_values(
        by="similarity",
        ascending=False
    )

    return df.reset_index(drop=True)


def get_summary(df):

    if df.empty:
        return {
            "total_matches": 0,
            "average_similarity": 0,
            "sources_found": 0
        }

    return {
        "total_matches": len(df),
        "average_similarity": round(
            df["similarity"].mean(), 2
        ),
        "sources_found": df["source_name"].nunique()
    }
