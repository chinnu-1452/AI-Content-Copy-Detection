
def filter_matches(results, threshold=70):
    matches = []

    for item in results:
        if item["similarity"] >= threshold:
            matches.append(item)

    return matches
