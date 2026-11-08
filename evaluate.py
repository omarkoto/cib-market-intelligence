import csv

from rag import retrieve

K = 5
RESULTS_FILE = "results.csv"

TEST_QUESTIONS = [
    ("What were the company's quarterly profits and earnings per share?", "earn"),
    ("Which company reported a net loss for the year?", "earn"),
    ("What dividend did the company declare for shareholders?", "earn"),
    ("Which company agreed to acquire another company?", "acq"),
    ("Who made a takeover bid for the company's shares?", "acq"),
    ("What stake did the investor buy in the company?", "acq"),
    ("What did OPEC decide about oil output?", "crude"),
    ("What is happening to crude oil prices?", "crude"),
    ("How much oil is being produced by the Gulf countries?", "crude"),
    ("What is the trade deficit between the US and Japan?", "trade"),
    ("Which tariffs or import restrictions were announced?", "trade"),
    ("What are the trade talks about exports and imports?", "trade"),
    ("What is happening to the dollar against the yen?", "money-fx"),
    ("Did central banks intervene in the currency market?", "money-fx"),
    ("How is the exchange rate of the mark changing?", "money-fx"),
    ("Did the Federal Reserve change interest rates?", "interest"),
    ("What happened to the bank prime lending rate?", "interest"),
    ("Why are interest rates rising or falling?", "interest"),
    ("What is happening to tankers in the Gulf?", "ship"),
    ("Which shipping company has problems with its vessels?", "ship"),
    ("What is the news about port strikes and cargo ships?", "ship"),
    ("How much wheat is being exported?", "grain"),
    ("What is the news about the corn harvest and crop prices?", "grain"),
    ("Did the government buy grain from farmers?", "grain"),
]


OUT_OF_DOMAIN_QUESTIONS = [
    "What will the weather be like in Paris tomorrow?",
    "Who won the football world cup?",
    "How do I bake a chocolate cake?",
    "What is the price of Apple stock today?",
    "Can you recommend a good movie to watch?",
    "How do I learn to play the guitar?",
    "What is the capital city of Australia?",
    "How many calories are in a banana?",
]


def best_distance(question):
    chunks = retrieve(question, k=K, role="admin")
    return chunks[0]["distance"]


def calibrate_threshold():
    print()
    print("Best distance per question (in-domain):")
    in_domain = []
    for question, expected in TEST_QUESTIONS:
        distance = best_distance(question)
        in_domain.append(distance)
        print(round(distance, 3), "|", question)

    print()
    print("Best distance per question (out-of-domain):")
    out_of_domain = []
    for question in OUT_OF_DOMAIN_QUESTIONS:
        distance = best_distance(question)
        out_of_domain.append(distance)
        print(round(distance, 3), "|", question)

    print()
    print("Largest in-domain best distance:", round(max(in_domain), 3))
    print("Smallest out-of-domain best distance:", round(min(out_of_domain), 3))
    print("Choose MAX_DISTANCE between these two numbers.")


def precision_at_k(question, expected_category):
    chunks = retrieve(question, k=K, role="admin")
    correct = 0
    for chunk in chunks:
        if chunk["category"] == expected_category:
            correct = correct + 1
    return correct / K


def run_all():
    results = []
    for question, expected in TEST_QUESTIONS:
        precision = precision_at_k(question, expected)
        results.append((question, expected, precision))
    return results


def print_report(results):
    print("question | expected | precision@5")
    for question, expected, precision in results:
        print(question, "|", expected, "|", precision)

    print()
    print("Average precision@5 per category:")
    categories = []
    for question, expected, precision in results:
        if expected not in categories:
            categories.append(expected)
    for category in categories:
        total = 0
        count = 0
        for question, expected, precision in results:
            if expected == category:
                total = total + precision
                count = count + 1
        print(category, ":", round(total / count, 2))

    overall_total = 0
    for question, expected, precision in results:
        overall_total = overall_total + precision
    print()
    print("Overall average precision@5:", round(overall_total / len(results), 2))


# Saves the results to a CSV file.
# WHY: so we can open it in Excel or keep it to compare after changing the system.
def save_csv(results):
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "expected", "precision_at_5"])
        for question, expected, precision in results:
            writer.writerow([question, expected, precision])
    print("Saved results to", RESULTS_FILE)


if __name__ == "__main__":
    all_results = run_all()
    print_report(all_results)
    save_csv(all_results)
    calibrate_threshold()
