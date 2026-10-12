# pip install transformers==4.39.3

# ============================================================
# Hallucination Detection with HHEM
# ============================================================
#
# In spite of the amazing power of LLMs, they still hallucinate.
# In some cases, where creativity is required, hallucinations are
# acceptable or even necessary, but in most enterprise RAG systems
# the response must be factually grounded.
#
# HHEM (Hughes Hallucination Evaluation Model) is a neural network
# specifically designed to measure hallucinations.
#
# It takes:
#   1. A source text (Premise)
#   2. A generated text (Hypothesis)
#
# and returns a score between 0 and 1 indicating how factually
# consistent the generated text is with the source.
# 
# Vectara's HHEM hallucination detection model is used.
#
# ------------------------------------------------------------

from transformers import pipeline, AutoTokenizer


# ------------------------------------------------------------
# Define Test Cases
# ------------------------------------------------------------
#
# Four article-summary pairs:
#
#   1. Correct summary
#   2. Completely hallucinated summary
#   3. Subtle factual errors
#   4. Fabricated information
#
# ------------------------------------------------------------

example_pairs = [

    # --------------------------------------------------------
    # Good summary
    # --------------------------------------------------------
    {
        "article":
        "The woman is playing mario cart while resting on the couch",

        "summary":
        "The woman is playing a game resting"
    },

    # --------------------------------------------------------
    # Completely incorrect summary
    # --------------------------------------------------------
    {
        "article":
        "A person on a horse jumps over a broken down airplane.",

        "summary":
        "A person is at a diner, ordering an omelette."
    },

    # --------------------------------------------------------
    # Wrong units (kg vs lbs, 1m vs 2m)
    # --------------------------------------------------------
    {
        "article":
        "Goldfish are being caught weighing up to 2kg and koi carp "
        "up to 8kg and one metre in length",

        "summary":
        "Koi carp can be as heavy as 2lbs and as long as two meters"
    },

    # --------------------------------------------------------
    # Fabricated estimated value
    # --------------------------------------------------------
    {
        "article":
        "The plants were found during the search of a warehouse "
        "near Ashbourne on Saturday morning. Police said they "
        "were in 'an elaborate grow house'. A man in his late "
        "40s was arrested at the scene.",

        "summary":
        "Police have arrested a man in his late 40s after "
        "cannabis plants worth an estimated £100,000 were "
        "found in a warehouse near Ashbourne."
    }
]


# ------------------------------------------------------------
# Build HHEM Prompt
# ------------------------------------------------------------
#
# HHEM expects input in the following format:
#
# Premise   -> Original article
# Hypothesis -> Generated summary
#
# ------------------------------------------------------------

prompt = """
<pad> Determine if the hypothesis is true given the premise?

Premise: {text1}

Hypothesis: {text2}
"""

input_pairs = [
    prompt.format(
        text1=pair["article"],
        text2=pair["summary"]
    )
    for pair in example_pairs
]


# ------------------------------------------------------------
# Load HHEM Model
# ------------------------------------------------------------

classifier = pipeline(
    "text-classification",
    model="vectara/hallucination_evaluation_model",
    tokenizer=AutoTokenizer.from_pretrained(
        "google/flan-t5-base"
    ),
    trust_remote_code=True
)


# ------------------------------------------------------------
# Run Evaluation
# ------------------------------------------------------------
#
# HHEM returns two labels:
#
#   consistent
#   inconsistent
#
# We keep only the "consistent" probability.
#
# ------------------------------------------------------------

full_scores = classifier(
    input_pairs,
    top_k=None
)

hhem_scores = [

    round(score_dict["score"], 4)

    for score_for_labels in full_scores

    for score_dict in score_for_labels

    if score_dict["label"] == "consistent"
]


# ------------------------------------------------------------
# Display Results
# ------------------------------------------------------------

print("\nHHEM Consistency Scores\n")

for i, score in enumerate(hhem_scores, start=1):

    print(f"Example {i}: {score}")


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------
#
# Score close to 1.0
#     Strong factual consistency
#
# Score close to 0.0
#     Likely hallucination
#
# Example output:
#
# Example 1 : 0.9182
# Example 2 : 0.0114
# Example 3 : 0.1654
# Example 4 : 0.0823
#
# Interpretation:
#
# Example 1
#     Correct summary
#
# Example 2
#     Completely hallucinated
#
# Example 3
#     Wrong measurements (kg/lbs, metres)
#
# Example 4
#     Fabricated monetary value
#
# ------------------------------------------------------------