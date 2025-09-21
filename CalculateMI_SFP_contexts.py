import sys
import pycantonese
import pandas as pd
from collections import defaultdict
import math
import itertools
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import os
from datetime import datetime
import re
from dotenv import load_dotenv

"""
to calculate Mutual Information of the SFPs (sentence-final particles) and the
sequence patterns of POS pos tags

Version Memo:

1. No overcounting,
2. Using MI (not PMI),
3. Interrogative tag added

"""

load_dotenv()
FONT_PATH = os.getenv('FONT_PATH')

# POS tag to description mapping
POS_MAPPING = {
    'Ag': 'Adjective Morpheme', 'A': 'Adjective', 'AD': 'Adjective as Adverbial',
    'AN': 'Adjective with Nominal Function', 'BG': 'Non-predicate Adjective Morpheme',
    'B': 'Non-predicate Adjective', 'C': 'Conjunction', 'DG': 'Adverb Morpheme',
    'D': 'Adverb', 'E': 'Interjection', 'F': 'Directional Locality', 'G': 'Morpheme',
    'H': 'Prefix', 'I': 'Idiom', 'J': 'Abbreviation', 'K': 'Suffix', 'L': 'Fixed Expression',
    'MG': 'Numeric Morpheme', 'M': 'Numeral', 'NG': 'Noun Morpheme', 'N': 'Common Noun',
    'NR': 'Personal Name', 'NS': 'Place Name', 'NT': 'Organisation Name',
    'NX': 'Nominal Character String', 'NZ': 'Other Proper Noun', 'O': 'Onomatopoeia',
    'P': 'Preposition', 'QG': 'Classifier Morpheme', 'Q': 'Classifier',
    'RG': 'Pronoun Morpheme', 'R': 'Pronoun', 'S': 'Space Word', 'TG': 'Time Word Morpheme',
    'T': 'Time Word', 'UG': 'Auxiliary Morpheme', 'U': 'Auxiliary',
    'VG': 'Verb Morpheme', 'V': 'Verb', 'VD': 'Verb as Adverbial',
    'VN': 'Verb with Nominal Function', 'W': 'Punctuation', 'X': 'Unclassified Item',
    'YG': 'Modal Particle Morpheme', 'Y': 'Modal Particle', 'Z': 'Descriptive',
    'IR': 'interrogative'
}

# Add Interrogative tag to mapping and define the list of interrogative words
POS_MAPPING['interrogative'] = 'Interrogative Pronoun/Adverb'
INTERROGATIVE_WORDS = {'乜', '邊個', '邊度', '邊', '點解', '點樣', '幾時', '幾', '幾多', '咩'}


# 1. SFP Extraction with Context
def extract_sfps_with_context(corpus, n=2):
    """
    Extracts SFPs and their n-gram POS tag contexts, with a unique count
    based on the last three characters of the preceding phrase.
    """
    print("Extracting SFPs and their syntactic contexts...")

    # Use a set to track canonical contexts to avoid overcounting
    canonical_contexts_seen = set()
    sfp_context_data = defaultdict(list)
    sfp_frequency = defaultdict(int)
    all_utterances = list(corpus.utterances())

    for i, utt in enumerate(all_utterances):
        if i % 100 == 0 and i != 0:
            print(f"  - Processing utterance {i}...")

        tokens = utt.tokens
        #pos_tags = [token.pos for token in tokens]
        # Create a new list of POS tags, checking for interrogative words
        pos_tags = []
        for token in tokens:
            if token.word in INTERROGATIVE_WORDS:
                pos_tags.append('IR') #interrogative
            else:
                pos_tags.append(token.pos)
        sfp_indices = [j for j, pos in enumerate(pos_tags) if pos == 'Y']

        for sfp_index in sfp_indices:
            sfp_token = tokens[sfp_index]
            unique_sfp = (sfp_token.word, sfp_token.jyutping)

            # Get preceding words as a string, ignoring punctuation
            preceding_words = "".join([token.word for token in tokens[:sfp_index] if token.pos != 'w'])

            # Determine canonical context based on last 3 characters
            canonical_context = preceding_words[-1:]

            # Combine SFP and canonical context for a unique key
            unique_key = (unique_sfp, canonical_context)

            if unique_key not in canonical_contexts_seen:
                canonical_contexts_seen.add(unique_key)
                sfp_frequency[unique_sfp] += 1

                context_start = max(0, sfp_index - n)
                context_end = sfp_index
                context_ngram = tuple(pos_tags[context_start:context_end])

                # Store the canonical context and its first seen occurrence
                sfp_context_data[unique_sfp].append({
                    'context': context_ngram,
                    'utterance_text': "".join([token.word for token in tokens]),
                    'utterance_id': str(i)
                })

    print("Context extraction complete.")
    return sfp_context_data, sfp_frequency

#2.Mutual Information Calculation
def calculate_mutual_information(sfp_context_data):
    """
    Calculates the Mutual Information (MI) between SFPs and syntactic contexts
    based on the deduplicated data.
    """
    print("Calculating MI between SFPs and contexts...")
    mi_scores = defaultdict(dict)

    # Use the number of unique contexts as total count
    total_sfp_count = sum(len(contexts) for contexts in sfp_context_data.values())

    # Create frequency counts for the canonical POS tag patterns
    sfp_context_freq = defaultdict(lambda: defaultdict(int))
    context_freq = defaultdict(int)

    for sfp, contexts in sfp_context_data.items():
        for item in contexts:
            context_ngram = item['context']
            sfp_context_freq[sfp][context_ngram] += 1
            context_freq[context_ngram] += 1

    total_context_count = sum(context_freq.values())

    for sfp, sfp_context_counts in sfp_context_freq.items():
        sfp_freq = len(sfp_context_data[sfp])

        for context, joint_freq in sfp_context_counts.items():
            p_sfp_context = joint_freq / total_sfp_count
            p_sfp = sfp_freq / total_sfp_count
            p_context = context_freq[context] / total_context_count

            if p_sfp > 0 and p_context > 0:
                mi = p_sfp_context * math.log2(p_sfp_context / (p_sfp * p_context))
                mi_scores[sfp][context] = mi

    print("MI calculation complete.")
    return mi_scores

#3. Utterance Extraction
def extract_representative_utterances(sfp_context_data, pmi_scores, output_folder):
    """
    Extracts and saves utterances for top POS tag patterns for each SFP.
    """
    print("Extracting representative utterances...")

    utterance_folder = os.path.join(output_folder, "utterance_lists")
    if not os.path.exists(utterance_folder):
        os.makedirs(utterance_folder)

    for sfp, scores in pmi_scores.items():
        # Sort contexts by PMI score in descending order
        sorted_contexts = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sfp_char = sfp[0]

        output_path = os.path.join(utterance_folder, f"{sfp_char}_utterances.txt")

        with open(output_path, "w", encoding='utf-8') as f:
            f.write(f"Representative Utterances for SFP: {sfp_char} ({sfp[1]})\n\n")

            for context, pmi_score in sorted_contexts:
                context_desc = " ".join([POS_MAPPING.get(tag, tag) for tag in context])
                f.write(f"--- Context: {context_desc} (PMI Score: {pmi_score:.4f}) ---\n")

                # Get utterances for this specific context and SFP
                utterances_for_context = [
                    item['utterance_text'] for item in sfp_context_data[sfp]
                    if item['context'] == context
                ]

                num_to_extract = min(100, len(utterances_for_context))

                for utt in utterances_for_context[:num_to_extract]:
                    f.write(f"- {utt}\n")
                f.write("\n")

            print(f"Saved utterances for SFP '{sfp_char}' to {output_path}")

    print("Utterance extraction complete.")

# --- 4. Plotting the Results ---
def plot_mi_scores(mi_scores, sfp_frequency, pdf_pages, n=2):
    """
    Generates and saves bar graphs for the top 10 MI scores for each SFP
    into a single PDF file, sorted by SFP frequency.
    """
    print("Generating plots for MI scores...")

    # Sort SFPs by frequency in descending order
    sorted_sfps = sorted(sfp_frequency.items(), key=lambda item: item[1], reverse=True)

    for sfp, freq in sorted_sfps:
        scores = mi_scores.get(sfp, {})
        sorted_contexts = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10]

        if not sorted_contexts:
            continue

        contexts_desc = [" ".join(POS_MAPPING.get(tag, tag) for tag in c) for c, score in sorted_contexts]
        mi_values = [score for c, score in sorted_contexts]

        # Define the font file path and its name
        font_file = FONT_PATH
        font_name = 'Noto Sans HK'
        # Create a font entry and insert it into Matplotlib's font list
        fe = fm.FontEntry(fname=font_file, name=font_name)
        fm.fontManager.ttflist.insert(0, fe)
        # Rebuild the font cache (optional but recommended)
        #fm._rebuild()
        plt.rcParams['font.sans-serif'] = font_name

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(contexts_desc, mi_values, color='skyblue')
        ax.set_xlabel("Mutual Information Score")
        ax.set_ylabel("Syntactic Context (POS tags)")
        ax.set_title(f"Top {n}-gram Contexts for SFP: {sfp[0]} ({sfp[1]}) - Freq: {freq}")
        ax.invert_yaxis()

        pos_tags_in_plot = set(tag for c, score in sorted_contexts for tag in c)
        description_text = "\n".join([f"{tag}: {POS_MAPPING.get(tag, 'Undefined')}" for tag in sorted(list(pos_tags_in_plot))])

        plt.text(1.02, 1.0, description_text.strip(), transform=ax.transAxes,
                 fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)

    print("All plots saved to PDF.")

#5. Main
if __name__ == '__main__':
    print("Starting analysis...")

    # Create dated output folder
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_folder = f"sfp_analysis_results_{current_date}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    hcc_corpus = pycantonese.hkcancor()
    print("HKCanCor corpus loaded successfully.")

    sfp_context_data, sfp_frequency = extract_sfps_with_context(hcc_corpus, n=2)
    mi_scores = calculate_mutual_information(sfp_context_data)

    print("\nSaving results...")

    # Save SFP frequency table
    sfp_freq_df = pd.DataFrame(sfp_frequency.items(), columns=['SFP', 'Frequency'])
    sfp_freq_df['SFP_Char'] = sfp_freq_df['SFP'].apply(lambda x: x[0])
    sfp_freq_df['SFP_Jyutping'] = sfp_freq_df['SFP'].apply(lambda x: x[1])
    sfp_freq_df = sfp_freq_df.sort_values(by='Frequency', ascending=False).drop(columns=['SFP'])
    sfp_freq_df.to_csv(os.path.join(output_folder, "sfp_frequency.csv"), index=False)
    print(f"SFP frequency table saved to {output_folder}/sfp_frequency.csv")

    # Create and save the Mutual Information table
    data_for_df = []
    for sfp, scores in mi_scores.items():
        for context, mi_score in scores.items():
            data_for_df.append({
                'sfp_character': sfp[0],
                'sfp_jyutping': sfp[1],
                'context': " ".join(context),
                'mi_score': mi_score
            })
    df = pd.DataFrame(data_for_df)
    df.to_csv(os.path.join(output_folder, "sfp_syntactic_correlations.csv"), index=False)
    print(f"Syntactic correlations table saved to {output_folder}/sfp_syntactic_correlations.csv")

    # Extract and save representative utterances
    extract_representative_utterances(sfp_context_data, mi_scores, output_folder)

    # Generate and save plots to a single PDF
    output_pdf_path = os.path.join(output_folder, "sfp_mi_scores.pdf")
    with PdfPages(output_pdf_path) as pdf_pages:
        plot_mi_scores(mi_scores, sfp_frequency, pdf_pages)
    print(f"All plots saved to a single PDF file: {output_pdf_path}")

    print("\nSyntactic analysis and data saving complete.")
