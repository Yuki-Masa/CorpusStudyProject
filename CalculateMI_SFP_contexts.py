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
    'IR': 'Interrogative', 'V1': '冇','D1': '鬼','XN': "Foreign Noun", 'VU': 'Auxiliary Verb'
}

# Add Interrogative tag to mapping and define the list of interrogative words
POS_MAPPING['interrogative'] = 'Interrogative Pronoun/Adverb'
INTERROGATIVE_WORDS = {'乜', '邊個', '邊度', '邊', '點解', '點樣', '幾時', '幾', '幾多', '咩'}

# Define a set of filler words that should be ignored when looking for the core word
FILLER_WORDS = {
    '即係', '噉', '嘩', '唉', '哦', '誒', '唔', '都', '即', '係', '嘅', '㗎', '啦', '囖', '喇', '吖', '咩', '喎', '呢', '嘞',
    '咁', '噉樣', '好', '個', '嗰啲', '啲', '哩啲', '嗰個', '嗰陣', '即', '總之', '所以', '沖', '喂', '你', '佢', '我', '但係'
}

# 1. SFP Extraction with Context
def extract_sfps_with_context(corpus, n=2):
    """
    Extracts SFPs and their n-gram POS tag contexts, with a unique count
    based on a robust, hybrid deduplication principle.
    """
    print("Extracting SFPs and their syntactic contexts...")

    # Use a dictionary to store all occurrences, grouped by a normalized key
    normalized_patterns = defaultdict(list)
    # Use a set to track which sentences have been added to a pattern's list
    seen_utterance_and_pattern = set()

    all_utterances = list(corpus.utterances())

    for i, utt in enumerate(all_utterances):
        if i % 100 == 0 and i != 0:
            print(f"  - Processing utterance {i}...")

        tokens = utt.tokens

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

            # --- DEDUPLICATION LOGIC BASED ON USER PRINCIPLES ---
            dedupe_key_words = "<empty_context>"
            preceding_tokens = tokens[:sfp_index]

            # Iterate backward to find the first non-filler word, ignoring punctuation
            for j in range(len(preceding_tokens) - 1, -1, -1):
                token = preceding_tokens[j]

                # Ignore punctuation based on the user's principle
                if token.pos == 'w':
                    continue
                # Find the first non-filler word
                if token.word not in FILLER_WORDS:
                    dedupe_key_words = token.word
                    break

            # The final deduplication key for this specific occurrence
            dedupe_key = (unique_sfp, dedupe_key_words)

            # Use the full utterance text to ensure no duplicates in the output list
            full_utterance_text = "".join([token.word for token in tokens])

            # The key for tracking unique combinations of utterance and pattern
            unique_key_for_this_occurrence = (full_utterance_text, dedupe_key)

            if unique_key_for_this_occurrence not in seen_utterance_and_pattern:
                seen_utterance_and_pattern.add(unique_key_for_this_occurrence)

                # Store the current occurrence's context and utterance
                context_start = max(0, sfp_index - n)
                context_end = sfp_index
                context_ngram = tuple(pos_tags[context_start:context_end])

                occurrence_data = {
                    'context': context_ngram,
                    'utterance_text': full_utterance_text,
                    'utterance_id': str(i)
                }

                # Append this occurrence to the list associated with the normalized key
                normalized_patterns[dedupe_key].append(occurrence_data)

    print("Context extraction complete.")
    return normalized_patterns

#2.Mutual Information Calculation
def calculate_mutual_information(normalized_patterns):
    """
    Calculates the Mutual Information (MI) between SFPs and syntactic contexts
    based on the deduplicated and normalized data.
    """
    print("Calculating MI between SFPs and contexts...")
    mi_scores = defaultdict(dict)

    # Count the number of unique, deduplicated patterns
    sfp_frequency = defaultdict(int)
    for dedupe_key in normalized_patterns.keys():
        sfp = dedupe_key[0]
        sfp_frequency[sfp] += 1

    total_unique_count = sum(sfp_frequency.values())

    # Create frequency counts for the normalized POS tag patterns
    sfp_context_joint_freq = defaultdict(lambda: defaultdict(int))
    context_freq = defaultdict(int)

    for dedupe_key, occurrences in normalized_patterns.items():
        sfp = dedupe_key[0]
        context_ngram = occurrences[0]['context']

        # Apply the normalization rule for POS tags: ","+"X" and "X" are identical.
        normalized_context = context_ngram
        if len(context_ngram) > 1 and context_ngram[-2] == 'w':
            normalized_context = (context_ngram[-1],)

        sfp_context_joint_freq[sfp][normalized_context] += 1
        context_freq[normalized_context] += 1

    total_unique_context_count = sum(context_freq.values())

    for sfp, sfp_context_counts in sfp_context_joint_freq.items():
        sfp_freq = sfp_frequency[sfp]

        for context, joint_freq in sfp_context_counts.items():
            p_sfp_context = joint_freq / total_unique_count
            p_sfp = sfp_freq / total_unique_count
            p_context = context_freq[context] / total_unique_context_count

            if p_sfp > 0 and p_context > 0 and p_sfp_context > 0:
                mi = p_sfp_context * math.log2(p_sfp_context / (p_sfp * p_context))
                mi_scores[sfp][context] = mi

    print("MI calculation complete.")
    return mi_scores, sfp_frequency

#3. Utterance Extraction
def extract_representative_utterances(normalized_patterns, mi_scores, output_folder):
    """
    Extracts and saves up to 100 utterances for top POS tag patterns for each SFP.
    """
    print("Extracting representative utterances...")

    utterance_folder = os.path.join(output_folder, "utterance_lists")
    if not os.path.exists(utterance_folder):
        os.makedirs(utterance_folder)

    # Invert the normalized_patterns dictionary to a more useful structure for this function
    patt_to_utterances = defaultdict(list)
    for dedupe_key, occurrences in normalized_patterns.items():
        sfp = dedupe_key[0]
        pos_context = occurrences[0]['context']
        patt_to_utterances[(sfp, pos_context)].extend([o['utterance_text'] for o in occurrences])

    for sfp, scores in mi_scores.items():
        sorted_contexts = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sfp_char = sfp[0]
        output_path = os.path.join(utterance_folder, f"{sfp_char}_utterances.txt")

        with open(output_path, "w", encoding='utf-8') as f:
            f.write(f"Representative Utterances for SFP: {sfp_char} ({sfp[1]})\n\n")

            for context, mi_score in sorted_contexts:
                context_desc = " ".join([POS_MAPPING.get(tag, tag) for tag in context])
                f.write(f"--- Context: {context_desc} (MI Score: {mi_score:.4f}) ---\n")

                # Get all utterances for this specific context and SFP
                utterances_for_context = patt_to_utterances.get((sfp, context), [])

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
        if not font_file:
            print("Font path not set in environment variables. Using default.")
            font_file = 'C:/Windows/Fonts/NotoSansHK-VariableFont_wght.ttf' # Fallback path
        font_name = 'Noto Sans HK'

        try:
            fe = fm.FontEntry(fname=font_file, name=font_name)
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams['font.sans-serif'] = font_name
        except FileNotFoundError:
            print(f"Warning: Font file not found at {font_file}. Plots may not display correctly.")
            print("Please set the FONT_PATH environment variable to the correct path.")

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

    print("All plots saved to a single PDF file: {output_pdf_path}")

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

    normalized_patterns = extract_sfps_with_context(hcc_corpus, n=2)
    mi_scores, sfp_frequency = calculate_mutual_information(normalized_patterns)

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
    extract_representative_utterances(normalized_patterns, mi_scores, output_folder)

    # Generate and save plots to a single PDF
    output_pdf_path = os.path.join(output_folder, "sfp_mi_scores.pdf")
    with PdfPages(output_pdf_path) as pdf_pages:
        plot_mi_scores(mi_scores, sfp_frequency, pdf_pages)
    print(f"All plots saved to a single PDF file: {output_pdf_path}")

    print("\nSyntactic analysis and data saving complete.")
