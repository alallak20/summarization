import os
from frequency_based.summary import generate_summary_F_Based
from tf_idf.summary import generate_summary_manual_TF_IDF
from rouge import Rouge

# Path to the folder containing input documents
folder_path = 'C:/Users/bvvgg/Desktop/finalProject/input_folder'
# Initialize an empty string to store the concatenated text from input documents
document_text = ''

# Loop through each file in the input folder
for filename in os.listdir(folder_path):
    # Open each file and read its content
    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
        # Concatenate the content to the text string
        document_text += f.read()

# Generate the summary using the provided text and specify the number of sentences for the summary
f_based_summary = generate_summary_F_Based(document_text, 5)
tf_idf_summary = generate_summary_manual_TF_IDF(document_text, 5)


# List the files in the directory
reference_files = [file for file in os.listdir("references") if file.startswith('reference_')]

# Initialize the list to store reference summaries
reference_summaries = []

# Read each reference file and append its contents to reference_summaries
for file_name in reference_files:
    with open(os.path.join('references', file_name), 'r') as file:
        reference_summaries.append(file.read())



# Initialize Rouge object
rouge = Rouge()

# Calculate ROUGE scores
print('\n')
print("ROUGE scores for Frequency_based:")
for ref_summary in reference_summaries:
    scores = rouge.get_scores(f_based_summary, ref_summary)
    print('\n')
    for score in scores:
        for metric, values in score.items():
            print(f"'{metric}': {values}")

print("-------------------------------------------------------------")

print("\n", "ROUGE scores for TF IDF:")

for ref_summary in reference_summaries:
    scores = rouge.get_scores(tf_idf_summary, ref_summary)
    print('\n')
    for score in scores:
        for metric, values in score.items():
            print(f"'{metric}': {values}")



#Some details about the result.

#ROUGE-1 emphasizes individual word overlap.
#ROUGE-2 considers pairs of consecutive words.
#ROUGE-L focuses on the longest common subsequence.