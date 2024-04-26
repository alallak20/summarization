import os
from TF_IDF.summary import generate_summary_manual_TF_IDF
from Frequency_based.summary import generate_summary_F_Based


# Path to the folder containing input documents
folder_path = 'C:/Users/bvvgg/Desktop/finalProject/input_folder'
# Initialize an empty string to store the concatenated text from input documents
text = ''

# Loop through each file in the input folder
for filename in os.listdir(folder_path):
    # Open each file and read its content
    with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
        # Concatenate the content to the text string
        text += f.read()

#TF IDF

# Generate the summary using the provided text and specify the number of sentences for the summary
tf_idf_summary = generate_summary_manual_TF_IDF(text, 10)
# Split the summary into individual sentences
summary_1_sentences = tf_idf_summary.split('. ')
# Join the sentences with a newline character to format the summary
formatted_summary_tf_idf = '.\n'.join(summary_1_sentences)

# Write the formatted summary to a text file
with open('results/tf_idf_summary.txt', 'w') as f:
    f.write(formatted_summary_tf_idf)
#---------------------------------------------------------------------------
#Frequency_based
F_based_summary = generate_summary_F_Based(text, 10)
summary_2_sentences = F_based_summary.split('. ')
formatted_summary_f_based = '.\n'.join(summary_2_sentences)

with open('results/f_based_summary.txt', 'w') as f:
    f.write(formatted_summary_f_based)

# Print a confirmation message
print("\nIt works!")