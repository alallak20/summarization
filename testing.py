import os
from Frequency_based.summary import generate_summary_F_Based
from TF_IDF.summary import generate_summary_manual_TF_IDF
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

reference_summaries = [
    "Weather is portrayed as a silent conductor orchestrating the symphony of existence, where raindrops, wind whispers, and the sun's fiery blaze play harmoniously. Each dawn unveils a new chapter as clouds wander across the sky's canvas, painting it with the colors of morning. Storms represent a visceral dance of nature's elements, echoing its tumultuous rhythm with thunder and lightning. Amidst this chaos lies a tranquil serenade, embodied by the gentle breeze and dew-kissed grass, illustrating the delicate balance of existence. Seasons mark the passage of time, weaving tales of transformation from winter's frost to spring's bloom. Nature, as the master artisan, sculpts landscapes with precision, from deserts to rainforests, with weather as the silent narrator of Earth's story. In the whispers of the elements, the future is foretold, written in the shifting clouds and winds, inviting those who listen to decipher its celestial manuscript. Let us revel in the poetry of weather, finding solace in its ever-changing embrace, a reminder of our cosmic connection and the eternal dance of existence.",

    "Weather is like a carefully orchestrated dance of physics, where molecules collide and energy flows, giving rise to a myriad of atmospheric phenomena. Clouds form and disperse in the sky, following the laws of thermodynamics, as moisture condenses and evaporates in a delicate balance. Precipitation, whether rain, snow, or hail, emerges from the intricate interplay of air masses and warm currents ascending. Winds, unseen yet powerful, sculpt landscapes as they traverse continents, driven by pressure differentials and the Earth's rotation. Temperature serves as the conductor of climate, dictating the rhythms of seasons as the sun's rays fluctuate with the tilt of the Earth's axis. Storms, chaotic yet mesmerizing, reveal the principles of chaos theory, where swirling vortices converge in a display of turbulence and order. From the microscale of individual clouds to the macroscale of global climate, weather emerges from a complex interplay of variables, offering a rich field for exploration and understanding. Meteorologists rely on data from satellites and atmospheric sensors to predict weather patterns, offering us a glimpse into the intricate beauty of the cosmos through the lens of science",

    "The weather, a continuous dialogue within Earth's atmosphere, is a complex interplay of temperature, pressure, and moisture, guided by the movements of the sun and the planet. It's essentially the result of how warm air rises, cool air descends, and winds circulate, shaping the environments we live in and the climates we experience. Each weather event, from gentle breezes to powerful storms, provides insights into how our planet functions, showcasing its ability to adapt and endure. Meteorologists study weather patterns using data and technology like satellites and computer models to predict its behavior. Weather affects every aspect of human life, influencing activities like farming, travel, business, and leisure, and shaping our daily routines and societal awareness. Yet, weather isn't just a human concern; it affects all life on Earth, influencing the behavior of animals, plant distribution, and ecosystem health. With our planet undergoing significant changes, understanding weather is crucial for our stewardship of Earth. Let's marvel at its wonders and appreciate its variability as a reflection of our planet's vitality and dynamism."
]

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

print("""



""")

print("ROUGE scores for TF IDF:")
# for ref_summary in reference_summaries:

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