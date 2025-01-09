import spacy  # Importing spacy
import warnings  # Importing warnings

# Suppress specific Spacy warnings
warnings.filterwarnings(
    "ignore",
    message=r".*\[W007\].*",
    category=UserWarning
)

# Load the medium-sized English language model
nlp = spacy.load("en_core_web_md")

# Word Similarity Comparisons
print("\n\tSIMILARITY WITH SPACY\n")
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
word4 = nlp("dog")  # Additional word for comparison as an example of my own
word5 = nlp("bone")  # Additional word for comparison as an example of my own

print(f"Similarity between '{word1}' and '{word2}': {word1.similarity(word2)}")  # Cat and monkey
print(f"Similarity between '{word3}' and '{word2}': {word3.similarity(word2)}")  # Banana and monkey
print(f"Similarity between '{word3}' and '{word1}': {word3.similarity(word1)}")  # Banana and cat
print(f"Similarity between '{word1}' and '{word4}': {word1.similarity(word4)}")  # Cat and dog
print(f"Similarity between '{word4}' and '{word5}': {word4.similarity(word5)}")  # Dog and bone
print(f"Similarity between '{word1}' and '{word5}': {word1.similarity(word5)}")  # Cat and bone

# Vector Similarity Comparisons
print("\n\tWORKING WITH VECTORS\n")
tokens = nlp("cat monkey banana dog bone")
for token1 in tokens:
    for token2 in tokens:
        print(f"{token1.text} - {token2.text}: {token1.similarity(token2)}")

# Sentence Similarity Comparisons
print("\n\tWORKING WITH SENTENCES\n")
sentence_to_compare = "A monkey is eating a banana in the jungle"
sentences = [
    "A tiger roams in the forest",
    "The cat is resting under the tree",
    "Monkeys swing from trees and enjoy bananas",
    "Bananas are a common fruit in tropical jungles",
    "A child is playing with a ball"
]
model_sentence = nlp(sentence_to_compare)
print("\nComparing sentences to: \"" + sentence_to_compare + "\"")
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(f"\"{sentence}\" - {similarity}")

# Additional Sentence Similarity Example
print("\n\tMY OWN EXAMPLE WITH SENTENCES\n")
my_sentence_to_compare = "A dog is chewing on a bone in the yard"
my_sentences = [
    "The cat is playing with a ball of yarn",
    "Dogs often enjoy chewing bones",
    "A group of children is playing soccer in the park",
    "The bird is perched on a tree",
    "Bones are commonly given to dogs as treats"
]
my_model_sentence = nlp(my_sentence_to_compare)
print("\nComparing sentences to: \"" + my_sentence_to_compare + "\"")
for sentence in my_sentences:
    similarity = nlp(sentence).similarity(my_model_sentence)
    print(f"\"{sentence}\" - {similarity}")

# Highlighting Why Sentence Similarity Matters
print("\n\tWHY SENTENCE SIMILARITY MATTERS\n")
explanation = (
    "While comparing individual words like 'cat', 'monkey', and 'banana' provides "
    "insights into basic semantic relationships, comparing full sentences allows us "
    "to evaluate the model's ability to understand context and meaning in a broader scope. "
    "For instance, sentences involving 'monkey', 'banana', or 'jungle' are more relevant "
    "to the model sentence, showing Spacy's understanding of connected ideas. "
    "Similarly, the custom example demonstrates how the model relates sentences about dogs "
    "and bones more closely than unrelated sentences about birds or children playing. "
    "This demonstrates a deeper level of natural language processing."
)
print(explanation)

# Run the example file with the simpler language model "en_core_web_sm" and write a note on what you notice is different
# From the model "en_core_web_md"

'''When i run the example file i figured the environment is encountering repeated memory allocation issues
particularly when using the "en_core_web_sm" model. This can  happen due to the resource-intensive nature 
of running word and sentence similarity computations, meanwhile using "en_core_web_md" doesn't encounter
any warnings'''

"""
NOTE: 
1. As requested from previous submission i have Imported warnings Module  
to handle Spacy warnings
2.i have  Used warnings.filterwarnings to suppress messages about specific issues, like no matching vectors
for certain components (common in Spacy when working with similarity).
"""
