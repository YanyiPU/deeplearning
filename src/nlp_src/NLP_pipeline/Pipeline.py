"""
Coding the NLP Pipeline in Python
"""

import spacy


# 1.建立 NLP Pipeline

# 1.1 数据

# Load the large English NLP model
nlp = spacy.load("en_core_web_lg")

# The text we want to examine
text = """London is the capital and most populous city of England and 
    the United Kingdom.  Standing on the River Thames in the south east 
    of the island of Great Britain, London has been a major settlement 
    for two millennia. It was founded by the Romans, who named it Londinium.
    """

# Parse the text with spaCy. This runs the entire pipeline.
doc = nlp(text)

for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")









# 1.2 分句



# 1.3 分词





