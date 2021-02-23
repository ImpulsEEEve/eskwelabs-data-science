import spacy

def border():
    print('\n\n' + 30*'=' + '\n\n')

# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_lg')

# tokenization
doc = nlp('I am flying to Manila.')
print([w.text for w in doc])

border()

# lemmatization
doc = nlp('This product integrates both libraries for downloading and applying patches')
for token in doc:
    print(token.text, token.lemma_)

border()

# part of speech tagging
doc = nlp('I have flown to Singapore. I am flying to Manila.')
for token in doc:
    print(token.text, token.pos_, token.tag_)

border()

print(spacy.explain('AUX'))
print(spacy.explain('VBZ'))

border()

print([w.text for w in doc if w.tag_=='VBG' or w.tag_=='VB'])

border()

for sent in doc.sents:
    print([sent[i] for i in range(len(sent))])

border()

doc = nlp('The Golden Gate Bridge is an iconic landmark in San Francisco.')
print([w.text for w in doc])
with doc.retokenize() as retokenizer:
    retokenizer.merge(doc[1:4])
with doc.retokenize() as retokenizer:
    retokenizer.merge(doc[7:9])
for token in doc:
    print(token.text, token.lemma_, token.pos_)

border()

#dependency parsing
doc = nlp('I want a green apple.')
for token in doc:
    print(token.text, token.pos_, token.dep_, spacy.explain(token.dep_))

from spacy import displacy
# displacy.serve(doc, style='dep')

border()

#entity recognition
doc = nlp('The firm earned $1.5 million in 2017, in comparison with $1.2 million in 2016.')
phrase = ''
for token in doc:
    if token.tag_ == '$':
        phrase = token.text
        i = token.i + 1
        while doc[i].tag_ == 'CD':
            phrase += doc[i].text + ' '
            i += 1
        phrase = phrase[:-1]
        print(phrase)

border ()

from IPython.core.display import display, HTML
doc = nlp('I want to fly to Manila.')
html = displacy.render(doc, style='ent', page=True)
display(HTML(html))

print(spacy.explain('GPE'))
print(spacy.explain('ORG'))

border()

#word similarity
print(nlp('apple').similarity(nlp('banana')))
print(nlp('king').similarity(nlp('queen')))

doc = nlp('I want a green apple.')
print(doc.similarity(doc[2:5]))
