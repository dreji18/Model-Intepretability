# import libraries
import streamlit as st
from streamlit import components

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from transformers_interpret import SequenceClassificationExplainer

# functions
@st.experimental_singleton
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
    model = AutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model", from_tf=True)
    
    return model, tokenizer

def model_intepret(sentence, model, tokenizer):
    cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)
    word_attributions = cls_explainer(sentence)
    return cls_explainer, word_attributions

def transformers_visual(doc):
    model, tokenizer = load_model()
    cls_explainer, word_attributions = model_intepret(doc, model, tokenizer)
                    #st.write(word_attributions)
    return components.v1.html(cls_explainer.visualize()._repr_html_(), scrolling=True, height=350)

def main():
    st.title("Visualize Transformer Model")
    doc = st.text_area("Enter your sentence to visualize", "")  
    
    if doc:
        transformers_visual(doc)

# calling the main function
if __name__ == "__main__":
    main()    