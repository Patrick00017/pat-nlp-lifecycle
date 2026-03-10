import fitz  # PyMuPDF
import spacy
from datasets import Dataset

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text


def identify_questions(text):
    doc = nlp(text)
    questions = []
    for sent in doc.sents:
        if sent[-1].text == "?":
            questions.append(sent.text)
    return questions


def simple_qa_pairs(text):
    questions = identify_questions(text)
    qa_pairs = []
    doc = nlp(text)
    sentences = list(doc.sents)
    for question in questions:
        question_idx = [i for i, sent in enumerate(sentences) if sent.text == question][
            0
        ]
        if question_idx > 0:
            answer = sentences[question_idx - 1].text
            qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs


def create_qa_dataset(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    qa_pairs = simple_qa_pairs(text)
    dataset = Dataset.from_list(qa_pairs)
    return dataset


# Example usage
pdf_path = "TPLogAD.pdf"
dataset = create_qa_dataset(pdf_path)

# Save the dataset to disk
dataset.save_to_disk("qa_dataset")

# Load the dataset to verify
loaded_dataset = Dataset.load_from_disk("qa_dataset")
for example in loaded_dataset:
    print("Question:", example["question"])
    print("Answer:", example["answer"])
