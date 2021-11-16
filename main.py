from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")

model = AutoModelForSequenceClassification.from_pretrained(
    "finiteautomata/beto-sentiment-analysis", num_labels=2
)

text = "Good night 😊"
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)
