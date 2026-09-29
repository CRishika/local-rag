from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

vec = model.encode("Telangana's capital city is Hyderabad")
print(len(vec))
print(vec[:5])

vec_1 = model.encode("I like reading books")
print(len(vec_1))