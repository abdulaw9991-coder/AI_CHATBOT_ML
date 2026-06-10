# import pickle
# import json
# import random
# import datetime

# # LOAD MODEL
# model = pickle.load(open("model.pkl", "rb"))
# vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
# le = pickle.load(open("label_encoder.pkl", "rb"))

# data = json.loads(open("intents.json").read())

# def get_response(tag):
#     for intent in data["intents"]:
#         if intent["tag"] == tag:

#             if tag == "time":
#                 return datetime.datetime.now().strftime("%H:%M:%S")

#             return random.choice(intent["responses"])

#     return "I don't understand"

# print("🤖 Chatbot Started (ML Model)")

# while True:
#     user = input("You: ")

#     if user.lower() == "bye":
#         print("Bot: Goodbye!")
#         break

#     X = vectorizer.transform([user])
#     prediction = model.predict(X)
#     tag = le.inverse_transform(prediction)[0]

#     response = get_response(tag)

#     print("Bot:", response)
###########################################3
import pickle
import json
import random
import datetime

from memory import add_memory, get_memory

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

data = json.loads(open("intents.json").read())

def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I don't understand"

print("🤖 Chatbot Started (type 'bye' to exit, 'history' to see memory)")

while True:
    user = input("You: ")

    # EXIT
    if user.lower() == "bye":
        print("Bot: Goodbye!")
        add_memory(user, "Goodbye!")
        break

    # SHOW MEMORY
    if user.lower() == "history":
        print("\n📜 Chat History:")
        for chat in get_memory():
            print(f"You: {chat['user']} | Bot: {chat['bot']}")
        continue

    # TIME FEATURE (optional smart feature)
    if "time" in user.lower():
        bot_reply = datetime.datetime.now().strftime("%H:%M:%S")
        print("Bot:", bot_reply)
        add_memory(user, bot_reply)
        continue

    # ML PREDICTION
    X = vectorizer.transform([user])
    prediction = model.predict(X)
    print("DEBUG: Vectorized input =", X.toarray())

    tag = le.inverse_transform(prediction)[0]
    print("DEBUG: Prediction =", prediction)
    # RESPONSE
    bot_reply = get_response(tag)

    print("Bot:", bot_reply)

    # SAVE TO MEMORY
    add_memory(user, bot_reply)