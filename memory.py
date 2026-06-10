# Stores chat history

chat_memory = []

def add_memory(user, bot):
    chat_memory.append({
        "user": user,
        "bot": bot
    })

def get_memory():
    return chat_memory

def clear_memory():
    chat_memory.clear()