from mem0 import MemoryClient
client = MemoryClient(api_key="m0-Dsm5v1f6I6xViAex1HbSDGzIXPHQKsJnGCFTn62A")

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."}
]

# The default output_format is v1.0
client.add(messages, user_id="alex", output_format="v1.0")

output = client.get_all(user_id ="alex", output_format="v1.0")
memory = []
for outputs in output:
    memory.append(outputs["memory"])

print(memory)
