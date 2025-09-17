from query_agent import QueryAgent, QueryAgentDataBase
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"




desc_db = QueryAgentDataBase(
    name="image_descriptions",
    read_sql_command="SELECT id, description FROM image_descriptions",
    encoder_type="text"
)

image_db = QueryAgentDataBase(
    name="image_embeddings",
    read_sql_command="SELECT id, filepath FROM image_embeddings",
    encoder_type="image"
)

notes_db = QueryAgentDataBase(
    name="notes",
    read_sql_command="SELECT id, content FROM notes",
    encoder_type="text"
)

agent = QueryAgent(
    name="image_multimodal",
    description="Agent RAG multimodal images",
    faiss_database=[notes_db, desc_db, image_db]
)

a = lambda payload: f"INSERT INTO image_descriptions (description) VALUES ('{payload['description']}')"

def test():
    print("Test function called")
    result = a({"description": "A beautiful sunset over the mountains."})
    print(f"Result: {result}")




# 📌 Fonction pour générer l'INSERT SQL pour chaque DB
def notes_add_data(payload):
    content = payload.get("content")
    author = payload.get("author")
    timestamp = payload.get("timestamp")
    if not content:
        raise ValueError("Missing 'content'")
    return f"INSERT INTO notes (content, author, timestamp) VALUES ('{content}', '{author}', '{timestamp}')"

if __name__ == "__main__":
    # Create the agent and its databases

    # Add some data to the image descriptions database
    #desc_db.set_add_data_function(lambda payload: f"INSERT INTO image_descriptions (description) VALUES ('{payload['description']}');")
    #agent.add_data(desc_db.name, {"description": "A beautiful sunset over the sea."})

    # Add some data to the image embeddings database
    #image_db.set_add_data_function(lambda payload: f"INSERT INTO image_embeddings (filepath) VALUES ('{payload['filepath']}');")
    #agent.add_data(image_db.name, {"filepath": "./data/sunset_over_mountais.jpg"})
    #,otes_db.set_add_data_function(notes_add_data)
    #agent.add_data(notes_db.name, {"content": "This is a test note.", "author": "Test User", "timestamp": "2023-10-01 12:00:00"})
    #agent.create()

    # Query the agent
    #response = agent.query("What does the sunset look like?")
    #print(response.content)  # Should return relevant information based on the added data
    #print("\n" + "=" * 60 + "\n")
    #response = agent.retrieve_best_image("sunset over mountains")
    #print(response)  # Should return the best matching
    print(agent)
    agent2 = QueryAgent.import_from_file("agent_config.json")
    agent2.load()
    print(agent2)