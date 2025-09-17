from flask import Flask, request, jsonify
from query_agent import QueryAgent, QueryAgentDataBase
import os
import base64

app = Flask(__name__)

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

IMAGE_STORAGE_FOLDER = r"H:\github\Les Reliques des Aînées Assistant\Generic Assistant\images"

# 📌 Fonction pour générer l'INSERT SQL pour chaque DB
def notes_add_data(payload):
    content = payload.get("content")
    author = payload.get("author")
    timestamp = payload.get("timestamp")
    if not content:
        raise ValueError("Missing 'content'")
    return f"INSERT INTO notes (content, author, timestamp) VALUES ('{content}', '{author}', '{timestamp}')"

def images_add_data(payload):
    image_base64 = payload.get("base64")
    filename = payload.get("file_name")

    if not image_base64:
        print("[ERROR] Missing base64 data in payload")
        raise ValueError("Missing 'base64'")

    os.makedirs(IMAGE_STORAGE_FOLDER, exist_ok=True)
    filepath = os.path.join(IMAGE_STORAGE_FOLDER, filename)

    with open(filepath, "wb") as f:
        f.write(base64.b64decode(image_base64))

    return f"INSERT INTO image_embeddings (filepath) VALUES ('{filepath}');"



desc_db.set_add_data_function(notes_add_data)
notes_db.set_add_data_function(notes_add_data)
image_db.set_add_data_function(images_add_data)

ChunkDatabase = QueryAgentDataBase(
    name="chunk",
    read_sql_command="SELECT id, text FROM chunk",
    encoder_type="text"
)

Agent = QueryAgent(
    name="image_multimodal",
    description="Agent RAG multimodal images + transcriptions",
    faiss_database=[desc_db, image_db, ChunkDatabase, notes_db]
)



# --------------------------------------------------------
#                   ROUTES FLASK
# --------------------------------------------------------


@app.route('/add_data/<db_name>', methods=['POST'])
def add_data(db_name):
    """
    Ajoute des données à la DB spécifiée.
    Exemple: POST /add_data/notes
             { "text": "Ma note test" }
    """
    try:
        print(f"[INFO] Adding data to {db_name} database")
        payload = request.json
        Agent.add_data(db_name, payload)
        return jsonify({"status": "success", "db": db_name}), 200
    except Exception as e:
        print(f"[ERROR] Failed to add data to {db_name}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route('/reload/<db_name>', methods=['POST'])
def reload_data(db_name):
    """
    Recharge les données de la DB spécifiée.
    Exemple: POST /reload/notes
    """
    try:
        Agent.reload_db(db_name)
        return jsonify({"status": "success", "db": db_name}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400



@app.route('/query', methods=['POST'])
def query():
    """
    Effectue une recherche dans l'agent.
    Exemple: POST /query
             { "question": "Quel est le contenu de ma note ?" }
    """
    try:
        payload = request.json
        question = payload.get("question")
        if not question:
            raise ValueError("Missing 'question' in payload")
        if "image" in question.lower():
            response = Agent.retrieve_best_image(question)
            responses = []
            for image in response:
                image_path = os.path.join(IMAGE_STORAGE_FOLDER, image)
                if not os.path.exists(image_path):
                    print(f"[ERROR] Image not found: {image_path}")
                    continue
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                responses.append(base64.b64encode(image_data).decode('utf-8'))
            return jsonify({"image": responses}), 200
        else:
            response = Agent.query(question)
            if isinstance(response, str):
                return jsonify({"result": response}), 200
            elif hasattr(response, 'content'):
                # If response is a LangChain response object
                return jsonify({"result": response.content}), 200
            return jsonify({"result": response}), 200
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return jsonify({"status": "reason", "error": str(e)}), 400

@app.route('/health', methods=['GET'])
def healthcheck():
    try:
        # Vérifie si l'agent est chargé
        status = "ok" if Agent.loaded else "initializing"
        return jsonify({"status": status}), 200
    except Exception as e:
        return jsonify({"status": "error", "details": str(e)}), 500

# --------------------------------------------------------

if __name__ == '__main__':
    # Facultatif: charger les index FAISS au démarrage
    Agent.load()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
