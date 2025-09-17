import os
import sqlite3
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from encoders.clip_embeddings import CLIPEmbeddings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class QueryAgentDataBase:
    def __init__(
        self,
        name: str,
        read_sql_command: str,
        encoder_type: str = "text",
        table_schema_sql: str = None
    ):
        self.name = name
        self.read_sql_command = read_sql_command
        self.agent = ""
        self.faiss_database = ""
        self.sql_database = ""
        self.encoder_type = encoder_type
        self.table_schema_sql = table_schema_sql
        self.columns = self._parse_columns_from_select()


    def _parse_columns_from_select(self):
        try:
            sql_lower = self.read_sql_command.lower()
            select_part = sql_lower.split('from')[0]
            columns_part = select_part.replace('select', '').strip()
            columns = [c.strip() for c in columns_part.split(',') if c.strip() and c.strip() != "id"]
            return columns
        except Exception as e:
            raise ValueError(f"Unable to parse columns from read_sql_command: {e}")

    def _assign(self, agentName: str):
        self.faiss_database = f"./data/{agentName}/faiss_{self.name}"
        self.sql_database = f"./data/{agentName}/{self.name}.db"

    def set_add_data_function(self, func: callable):
        self._add_data_function = func

    def get_insert_sql(self, payload: dict):
        """Renvoie la requête SQL d'insertion pour le payload donné."""
        if hasattr(self, "_add_data_function") and self._add_data_function is not None:
            # Si on a spécifié une fonction d'ajout personnalisée
            # On la renvoie
            return self._add_data_function(payload)

        # Sinon, on construit la requête SQL
        values = []
        for col in self.columns:
            val = payload.get(col, "")
            if isinstance(val, str):
                val = val.replace("'", "''")
            values.append(f"'{val}'")

        columns_sql = ", ".join(self.columns)
        values_sql = ", ".join(values)
        return f"INSERT INTO {self.name} ({columns_sql}) VALUES ({values_sql})"

    def __repr__(self):
        return (
            f"<QueryAgentDataBase(name='{self.name}', encoder='{self.encoder_type}', "
            f"sql_db='{self.sql_database}', "
            f"read_sql='{self.read_sql_command[:40]}...')>"
        )
    
    def to_dict(self):
        return {
            "name": self.name,
            "read_sql_command": self.read_sql_command,
            "encoder_type": self.encoder_type,
            "table_schema_sql": self.table_schema_sql
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            read_sql_command=data["read_sql_command"],
            encoder_type=data.get("encoder_type", "text"),
            table_schema_sql=data.get("table_schema_sql")
        )


class QueryAgent:
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Tu es un assistant expert qui doit répondre à la question en t'appuyant sur les extraits suivants.

        **Instructions importantes :**
        - Lis attentivement les extraits fournis.
        - Tu dois toujours choisir et utiliser l'un des trois formats de réponse ci-dessous, sans jamais en inventer un quatrième.
        - Choisis le format en fonction des extraits :

        [1] Si tu trouves directement la réponse exacte, mot pour mot ou quasi identique dans les extraits, alors réponds au format :
            [REPONSE EXACTE] la réponse trouvée telle quelle ou quasi identique.

        [2] Si tu ne trouves pas la réponse exacte mais que les extraits contiennent assez d'informations pour la déduire ou la reformuler, alors réponds au format :
            [REPONSE GENEREE] ta réponse synthétique ou déduite en t'appuyant sur les extraits.

        [3] Si tu ne trouves pas la réponse et que les extraits ne contiennent pas assez d'informations pour répondre même en devinant, alors réponds au format :
            [REPONSE IMPOSSIBLE] Je n'ai pas les éléments suffisants pour répondre à cette question.

        ---

        **Extraits :**
        {context}

        **Question :**
        {question}

        **Réponse :**
        """
    )

    def __init__(
        self,
        name: str,
        description: str,
        faiss_database: list[QueryAgentDataBase],
        llm_model="gpt-4.1-nano"
    ):
        self.name = name
        self.description = description
        self.faiss_database = faiss_database
        self.llm_model = llm_model
        self.active_databases = []

        agent_folder = f"./data/{self.name}"
        os.makedirs(agent_folder, exist_ok=True)

        for db in self.faiss_database:
            db._assign(self.name)

        self.encoders = {}
        self._init_encoders()
        self.loaded = False

    def _init_encoders(self):
        print(f"[INFO] {self.name} Initializing text encoder...")
        try:
            text_embed = OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            _ = text_embed.embed_query("test")
        except Exception:
            print("[WARNING] Fallback to ada-002")
            text_embed = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        self.encoders["text"] = text_embed
        self.encoders["image"] = CLIPEmbeddings()
        print(f"[INFO] {self.name} CLIP encoder initialized.")

    def add_database(self, db: QueryAgentDataBase):
        db._assign(self.name)
        self.faiss_database.append(db)
        print(f"[INFO] Database {db.name} added to agent.")

    def _guess_table_schema_sql(self, db: QueryAgentDataBase):
        print(f"[WARNING] No schema provided for {db.name}. Guessing table structure from read_sql_command...")

        try:
            # Naïf : on va prendre ce qu'il y a entre SELECT et FROM
            read_sql = db.read_sql_command.lower()
            select_part = read_sql.split('from')[0]
            columns_part = select_part.replace('select', '').strip()
            columns = [c.strip() for c in columns_part.split(',') if c.strip()]

            # Construire les colonnes
            schema_cols = []
            for col in columns:
                if col.lower() == "id":
                    schema_cols.append("id INTEGER PRIMARY KEY AUTOINCREMENT")
                else:
                    schema_cols.append(f"{col} TEXT")

            create_sql = f"""
                CREATE TABLE IF NOT EXISTS {db.name} (
                    {", ".join(schema_cols)}
                )
            """
            print(f"[INFO] Guessed schema for {db.name}: {create_sql.strip()}")
            return create_sql
        except Exception as e:
            raise ValueError(f"Unable to guess schema for {db.name}: {e}")


    def _initialize_sql_database(self, db):
        if not db.table_schema_sql:
            db.table_schema_sql = self._guess_table_schema_sql(db)
        print(f"[DATA] Creating new SQLite database for {db.name}...")
        conn = sqlite3.connect(db.sql_database)
        cursor = conn.cursor()
        cursor.execute(db.table_schema_sql)
        conn.commit()
        conn.close()
        print(f"[INFO] Database {db.name}.db created with schema.")

    def load(self):
        print(f"[INFO] {self.name} Loading agent...")
        self.stores = []
        self.retrievers = []
        self.active_databases = []
        
        for db in self.faiss_database:
            # Check and initialize SQL db if missing
            if not os.path.exists(db.sql_database):
                print(f"[ERROR] SQLite DB not found for {db.name}, creating...")
                self._initialize_sql_database(db)

            # Check and create FAISS index if missing
            if not os.path.exists(db.faiss_database):
                print(f"[ERROR] FAISS index not found for {db.name}, building...")
                docs = self.load_data_from_db(db)
                self.build_vectorstore(docs, db)

            # Load FAISS
            encoder = self.encoders[db.encoder_type]
            try:
                store = FAISS.load_local(db.faiss_database, encoder, allow_dangerous_deserialization=True)
                self.stores.append(store)
                self.retrievers.append(store.as_retriever(search_kwargs={"k": 15}))
                self.active_databases.append(db)

                print(f"[INFO]{db.name} FAISS loaded and retriever initialized.")
            except Exception as e:
                print(f"[ERROR] Failed to load FAISS for {db.name}: {e}")


        self.llm = ChatOpenAI(
            model=self.llm_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.2
        )
        self.chain = QueryAgent.prompt | self.llm
        self.loaded = True
        print(f"[INFO] {self.name} Agent loaded with {len(self.faiss_database)} databases.")

    def _extract_filepath(self, text):
        import ast
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, tuple):
                for item in parsed:
                    if isinstance(item, str) and os.path.exists(item):
                        return item
        except Exception:
            pass
        if os.path.exists(text.strip()):
            return text.strip()
        return text.strip().strip('"').strip("'")

    def pretty_print(doc):
        try:
            parsed = eval(doc.page_content)
            if isinstance(parsed, tuple):
                return " - ".join(str(item) for item in parsed if item and str(item).strip() not in ("None", ""))
            return doc.page_content
        except:
            return doc.page_content

    def retrieve_best_image(self, query_text):
        """Récupère la meilleure image correspondant à la requête."""
        clip_encoder = self.encoders["image"]
        query_vector = clip_encoder.embed_query(query_text)

        image_store = None
        for db, store in zip(self.faiss_database, self.stores):
            if db.encoder_type == "image":
                image_store = store
                break

        if not image_store:
            # Pas de database d'images...
            raise ValueError("No image index found.")

        results = image_store.similarity_search_by_vector(query_vector, k=4)
        if results:
            print(f"[INFO] Found {len(results)} images matching the query.")
            print(f"[DEBUG] Best match: {results}")
            paths = [self._extract_filepath(doc.page_content) for doc in results]
            if paths:
                return list(set(paths))  # Retirer les doublons
        return None

    def query(self, question: str):
        """
        Effectue une recherche dans les bases de données de l'agent.
        Retourne une réponse basée sur les données disponibles.
        """
        if not self.loaded:
            self.load()

        text_parts = []
        if not self.retrievers:
            print("[WARNING] No retrievers available. Please check if your databases have data and FAISS indexes.")
            return "[REPONSE IMPOSSIBLE] Aucune base de données n'a pu être chargée pour répondre."

        for retriever, db in zip(self.retrievers, self.active_databases):
            if db.encoder_type == "text":
                result = retriever.invoke(question)
                text_parts.extend([QueryAgent.pretty_print(d) for d in result])
                print(f"[DEBUG] Results from {db.name}: {text_parts[-1]}")

        context_text = "DESCRIPTIONS:\n" + "\n".join(text_parts)
        print(f"[DEBUG] Context text: {context_text}")
        response = self.chain.invoke({
            "context": context_text,
            "question": question
        })
        return response

    def add_data(self, base_name, payload):
        target_db = next((db for db in self.faiss_database if db.name == base_name), None)
        if target_db is None:
            raise ValueError(f"Unknown database: {base_name}")

        insert_sql = target_db.get_insert_sql(payload)

        conn = sqlite3.connect(target_db.sql_database)
        cursor = conn.cursor()
        cursor.execute(insert_sql)
        conn.commit()
        conn.close()
        print(f"[INFO] Data inserted into {base_name}.")

    def reload_db(self, target_db_name):
        target_db = next((db for db in self.faiss_database if db.name == target_db_name), None)
        if target_db is None:
            raise ValueError(f"Unknown database: {target_db_name}")

        docs = self.load_data_from_db(target_db)
        self.build_vectorstore(docs, target_db)
        print(f"[INFO] {target_db_name} reloaded successfully.")

    def load_data_from_db(self, agent_db: QueryAgentDataBase):
        print(f"[INFO] Loading data from {agent_db.name}...")
        conn = sqlite3.connect(agent_db.sql_database)
        cursor = conn.cursor()
        cursor.execute(agent_db.read_sql_command)
        rows = cursor.fetchall()
        conn.close()

        docs = []
        for row in rows:
            metadata = {f"column_{i}": value for i, value in enumerate(row)}
            docs.append(Document(page_content=str(row), metadata=metadata))
        print(f"[INFO] Loaded {len(docs)} documents from {agent_db.name}.")
        return docs

    def build_vectorstore(self, docs, agent_db: QueryAgentDataBase):
        if not docs:
            print(f"[WARNING] No documents found for {agent_db.name}. Skipping FAISS build.")
            return

        print(f"[INFO] Building FAISS for {agent_db.name} with encoder {agent_db.encoder_type}...")
        encoder = self.encoders[agent_db.encoder_type]
        vectorstore = FAISS.from_documents(docs, embedding=encoder)
        vectorstore.save_local(agent_db.faiss_database)
        print(f"[INFO] FAISS index saved to {agent_db.faiss_database}")

    def create(self):
        for db in self.faiss_database:
            if not os.path.exists(db.sql_database):
                print(f"[ERROR] SQLite DB not found for {db.name}, creating...")
                self._initialize_sql_database(db)
            docs = self.load_data_from_db(db)
            self.build_vectorstore(docs, db)
            print(f"[INFO] Vectorstore for {db.name} built and saved.")

    def export(self, filepath):
        import json

        agent_data = {
            "name": self.name,
            "description": self.description,
            "llm_model": self.llm_model,
            "faiss_database": [db.to_dict() for db in self.faiss_database]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(agent_data, f, indent=4, ensure_ascii=False)

        print(f"[INFO] Agent configuration exported to {filepath}")


    def __repr__(self):
        return (
            f"<QueryAgent(name='{self.name}', model='{self.llm_model}', "
            f"databases={len(self.faiss_database)})>"
            f"  - [{',\n'.join(db.__repr__() for db in self.faiss_database)}]"
        )

    @classmethod
    def import_from_file(cls, filepath):
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            agent_data = json.load(f)

        # Reconstruire les DB
        faiss_dbs = [
            QueryAgentDataBase.from_dict(db_dict) for db_dict in agent_data["faiss_database"]
        ]

        # Créer l'agent
        agent = cls(
            name=agent_data["name"],
            description=agent_data["description"],
            faiss_database=faiss_dbs,
            llm_model=agent_data.get("llm_model", "gpt-4.1-nano")
        )

        print(f"✅ Agent configuration imported from {filepath}")
        return agent
