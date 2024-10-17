import os
import re
import cv2
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from neo4j import GraphDatabase

# Ensure required NLTK packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Neo4j Connection Configuration
uri = "bolt://localhost:7687"  # Update with your Neo4j connection details
username = "neo4j"  # Replace with your username
password = "your_password"  # Replace with your password

driver = GraphDatabase.driver(uri, auth=(username, password))

# Define the path to the directory with your data
data_dir = r"E:\BTP\PDF_Extracted"

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text (stopword removal, lemmatization)
def preprocess_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Simple section detection using regex
        sections = re.split(r'(Abstract|Introduction|Methods|Results|Conclusion)', text, flags=re.IGNORECASE)
        section_data = defaultdict(str)
        
        for i in range(1, len(sections), 2):
            section_title = sections[i].strip().lower()  # Normalize section title
            content = sections[i + 1].strip()

            # Tokenize, remove stop words, and lemmatize the content
            words = word_tokenize(content)
            filtered_content = [lemmatizer.lemmatize(w.lower()) for w in words if w.lower() not in stop_words and w.isalnum()]
            processed_content = ' '.join(filtered_content)

            section_data[section_title] = processed_content

        return section_data

    except Exception as e:
        print(f"Error processing text file {file_path}: {e}")
        return {}

# Function to process images using OpenCV (basic filtering to check relevance)
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image file {image_path} could not be loaded.")

        # Check image size and entropy (basic relevance filter)
        if image.size < 5000:  # Example: filter out small images
            return None

        return image_path  # For now, just return the valid image path
    
    except Exception as e:
        print(f"Error processing image file {image_path}: {e}")
        return None

# Neo4j helper function to create nodes and relationships
def create_node(session, node_label, node_name, attributes):
    query = f"CREATE (n:{node_label} {{name: $name, attributes: $attributes}})"
    session.run(query, name=node_name, attributes=attributes)

def create_relationship(session, node1, node2, relationship_type="LINKS"):
    query = f"MATCH (a {{name: $node1}}), (b {{name: $node2}}) CREATE (a)-[:{relationship_type}]->(b)"
    session.run(query, node1=node1, node2=node2)

# Process each paper in the directory
with driver.session() as session:
    for file_name in os.listdir(data_dir):
        if file_name.endswith('_text.txt'):
            try:
                paper_name = file_name.split('_')[0]  # Extract paper name
                page_number = file_name.split('_')[1]  # Extract page number
                txt_file_path = os.path.join(data_dir, file_name)
                img_file_path = os.path.join(data_dir, f"{paper_name}_{page_number}_image.png")

                # Preprocess text
                sections = preprocess_text(txt_file_path)

                # Add paper node to Neo4j
                create_node(session, "Paper", paper_name, {"type": "paper"})

                # Add section nodes and connect them to the paper
                for section, content in sections.items():
                    section_node = f"{paper_name}_{section}"
                    create_node(session, "Section", section_node, {"type": "section", "content": content})
                    create_relationship(session, paper_name, section_node)

                    # Preprocess and link images (if they exist and are relevant)
                    if os.path.exists(img_file_path):
                        image_data = preprocess_image(img_file_path)
                        if image_data:  # Only add relevant images
                            image_node = f"{paper_name}_image_{page_number}"
                            create_node(session, "Image", image_node, {"type": "image", "path": image_data})
                            create_relationship(session, section_node, image_node)

            except Exception as e:
                print(f"Error processing paper {file_name}: {e}")

print("Graph data uploaded to Neo4j successfully!")

# Close the Neo4j connection
driver.close()
