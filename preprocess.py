import os
import re
import cv2
import networkx as nx
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Initialize a directed graph
graph = nx.Graph()  # Changed to undirected graph

# Ensure required NLTK packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the path to the directory with your data
data_dir = './papers_data/'

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text (stopword removal, lemmatization, and optional summarization)
def preprocess_text(file_path):
    try:
        with open(file_path, 'r') as file:
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

# Process each paper in the directory
papers = []  # List to keep track of paper names
for file_name in os.listdir(data_dir):
    if file_name.endswith('_text.txt'):
        try:
            paper_name = file_name.split('_')[0]
            papers.append(paper_name)  # Add paper name to the list
            page_number = file_name.split('_')[1]
            txt_file_path = os.path.join(data_dir, file_name)
            img_file_path = os.path.join(data_dir, f"{paper_name}_{page_number}_image.png")

            # Preprocess text
            sections = preprocess_text(txt_file_path)

            # Add paper node if not already added
            if paper_name not in graph:
                graph.add_node(paper_name, type='paper')

            # Add section nodes and connect to paper
            for section, content in sections.items():
                section_node = f"{paper_name}_{section}"
                graph.add_node(section_node, type='section', content=content)
                graph.add_edge(paper_name, section_node)

                # Preprocess and link images (if they exist and are relevant)
                if os.path.exists(img_file_path):
                    image_data = preprocess_image(img_file_path)
                    if image_data:  # Only add relevant images
                        image_node = f"{paper_name}_image_{page_number}"
                        graph.add_node(image_node, type='image', path=image_data)
                        graph.add_edge(section_node, image_node)

        except Exception as e:
            print(f"Error processing paper {file_name}: {e}")

# Add undirected connections between every paper node
for i in range(len(papers)):
    for j in range(i + 1, len(papers)):
        graph.add_edge(papers[i], papers[j])  # Create an undirected connection

# Save the graph in multiple formats
nx.write_gml(graph, "paper_graph.gml")
nx.write_graphml(graph, "paper_graph.graphml")  # GraphML for better compatibility
nx.write_gpickle(graph, "paper_graph.gpickle")  # For Python object serialization

# Visualization
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(graph, seed=42)  # Positions for all nodes
nx.draw(graph, pos, with_labels=True, node_color='skyblue', font_size=10, font_weight='bold')
plt.title('Paper Graph with Undirected Connections between Paper Nodes')
plt.show()
