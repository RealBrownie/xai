import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import sys


class PokemonClassifier:
    def __init__(self, model_path):
        # List of Pokemon classes
        self.classes = ['Abra', 'Aerodactyl', 'Alakazam', 'Alolan Sandslash', 'Arbok', 'Arcanine',
                        'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise', 'Bulbasaur', 'Butterfree',
                        'Caterpie', 'Chansey', 'Charizard', 'Charmander', 'Charmeleon', 'Clefable',
                        'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett', 'Ditto', 'Dodrio',
                        'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee',
                        'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute', 'Exeggutor', 'Farfetchd',
                        'Fearow', 'Flareon', 'Gastly', 'Gengar', 'Geodude', 'Gloom', 'Golbat',
                        'Goldeen', 'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe', 'Gyarados',
                        'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea', 'Hypno', 'Ivysaur',
                        'Jigglypuff', 'Jolteon', 'Jynx', 'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna',
                        'Kangaskhan', 'Kingler', 'Koffing', 'Krabby', 'Lapras', 'Lickitung',
                        'Machamp', 'Machoke', 'Machop', 'Magikarp', 'Magmar', 'Magnemite',
                        'Magneton', 'Mankey', 'Marowak', 'Meowth', 'Metapod', 'Mew', 'Mewtwo',
                        'Moltres', 'MrMime', 'Muk', 'Nidoking', 'Nidoqueen', 'Nidorina', 'Nidorino',
                        'Ninetales', 'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect',
                        'Persian', 'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir', 'Poliwag',
                        'Poliwhirl', 'Poliwrath', 'Ponyta', 'Porygon', 'Primeape', 'Psyduck',
                        'Raichu', 'Rapidash', 'Raticate', 'Rattata', 'Rhydon', 'Rhyhorn',
                        'Sandshrew', 'Sandslash', 'Scyther', 'Seadra', 'Seaking', 'Seel',
                        'Shellder', 'Slowbro', 'Slowpoke', 'Snorlax', 'Spearow', 'Squirtle',
                        'Starmie', 'Staryu', 'Tangela', 'Tauros', 'Tentacool', 'Tentacruel',
                        'Vaporeon', 'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume',
                        'Voltorb', 'Vulpix', 'Wartortle', 'Weedle', 'Weepinbell', 'Weezing',
                        'Wigglytuff', 'Zapdos', 'Zubat']

        # Load the model
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def download_image(self, url):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    def preprocess_image(self, img):
        try:
            # Convert image to RGB if it's not
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize image to 160x160
            img = img.resize((160, 160))

            # Convert to array and preprocess
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Normalize the image
            img_array = img_array / 255.0

            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def predict(self, url, top_k=3):
        # Download image
        img = self.download_image(url)
        if img is None:
            return []

        # Preprocess image
        processed_img = self.preprocess_image(img)
        if processed_img is None:
            return []

        try:
            # Make prediction
            predictions = self.model.predict(processed_img)

            # Get top k predictions
            top_indices = predictions[0].argsort()[-top_k:][::-1]

            results = []
            for idx in top_indices:
                results.append({
                    'pokemon': self.classes[idx],
                    'confidence': float(predictions[0][idx])
                })

            return results
        except Exception as e:
            print(f"Error making prediction: {e}")
            return []

def main():
    # Initialize classifier
    classifier = PokemonClassifier('./Modelle/best_model.keras')  # Replace with your model path

    # Example usage
    while True:
        url = input("\nEnter image URL (or 'quit' to exit): ")
        if url.lower() == 'quit':
            break

        predictions = classifier.predict(url)

        if predictions:
            print("\nTop 3 predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['pokemon']} - {pred['confidence']*100:.2f}% confidence")
        else:
            print("Could not make predictions for this image.")

if __name__ == "__main__":
    main()