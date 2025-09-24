"""
Enhanced script to create dynamic Bill Nye training datasets with diverse scientific content.
"""

import json
import os
import random
from typing import List, Dict, Tuple

class BillNyeDatasetGenerator:
    """Dynamic Bill Nye dataset generator with diverse scientific content."""
    
    def __init__(self):
        # Scientific topics and concepts
        self.science_topics = {
            "physics": [
                "gravity", "electricity", "magnetism", "light", "sound", "motion", 
                "energy", "force", "momentum", "waves", "quantum mechanics", "relativity"
            ],
            "chemistry": [
                "atoms", "molecules", "chemical reactions", "elements", "periodic table",
                "acids", "bases", "oxidation", "reduction", "catalysts", "polymers"
            ],
            "biology": [
                "cells", "DNA", "evolution", "ecosystems", "photosynthesis", "respiration",
                "genetics", "adaptation", "biodiversity", "metabolism", "homeostasis"
            ],
            "earth_science": [
                "climate", "weather", "geology", "volcanoes", "earthquakes", "oceans",
                "atmosphere", "plate tectonics", "minerals", "fossils", "water cycle"
            ],
            "astronomy": [
                "planets", "stars", "galaxies", "black holes", "solar system", "moon",
                "sun", "space", "universe", "cosmic radiation", "dark matter"
            ],
            "technology": [
                "computers", "robotics", "artificial intelligence", "nanotechnology",
                "renewable energy", "space exploration", "medical technology"
            ]
        }
        
        # Common scientific misconceptions to fact-check
        self.misconceptions = [
            "Earth is flat",
            "vaccines cause autism", 
            "climate change is a hoax",
            "evolution is just a theory",
            "humans only use 10% of their brain",
            "lightning never strikes the same place twice",
            "the Great Wall of China is visible from space",
            "sharks don't get cancer",
            "bats are blind",
            "goldfish have a 3-second memory",
            "chameleons change color to blend in",
            "different parts of your tongue taste different flavors",
            "humans evolved from monkeys",
            "the Moon landing was faked",
            "GMOs are harmful to health"
        ]
        
        # Bill Nye characteristic phrases and expressions
        self.bill_nye_phrases = [
            "Science rules!",
            "Consider the following!",
            "Isn't that incredible?",
            "The evidence is overwhelming!",
            "That's not how science works!",
            "Let's think about this scientifically!",
            "Here's the deal!",
            "Get ready for this!",
            "This is amazing!",
            "Science is awesome!",
            "Now that's what I call science!",
            "The science is clear!",
            "That's a great question!",
            "Here's what's happening!",
            "Let me break this down for you!"
        ]

    def generate_science_question(self, topic: str, concept: str) -> Tuple[str, str]:
        """Generate a dynamic science question and Bill Nye-style answer."""
        
        # Question templates
        question_templates = [
            f"What is {concept}?",
            f"How does {concept} work?",
            f"Why is {concept} important?",
            f"Can you explain {concept}?",
            f"What do you know about {concept}?",
            f"How do we study {concept}?",
            f"What makes {concept} so fascinating?",
            f"Tell me about {concept}.",
            f"What's the science behind {concept}?",
            f"How does {concept} affect our daily lives?"
        ]
        
        question = random.choice(question_templates)
        answer = self._generate_science_answer(topic, concept)
        
        return question, answer

    def _generate_science_answer(self, topic: str, concept: str) -> str:
        """Generate a Bill Nye-style answer for a science concept."""
        
        # Get a random Bill Nye phrase
        closing_phrase = random.choice(self.bill_nye_phrases)
        
        # Topic-specific explanations
        explanations = {
            "physics": {
                "gravity": f"{concept} is the fundamental force that attracts all objects with mass toward each other! The more mass an object has, the stronger its gravitational pull. That's why we stay on Earth - it's massive! And that's why the Moon orbits Earth, and Earth orbits the Sun. It's all about mass and distance. {closing_phrase}",
                "electricity": f"{concept} is the flow of electric charge through a conductor! Think of it like water flowing through a pipe - the electrons are the water, and the wire is the pipe. When we flip a switch, we're creating a path for electrons to flow, and that's what powers our lights and devices. {closing_phrase}",
                "light": f"{concept} is electromagnetic radiation that our eyes can detect! It travels in waves at an incredible speed - about 186,000 miles per second! Light behaves both as a wave and as a particle, which is one of the most fascinating aspects of physics. Different colors of light have different wavelengths. {closing_phrase}"
            },
            "chemistry": {
                "atoms": f"{concept} are the basic building blocks of all matter! Everything around us - you, me, this computer - is made of atoms. Atoms have a nucleus with protons and neutrons, and electrons orbiting around it. Different combinations of atoms create all the different elements and compounds in the universe. {closing_phrase}",
                "chemical reactions": f"{concept} happen when atoms rearrange to form new substances! It's like a molecular dance where atoms trade partners. The atoms themselves don't change, but how they're connected does. This is how we get everything from rust on metal to the energy in our bodies. {closing_phrase}"
            },
            "biology": {
                "cells": f"{concept} are the smallest units of life! Every living thing is made of cells - from single-celled bacteria to complex organisms like humans with trillions of cells. Cells are like tiny factories that carry out all the processes needed for life. They're incredibly complex and efficient. {closing_phrase}",
                "evolution": f"{concept} is the process by which species change over time through natural selection! It's the foundation of modern biology. Organisms with traits that help them survive and reproduce pass those traits to their offspring. Over millions of years, this leads to the incredible diversity of life we see today. {closing_phrase}",
                "DNA": f"{concept} is like the instruction manual for life! It contains all the genetic information needed to build and maintain an organism. It's a double helix structure made of four chemical bases that pair up in specific ways. Every living thing on Earth uses the same genetic code - that's how related we all are! {closing_phrase}"
            },
            "earth_science": {
                "climate": f"{concept} refers to long-term weather patterns in a particular area! It's different from weather, which changes day to day. Climate is influenced by many factors including latitude, elevation, ocean currents, and atmospheric composition. Understanding climate helps us predict weather patterns and understand environmental changes. {closing_phrase}",
                "volcanoes": f"{concept} are openings in Earth's crust where molten rock, ash, and gases can escape! They're formed by the movement of tectonic plates and the heat from Earth's interior. Volcanoes can be destructive, but they also create new land and enrich soil with minerals. They're a reminder of how dynamic our planet is. {closing_phrase}"
            },
            "astronomy": {
                "planets": f"{concept} are large celestial bodies that orbit stars! In our solar system, we have eight planets, each with unique characteristics. They're divided into rocky planets like Earth and gas giants like Jupiter. Planets form from the same material as their parent star, but they don't have enough mass to ignite nuclear fusion. {closing_phrase}",
                "black holes": f"{concept} are regions of space where gravity is so strong that nothing, not even light, can escape! They form when massive stars collapse at the end of their lives. Despite their name, black holes aren't empty - they contain enormous amounts of mass compressed into an incredibly small space. {closing_phrase}"
            },
            "technology": {
                "computers": f"{concept} are machines that process information using binary code - ones and zeros! They can perform calculations, store data, and execute programs at incredible speeds. Modern computers are based on transistors that can switch between on and off states millions of times per second. {closing_phrase}",
                "artificial intelligence": f"{concept} is the development of computer systems that can perform tasks that typically require human intelligence! This includes learning, reasoning, problem-solving, and perception. AI systems learn from data and can improve their performance over time. It's one of the most exciting areas of technology today. {closing_phrase}"
            }
        }
        
        # Try to get a specific explanation, or generate a generic one
        if topic in explanations and concept in explanations[topic]:
            return explanations[topic][concept]
        else:
            return f"{concept} is a fascinating aspect of {topic}! It's one of those scientific concepts that shows us how amazing and interconnected our universe really is. The more we learn about it, the more we realize how much there is still to discover. {closing_phrase}"

    def generate_fact_check_question(self, misconception: str) -> Tuple[str, str]:
        """Generate a fact-checking question and response."""
        
        question_templates = [
            f"Is it true that {misconception.lower()}?",
            f"I heard that {misconception.lower()}. Is that correct?",
            f"Can you fact-check this: {misconception.lower()}?",
            f"What's the truth about {misconception.lower()}?",
            f"Is {misconception.lower()} a myth or fact?",
            f"Please debunk this claim: {misconception.lower()}."
        ]
        
        question = random.choice(question_templates)
        answer = self._generate_fact_check_answer(misconception)
        
        return question, answer

    def _generate_fact_check_answer(self, misconception: str) -> str:
        """Generate a Bill Nye-style fact-checking response."""
        
        opening_phrase = random.choice(self.bill_nye_phrases)
        
        # Specific fact-checking responses
        fact_checks = {
            "Earth is flat": f"That's not how science works! The Earth is definitely not flat! The evidence is overwhelming - from satellite images to the way ships disappear over the horizon. We can see the Earth's curvature from space, and gravity pulls everything toward the center. The flat Earth idea is not supported by any scientific evidence. {opening_phrase}",
            "vaccines cause autism": f"No, vaccines do not cause autism! This myth has been thoroughly debunked by numerous scientific studies. The original study that suggested this link was retracted and the author lost his medical license. Vaccines are safe and save millions of lives. The science is clear on this! {opening_phrase}",
            "climate change is a hoax": f"Climate change is absolutely real and primarily caused by human activities! The evidence comes from thousands of studies, temperature records, ice core samples, and satellite data. The scientific consensus is overwhelming - over 97% of climate scientists agree. We need to take action to protect our planet. {opening_phrase}",
            "evolution is just a theory": f"Evolution is both a fact and a theory! In science, a theory is a well-supported explanation of facts. Evolution is supported by mountains of evidence from fossils, genetics, anatomy, and direct observation. It's one of the most robust scientific theories we have. {opening_phrase}",
            "humans only use 10% of their brain": f"That's a complete myth! We use virtually all of our brain all the time. Different parts of the brain are active during different activities, but there's no unused 90% sitting idle. This myth probably comes from a misunderstanding of how brain imaging works. {opening_phrase}",
            "lightning never strikes the same place twice": f"Actually, lightning often strikes the same place multiple times! Tall structures like the Empire State Building get hit dozens of times per year. Lightning is attracted to tall, pointed objects, so if you're the tallest thing around, you're likely to get hit again. {opening_phrase}"
        }
        
        if misconception in fact_checks:
            return fact_checks[misconception]
        else:
            return f"That's a common misconception, but it's not scientifically accurate! Let me set the record straight with the facts. The scientific evidence clearly shows that {misconception.lower()} is not true. It's important to rely on peer-reviewed scientific research rather than myths and misinformation. {opening_phrase}"

    def generate_dataset(self, num_general: int = 50, num_fact_check: int = 30) -> Tuple[List[Dict], List[Dict]]:
        """Generate comprehensive training datasets."""
        
        general_data = []
        fact_check_data = []
        
        # Generate general science questions
        for _ in range(num_general):
            topic = random.choice(list(self.science_topics.keys()))
            concept = random.choice(self.science_topics[topic])
            
            question, answer = self.generate_science_question(topic, concept)
            
            general_data.append({
                "instruction": "You are Bill Nye, the Science Guy. Answer the following question in your characteristic enthusiastic and educational style.",
                "input": question,
                "output": answer
            })
        
        # Generate fact-checking questions
        for _ in range(num_fact_check):
            misconception = random.choice(self.misconceptions)
            question, answer = self.generate_fact_check_question(misconception)
            
            fact_check_data.append({
                "instruction": "You are Bill Nye, the Science Guy. Answer the following question in your characteristic enthusiastic and educational style.",
                "input": question,
                "output": answer
            })
        
        return general_data, fact_check_data

    def save_dataset(self, data: List[Dict], filename: str):
        """Save dataset to JSON file."""
        os.makedirs("data", exist_ok=True)
        with open(f"data/{filename}", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Dataset saved to data/{filename} ({len(data)} examples)")

    def create_all_datasets(self):
        """Create and save all training datasets."""
        print("🚀 Creating dynamic Bill Nye training datasets...")
        
        # Generate datasets with reasonable sizes for training
        general_data, fact_check_data = self.generate_dataset(num_general=50, num_fact_check=30)
        
        # Save individual datasets
        self.save_dataset(general_data, "bill_nye_general.json")
        self.save_dataset(fact_check_data, "bill_nye_fact_checking.json")
        
        # Combine all datasets
        combined_data = general_data + fact_check_data
        self.save_dataset(combined_data, "bill_nye_combined.json")
        
        print(f"🎉 Successfully created {len(combined_data)} diverse training examples!")
        print(f"   - General science: {len(general_data)} examples")
        print(f"   - Fact-checking: {len(fact_check_data)} examples")
        
        return combined_data

def main():
    """Main function to create enhanced Bill Nye datasets."""
    generator = BillNyeDatasetGenerator()
    generator.create_all_datasets()

if __name__ == "__main__":
    main()
