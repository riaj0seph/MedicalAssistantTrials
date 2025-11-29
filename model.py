import json
import math
import networkx as nx
import numpy as np
import re
import subprocess
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from collections import defaultdict

class MedicalAssistant:
    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2', alpha=0.7):
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.diseases_data = self._load_data(json_file_path)
        
        # Build graph
        self.G, self.symptom_idf = self._build_graph(self.diseases_data)
        
        # Initialize NLP components
        self.symptom_embeddings = self._precompute_symptom_embeddings()
        self.bm25_model, self.all_symptom_list, self.symptom_to_disease = self._build_bm25_index()
        
        # Conversation state - FIXED: track questions and responses properly
        self.matched_symptoms = {}
        self.asked_questions = []  # List to maintain order
        self.question_responses = {}  # Track responses to specific questions

    def _load_data(self, json_file_path):
        with open(json_file_path, 'r') as f:
            return json.load(f)

    def _build_graph(self, data):
        G = nx.Graph()
        symptom_df_count = {}
        total_diseases = len(data)

        for entry in data:
            disease = entry["disease"]
            G.add_node(disease, type="disease", precautions=entry["precautions"])
            for symptom in entry["symptoms"]:
                symptom_clean = symptom.lower().strip()
                G.add_node(symptom_clean, type="symptom")
                symptom_df_count[symptom_clean] = symptom_df_count.get(symptom_clean, 0) + 1

        symptom_idf = {symptom: math.log((total_diseases + 1) / (count + 1)) + 1
                      for symptom, count in symptom_df_count.items()}

        for entry in data:
            disease = entry["disease"]
            for symptom in entry["symptoms"]:
                symptom_clean = symptom.lower().strip()
                G.add_edge(symptom_clean, disease, idf=symptom_idf[symptom_clean])

        return G, symptom_idf

    def _precompute_symptom_embeddings(self):
        symptom_embeddings = {}
        for disease_info in self.diseases_data:
            disease = disease_info['disease']
            symptoms = disease_info['symptoms']
            embeddings = self.model.encode(symptoms)
            symptom_embeddings[disease] = {
                'symptoms': symptoms,
                'embeddings': embeddings
            }
        return symptom_embeddings

    def _build_bm25_index(self):
        all_symptoms = []
        symptom_to_disease = {}
        for disease_info in self.diseases_data:
            disease = disease_info['disease']
            for symptom in disease_info['symptoms']:
                all_symptoms.append(symptom.lower().split())
                symptom_to_disease[symptom] = disease
        bm25 = BM25Okapi(all_symptoms)
        return bm25, all_symptoms, symptom_to_disease

    def _hybrid_similarity(self, query, embeddings, symptoms):
        query_embedding = self.model.encode([query])[0]
        dense_scores = cosine_similarity([query_embedding], embeddings)[0]

        tokenized_query = query.lower().split()
        bm25_scores = []
        for symptom in symptoms:
            try:
                symptom_tokens = symptom.lower().split()
                if symptom_tokens in self.all_symptom_list:
                    idx = self.all_symptom_list.index(symptom_tokens)
                    bm25_score = self.bm25_model.get_scores(tokenized_query)[idx]
                else:
                    bm25_score = 0.0
                bm25_scores.append(bm25_score)
            except:
                bm25_scores.append(0.0)

        # Normalize only if we have variation in scores
        if np.ptp(dense_scores) > 0:
            dense_scores = (dense_scores - np.min(dense_scores)) / (np.ptp(dense_scores) + 1e-8)
        else:
            dense_scores = np.ones_like(dense_scores)
            
        if np.ptp(bm25_scores) > 0:
            bm25_scores = (np.array(bm25_scores) - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)
        else:
            bm25_scores = np.ones_like(bm25_scores)

        hybrid_scores = self.alpha * dense_scores + (1 - self.alpha) * bm25_scores
        return hybrid_scores

    def extract_symptoms_from_query(self, query, similarity_threshold=0.6):  # Increased threshold
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [query]

        matched_symptoms = {}
        
        for sentence in sentences:
            #  Use a more focused approach - prioritize direct matches
            for disease, symptom_data in self.symptom_embeddings.items():
                symptoms = symptom_data['symptoms']
                embeddings = symptom_data['embeddings']
                similarities = self._hybrid_similarity(sentence, embeddings, symptoms)

                for symptom, similarity in zip(symptoms, similarities):
                    symptom_key = symptom.lower().strip()
                    
                    #  Only add if it's a strong match OR it's a direct keyword
                    is_direct_match = any(word in sentence.lower() for word in symptom_key.split())
                    
                    if similarity >= similarity_threshold or (is_direct_match and similarity >= 0.4):
                        if symptom_key not in matched_symptoms or matched_symptoms[symptom_key] < similarity:
                            matched_symptoms[symptom_key] = min(similarity, 0.95)

        #  Limit the number of matched symptoms to avoid over-matching
        if len(matched_symptoms) > 10:
            # Keep only the top matches
            top_matches = sorted(matched_symptoms.items(), key=lambda x: x[1], reverse=True)[:10]
            matched_symptoms = dict(top_matches)
            
        return matched_symptoms

    def score_diseases_graph(self, matched_symptoms):
        scores = {}
        disease_coverage = {}
        
        for symptom, similarity in matched_symptoms.items():
            if symptom not in self.G:
                continue
                
            for disease in self.G.neighbors(symptom):
                if self.G.nodes[disease]['type'] != 'disease':
                    continue
                    
                disease_symptoms = [n for n in self.G.neighbors(disease) 
                                  if self.G.nodes[n]['type'] == 'symptom']
                disease_symptom_count = len(disease_symptoms)
                idf_weight = self.G.get_edge_data(symptom, disease)["idf"]
                
                coverage_ratio = 1 / disease_symptom_count
                contrib = similarity * idf_weight * coverage_ratio
                scores[disease] = scores.get(disease, 0) + contrib
                
                if disease not in disease_coverage:
                    disease_coverage[disease] = {'matched': set(), 'total': disease_symptom_count}
                disease_coverage[disease]['matched'].add(symptom)

        # Better coverage bonus calculation
        for disease in scores:
            if disease in disease_coverage:
                coverage_ratio = len(disease_coverage[disease]['matched']) / disease_coverage[disease]['total']
                # Only give bonus for good coverage (above 30%)
                if coverage_ratio > 0.3:
                    scores[disease] *= (1 + 0.3 * coverage_ratio)

        if not scores:
            return {}
            
        disease_names = list(scores.keys())
        score_values = np.array(list(scores.values()))
        
        exp_scores = np.exp(score_values - np.max(score_values))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return {disease: {
            "score": float(scores[disease]),
            "conf": float(prob),
            "coverage": (len(disease_coverage[disease]['matched']), disease_coverage[disease]['total'])
        } for disease, prob in zip(disease_names, probabilities)}

    def llm_question(self, symptom):
        prompt = f"""Ask exactly one brief YES/NO question about this medical symptom: "{symptom}". Keep it very short and clear."""
        
        try:
            result = subprocess.run(
                ["ollama", "run", "gemma3:1b"],
                input=prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",
                timeout=10
            )
            question = result.stdout.strip()
            question = re.sub(r'^[^a-zA-Z]*', '', question)
            return question if question else f"Do you have {symptom}?"
        except:
            return f"Do you have {symptom}?"

    def select_optimal_question(self, disease_candidates):
        if not disease_candidates:
            return None
            
        top_diseases = sorted(disease_candidates.items(), 
                            key=lambda x: x[1]["conf"], reverse=True)[:3]
        
        candidate_symptoms = defaultdict(float)
        
        for disease, info in top_diseases:
            disease_symptoms = [n for n in self.G.neighbors(disease) 
                              if self.G.nodes[n]['type'] == 'symptom']
            
            for symptom in disease_symptoms:
                # Check if we haven't asked this exact question before
                if (symptom not in [q[0] for q in self.asked_questions] and 
                    symptom not in self.matched_symptoms):
                    
                    idf_weight = self.G.get_edge_data(symptom, disease)["idf"]
                    disease_conf = info["conf"]
                    candidate_symptoms[symptom] += disease_conf * idf_weight
        
        if not candidate_symptoms:
            return None
            
        # Return the symptom, not the score
        best_symptom = max(candidate_symptoms.items(), key=lambda x: x[1])[0]
        return best_symptom

    def diagnose(self, query, max_questions=10):
        print(f"Query: {query}")
        print("Extracting symptoms...")
        
        # Reset conversation state
        self.matched_symptoms = {}
        self.asked_questions = []
        self.question_responses = {}
        
        # Step 1: NLP to extract base symptoms
        self.matched_symptoms = self.extract_symptoms_from_query(query)
        print(f"Initial matched symptoms: {list(self.matched_symptoms.keys())}")
        
        # Step 2: Ask questions to refine predictions
        for i in range(max_questions):
            # Find matches
            scores = self.score_diseases_graph(self.matched_symptoms)
            
            if not scores:
                print("No diseases match the symptoms.")
                return scores
            
            # Check if we're confident enough
            top_disease = max(scores.items(), key=lambda x: x[1]["conf"])
            if top_disease[1]["conf"] >= 0.6:
                print(f"Confidence threshold reached after {i} questions.")
                break
            
            # Ask next question
            next_question = self.select_optimal_question(scores)
            if not next_question:
                print("No more relevant questions to ask.")
                break
                
            question_text = self.llm_question(next_question)
            
            # Track the question properly
            self.asked_questions.append((next_question, question_text))
            
            print(f"\nQuestion {i+1}: {question_text}")
            answer = input("You (yes/no): ").lower().strip()
            
            # Store the response for this specific question
            self.question_responses[next_question] = answer
            
            if answer in ['yes', 'y', 'yeah', 'yep']:
                self.matched_symptoms[next_question] = 0.9A
            elif answer in ['no', 'n', 'nope']:
                self.matched_symptoms[next_question] = 0.1
            else:
                self.matched_symptoms[next_question] = 0.5
        
        # Step 3: Produce final results
        final_scores = self.score_diseases_graph(self.matched_symptoms)
        self._print_results(final_scores)
        return final_scores

    def _print_results(self, scores):
        if not scores:
            print("No diagnosis could be made.")
            return
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["conf"], reverse=True)
        top_10 = sorted_scores[:10]
        top_3 = sorted_scores[:3]
        final = sorted_scores[0]

        print("\n Matched Symptoms:")
        for symptom, similarity in self.matched_symptoms.items():
            print(f"- {symptom} (confidence: {similarity:.2f})")

        print("\n Questions Asked:")
        #Use the stored responses instead of matched_symptoms
        for symptom, question_text in self.asked_questions:
            response = self.question_responses.get(symptom, "Not answered")
            print(f"- {question_text} → {response}")

        print("\n Top 10 Predictions:")
        for i, (disease, info) in enumerate(top_10, 1):
            matches, total = info["coverage"]
            print(f"{i:2d}. {disease:25} — conf: {info['conf']:.3f} | coverage: {matches}/{total}")

        print("\n Top 3 Finalists:")
        for i, (disease, info) in enumerate(top_3, 1):
            matches, total = info["coverage"]
            print(f"{i}. {disease} (confidence: {info['conf']:.3f}, coverage: {matches}/{total})")

        final_d, final_info = final
        matches, total = final_info["coverage"]
        print(f"\n Final Prediction: {final_d} (confidence: {final_info['conf']:.3f})")
        
        print("\n🛡 Recommended Precautions:")
        precautions = self.G.nodes[final_d].get('precautions', [])
        if precautions:
            for precaution in precautions:
                print(f"• {precaution}")
        else:
            print("• No specific precautions available")


# Usage
if __name__ == "__main__":
    assistant = MedicalAssistant('medical_dataset.json')
    
    # Take query -> process -> diagnose -> show results
    query = input("Describe your symptoms: ")
    assistant.diagnose(query)