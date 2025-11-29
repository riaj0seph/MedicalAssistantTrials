import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

class DiseaseRAGSystem:
    def __init__(self, json_file_path, model_name='all-MiniLM-L6-v2', alpha=0.7):
        """
        Initialize the RAG system with disease data and embedding model.
        alpha: Weight for dense vs BM25 hybrid fusion. (1.0 = pure dense, 0.0 = pure BM25)
        """
        self.model = SentenceTransformer(model_name)
        self.alpha = alpha
        self.diseases_data = self._load_data(json_file_path)
        self.symptom_embeddings = self._precompute_symptom_embeddings()
        self.bm25_model, self.all_symptom_list, self.symptom_to_disease = self._build_bm25_index()

    def _load_data(self, json_file_path):
        """Load disease data from JSON file."""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        return data

    def _precompute_symptom_embeddings(self):
        """Precompute embeddings for all official symptoms."""
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
        """Build a global BM25 index for all symptoms across diseases."""
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
        """Combine cosine (dense) and BM25 (sparse) similarities."""
        # Dense part
        query_embedding = self.model.encode([query])[0]
        dense_scores = cosine_similarity([query_embedding], embeddings)[0]

        # Sparse (BM25) part
        tokenized_query = query.lower().split()
        bm25_scores = []
        for symptom in symptoms:
            bm25_score = self.bm25_model.get_scores(tokenized_query)[
                self.all_symptom_list.index(symptom.lower().split())
            ]
            bm25_scores.append(bm25_score)

        # Normalize both scores
        dense_scores = (dense_scores - np.min(dense_scores)) / (np.ptp(dense_scores) + 1e-8)
        bm25_scores = (np.array(bm25_scores) - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)

        # Hybrid fusion
        hybrid_scores = self.alpha * dense_scores + (1 - self.alpha) * bm25_scores
        return hybrid_scores

    def extract_symptoms_by_sentence(self, query, similarity_threshold=0.45):
        """Extract symptoms by splitting query into sentences and using hybrid matching."""
        sentences = re.split(r'[.!?]+', query)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [query]

        matched_symptoms = {}
        for sentence in sentences:
            for disease, symptom_data in self.symptom_embeddings.items():
                symptoms = symptom_data['symptoms']
                embeddings = symptom_data['embeddings']

                # Hybrid similarity fusion
                similarities = self._hybrid_similarity(sentence, embeddings, symptoms)

                for symptom, similarity in zip(symptoms, similarities):
                    if similarity >= similarity_threshold:
                        key = (symptom, disease)
                        if key not in matched_symptoms or matched_symptoms[key] < similarity:
                            matched_symptoms[key] = similarity

        result = [(symptom, score, disease) for (symptom, disease), score in matched_symptoms.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def calculate_disease_scores(self, matched_symptoms):
        """Calculate scores for each disease based on matched symptoms using confidence-weighted softmax."""
        if not matched_symptoms:
            return {}

        disease_matches = {}
        for symptom, similarity, disease in matched_symptoms:
            disease_matches.setdefault(disease, []).append((symptom, similarity))

        total_matched_symptoms = len(set([s[0] for s in matched_symptoms]))
        disease_scores = {}

        for disease, matches in disease_matches.items():
            num_disease_symptoms = len(self.symptom_embeddings[disease]['symptoms'])
            score = 0.0
            for symptom, similarity in matches:
                base_score = (1.0 / num_disease_symptoms) + (1.0 / total_matched_symptoms)
                score += base_score * similarity
            disease_scores[disease] = {
                'score': score,
                'matched_symptoms': [s[0] for s in matches],
                'num_matches': len(matches),
                'total_symptoms': num_disease_symptoms
            }

        # Confidence-normalized softmax
        scores = np.array([info['score'] for info in disease_scores.values()])
        exp_scores = np.exp(scores - np.max(scores))
        softmax_scores = exp_scores / np.sum(exp_scores)
        for i, (disease, info) in enumerate(disease_scores.items()):
            disease_scores[disease]['confidence'] = float(softmax_scores[i])

        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1]['confidence'], reverse=True)
        return dict(sorted_diseases)

    def get_disease_info(self, disease_name):
        for disease_info in self.diseases_data:
            if disease_info['disease'] == disease_name:
                return disease_info
        return None

    def diagnose(self, query, top_k=3, similarity_threshold=0.45):
        matched_symptoms = self.extract_symptoms_by_sentence(query, similarity_threshold)
        if not matched_symptoms:
            return {
                'status': 'no_match',
                'message': 'Could not identify any symptoms from the query.',
                'matched_symptoms': [],
                'top_diseases': []
            }

        disease_scores = self.calculate_disease_scores(matched_symptoms)
        top_diseases = []
        for i, (disease, score_info) in enumerate(disease_scores.items()):
            if i >= top_k:
                break
            disease_info = self.get_disease_info(disease)
            top_diseases.append({
                'disease': disease,
                'score': score_info['score'],
                'confidence': score_info['confidence'],
                'matched_symptoms': score_info['matched_symptoms'],
                'num_matches': score_info['num_matches'],
                'total_symptoms': score_info['total_symptoms'],
                'all_symptoms': disease_info['symptoms'],
                'precautions': disease_info['precautions']
            })

        return {
            'status': 'success',
            'query': query,
            'matched_symptoms': list(set([s[0] for s in matched_symptoms])),
            'top_diseases': top_diseases,
            'best_match': top_diseases[0] if top_diseases else None
        }

    def generate_response(self, diagnosis_result):
        if diagnosis_result['status'] == 'no_match':
            return diagnosis_result['message']

        best_match = diagnosis_result['best_match']
        response = f"Based on your symptoms, you may have **{best_match['disease']}**.\n\n"
        response += f"**Matched symptoms ({best_match['num_matches']}/{best_match['total_symptoms']}):**\n"
        for symptom in best_match['matched_symptoms']:
            response += f"- {symptom}\n"

        response += f"\n**Recommended precautions:**\n"
        for precaution in best_match['precautions']:
            response += f"- {precaution}\n"

        if len(diagnosis_result['top_diseases']) > 1:
            response += f"\n**Other possible conditions:**\n"
            for disease in diagnosis_result['top_diseases'][1:]:
                response += f"- {disease['disease']} (confidence: {disease['confidence']:.3f})\n"

        response += "\n*Note: This is an automated assessment. Please consult a healthcare professional for confirmation.*"
        return response


# Example usage
if __name__ == "__main__":
    print("Initializing Hybrid Disease RAG System...")
    rag = DiseaseRAGSystem('medical_dataset.json', alpha=0.7)

    query = "I have rashes on my shoulder which itches a lot. There are also dark patches on my neck region."
    print(f"\nQuery: {query}\nDiagnosing...\n")

    result = rag.diagnose(query, top_k=3, similarity_threshold=0.45)
    response = rag.generate_response(result)

    print(response)
    print("\n" + "="*50)
    print("DETAILED RESULTS")
    print("="*50)
    print(f"\nMatched symptoms: {result['matched_symptoms']}")
    for disease in result['top_diseases']:
        print(f"\n{disease['disease']}:")
        print(f"  Score: {disease['score']:.4f}")
        print(f"  Confidence: {disease['confidence']:.3f}")
        print(f"  Matched: {disease['num_matches']}/{disease['total_symptoms']} symptoms")
        print(f"  Matched symptoms: {disease['matched_symptoms']}")