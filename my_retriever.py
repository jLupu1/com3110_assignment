import math


class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
        if term_weighting == 'tf':
            self.doc_term_matrix = self.compute_doc_term_matrix_tf()
        elif term_weighting == "tfidf":
            self.doc_term_matrix = self.compute_doc_term_matrix_tfidf()
        else:
            self.doc_term_matrix = self.compute_doc_term_matrix_binary()
            
            
    
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    

    # Create matrix for binary weighting
    def compute_doc_term_matrix_binary(self):
        matrix = [{} for _ in range(self.num_docs)]
        
        for term,doc_counts in self.index.items():
            for doc_id in doc_counts:
                # doc_id - 1 for 0-based list index
                matrix[doc_id-1][term] = 1
        return matrix

    # Creates matrix each row is a doc with dict {term : count}
    # tf weighting - counting the freq of each term for each doc
    def compute_doc_term_matrix_tf(self):
        matrix = [{} for _ in range(self.num_docs)]
        
        # Iterate through each unique term in the index
        for term, doc_counts in self.index.items():
            for doc_id, count in doc_counts.items():
                # doc_id - 1 for 0-based list index
                # Store the term frequency
                matrix[doc_id-1][term] = count
        return matrix
    
    # Creates matrix each row is a doc with dict {term : count}
    # tf weighting - counting the freq of each term for each doc
    def compute_doc_term_matrix_tfidf(self):    
        matrix = [{} for _ in range(self.num_docs)]
        
        # Iterate through each unique term in the index
        for term, doc_counts in self.index.items():
            # Get Document Frequency (DF_w)
            df_w = len(doc_counts)
            
            idf = math.log10(((self.num_docs + 1) / (df_w + 1))) + 1
            
            for doc_id, count in doc_counts.items():
                
                tf = 1 + math.log10(count)
                
                # doc_id - 1 for 0-based list index
                matrix[doc_id-1][term] =  tf * idf
                
        return matrix
    
    
    
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        if self.term_weighting == "tfidf":
            query_matrix = self.query_matrix_tfidf(query)
        elif self.term_weighting == "tf":
            query_matrix = self.query_matrix_tf(query)
        else:
            query_matrix = self.query_matrix_binary(query)
            
        similarity_data = self.calculate_sim(query_matrix)
        sorted_items = sorted(similarity_data.items(), 
                              key=lambda item: item[1], reverse=True)
        return [item[0] for item in sorted_items[:10]]
    
    
    # turn the query into vector    
    # Query is passed as tuple (id,list(terms))
    
    
    def query_matrix_tf(self,query):
        matrix = {}
        for term in query:
            matrix[term] = matrix.get(term,0) + 1
        return matrix
    
    def query_matrix_binary(self,query):
        matrix = {}
        for term in query:
            matrix[term] = 1
        return matrix
    
    def query_matrix_tfidf(self,query):
        # Two pass solution, 1st counts tf for each term, 2nd calculates tfidf 
        
        matrix = {}
        # Counts freq (tf)
        for term in query:
            matrix[term] = matrix.get(term, 0) + 1
        
        for term, count in matrix.items():
            df_w = len(self.index.get(term, {}))
    
            idf = math.log10(((self.num_docs + 1) / (df_w + 1)))
            tf = 1 + math.log10(count)
            matrix[term] = tf * idf
            
        return matrix
         
        

    def calculate_sim(self, query_matrix):
        sim_scores = {}
    
        sqrt_sum_q2 = sum([w * w for w in query_matrix.values()])
        query_magnitude = math.sqrt(sqrt_sum_q2)
        
        if query_magnitude == 0:
            return {}
        
        for index, doc_vector in enumerate(self.doc_term_matrix):
            sum_qd = 0
            sqrt_sum_d2 = 0 
            
            for term, d_weight in doc_vector.items():
                sqrt_sum_d2 += (d_weight * d_weight)
                q_weight = query_matrix.get(term, 0)
                sum_qd += (q_weight * d_weight)    
                
            doc_magnitude = math.sqrt(sqrt_sum_d2)
    
            # Check for overlap
            if sum_qd > 0:
                denominator = query_magnitude * doc_magnitude
                score = sum_qd / denominator
                sim_scores[index + 1] = score
            else:
                pass 
                
        return sim_scores
    

