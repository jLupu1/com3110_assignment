import math
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.num_terms = self.compute_number_of_terms()
        self.term_index = self.term_index_dict()
        self.doc_term_matrix = self.compute_doc_term_matrix()
        print(self.num_docs)
        
    
    # Maps each term in self.index to an index
    def term_index_dict(self):
        res_dict = {}
        for index, term in enumerate(self.index):
            res_dict[term] = index
        return res_dict
            
    
    
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    
    
    def compute_number_of_terms(self):
        self.terms = set() 
        for term in self.index:
            self.terms.add(term)
        return len(self.terms)
    

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).​ Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        self.calculate_sim(self.query_matrix(query))
        return list(range(1,11))
    
    
    def compute_doc_term_matrix(self):
        matrix = [[0 for _ in range(self.num_terms)] 
                  for _ in range(self.num_docs)]
        
        
        for term_index, term in enumerate(self.index):
            for doc in self.index[term]:
                # doc-1 as docId startes at 1
                matrix[doc-1][term_index] = self.index[term][doc]
        return matrix
    
    
    # turn the query into vector
    
    # Query is passed as tuple (id,list(terms))
    def query_matrix(self, query):
        matrix = [0 for _ in range(self.num_terms)]
         
        for term in query:
            index = self.term_index.get(term,0)
            matrix[index] += 1
        return matrix


    def calculate_sim(self,query_matrix):
        # id(index + 1) : score
        sim_scores = {}
        sum_qd = 0
        sqrt_sum_d2 = 0
        
        
        for index,doc in enumerate(self.doc_term_matrix):
            sum_qd = 0
            sqrt_sum_d2 = 0
            for i in range(len(doc)):
                sum_qd += (query_matrix[i] * doc[i])
                sqrt_sum_d2 += (doc[i] * doc[i])
            sim_scores[index+1] = (sum_qd) / math.sqrt(sqrt_sum_d2)
            
        return sim_scores
            
    
    
    
        


