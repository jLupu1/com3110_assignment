
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting): 
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.num_terms = self.compute_number_of_terms()
        
    
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
        return list(range(1,11))
    
    
    def doc_term_matrix(self):
        matrix = [[0 for _ in range(self.num_terms)] 
                  for _ in range(self.num_docs)]
        
        
        for term_index, term in enumerate(self.index):
            for doc in self.index[term]:
                # doc-1 as docId startes at 1
                matrix[doc-1][term_index] = self.index[term][doc]
        return matrix
    
    
# turn the query into vector

        
                
        
    

