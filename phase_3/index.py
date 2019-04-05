#Python 3.0
import re
import os
import collections
import time
import math
from scipy import spatial
from sklearn.preprocessing import normalize 
from collections import Counter
from numpy import dot
import numpy as np
from numpy.linalg import norm
import ml_metrics as ml

class index:
        def __init__(self,path):

            self.path = path 
            
            # {docid: {termID:freq, termID:freq, ...} ,docid: {}, ...} 
            self.doc_to_term = {}

            # {termID : [idf, (docID, tfTD), (docID, tfID) , ...] , termID: [..] , ...}
            self.posting_list = {}

            # {term : termID, term: termID, ...}
            self.term_to_termID = {}

            # {termID: term, termID:term, ...}
            self.termID_to_term = {}

            # {docID : doc, docID : doc}
            self.doc_to_docID = {}

            # words to ignore from posting list
            self.stop_words = set()
            
            #docid:document vector
            self.document_vectors= {}

        def buildIndex(self):
		#function to read documents from collection, tokenize and build the index with tokens
		# implement additional functionality to support relevance feedback
		#use unique document integer IDs

            start_time = time.time()

            # compile stop word list:
            with open(self.path + 'TIME.STP') as f:
                p = re.compile('[a-z]+', re.IGNORECASE)
                self.stop_words.update(p.findall(f.read().lower()))
            
            #split entire doc on *, into indiv. docs
            with open(self.path + 'TIME.ALL') as f:
                docs = f.read().lower().replace('\n', ' ').split('*')[1:-1]
            
            #removing all extranious spaces
            for i in range(len(docs)):
                docs[i]= ' '.join(docs[i].split())
                
                # set the docID, doc title
                self.doc_to_docID[i] = docs[i][:8] + '.txt'
                # set the docID in doc_to_term
                self.doc_to_term[i] = {}

                #pattern = re.compile('\w+(?:-\w+)+|\w+[^ ]\w+|\w+',re.IGNORECASE)
                pattern = re.compile('\w+',re.IGNORECASE)
        
                tokens = re.findall(pattern, docs[i][27:])
                #tokens = re.split('[\W]', docs[i][27:])
                #now we populate doc_to_term, term_to_termID
                for token in tokens:
                    if token not in self.stop_words:
                        
                        #just dealing with token_to_tokenID
                        if token in self.term_to_termID:
                            termID = self.term_to_termID[token]
                        else:
                            #add token to token_to_tokenID
                            termID = len(self.term_to_termID)
                            self.term_to_termID[token] = termID
                        
                        # populate doc_to_term with all terms per doc
                        if self.term_to_termID[token] in self.doc_to_term[i]:
                            self.doc_to_term[i][termID] += 1
                        else:
                            self.doc_to_term[i][termID] = 1
                
                for termID in self.doc_to_term[i]:
                    if termID not in self.posting_list:
                        tfid = math.log10(len(docs))
                        self.posting_list[termID] = [tfid,(i,self.doc_to_term[i][termID])]
                    else:
                        self.posting_list[termID][0]= math.log10(len(docs)/len(self.posting_list[termID]))
                        self.posting_list[termID].append((i,self.doc_to_term[i][termID]))

            self.termID_to_term = dict([[v,k] for k,v in self.term_to_termID.items()])
            print('Index built in {0}'.format(time.time()-start_time))



        def build_index2(self):
            with open(self.path + 'TIME.STP') as f:
                p = re.compile('\w+', re.IGNORECASE)
                self.stop_words.update(p.findall(f.read().lower()))
                self.stop_words.add('')
            # split entire doc on *, into indiv. docs
            with open(self.path + 'TIME.ALL') as f:
                docs = f.read().lower().replace('\n', ' ').split('*')[1:-1]
            

            # removing all extranious spaces
            for i in range(len(docs)):
                docs[i]= ' '.join(docs[i].split())
                 
                # set the docID, doc title
                self.doc_to_docID[i] = docs[i][:8] + '.txt'
                # set the docID in doc_to_term
                self.doc_to_term[i] = {}

                #pattern = re.compile('\w+(?:-\w+)+|\w+[^ ]\w+|\w+',re.IGNORECASE)
                pattern = re.compile('\w+',re.IGNORECASE)
                tokens = re.findall(pattern, docs[i][27:])
                #tokens = re.split('\W+', docs[i][26:].lower())
                for token in tokens:
                    if token not in self.stop_words:
                        # if not a stop word, then craete a token ID if we have not seen the word 
                        if token in self.term_to_termID:
                            tokenID = self.term_to_termID[token]
                        else:
                            tokenID = len(self.term_to_termID)
                            self.term_to_termID[token] = tokenID
                
                        
                        if tokenID not in self.doc_to_term[i]:
                            self.doc_to_term[i][tokenID] = 1
                        else:
                            self.doc_to_term[i][tokenID] += 1
                
                # now each doc_to_term dict is compiled, so we can populate the global
                # posting list with the terms and their frequency
                for termID in self.doc_to_term[i]:
                    if termID in self.posting_list:
                        self.posting_list[termID].append(tuple((i, self.doc_to_term[i][termID])))
                        tfid = math.log10(len(docs) / (len(self.posting_list[termID])-1))
                        self.posting_list[termID][0] = tfid
                    else:
                        tfid = math.log10(len(docs))
                        self.posting_list[termID] = [tfid, tuple((i,self.doc_to_term[i][termID]))]
            
            self.termID_to_term = dict([[v,k] for k,v in self.term_to_termID.items()])
                

        # exact top-k search
        # return top-k docIDs and their scores
        def exact_search(self, query, k):
            docs_to_score = self.find_all_docs(query) 
            doc_scores = self.score2(query, docs_to_score)
            print(doc_scores[:10])
            if len(doc_scores) > k:
                docs = [i[1] for i in doc_scores[:k]]
                return docs
            else:
                return doc_scores 

   
        

	#function to implement rocchio algorithm
	#pos_feedback - documents deemed to be relevant by the user - list,iterable
	#neg_feedback - documents deemed to be non-relevant by the user - list,iterable
	#Return the new query  terms and their weights
        def rocchio(self, query, pos_feedback, neg_feedback, alpha = 1, beta = 0.75 , gamma = 0.15):
            #print(pos_feedback, neg_feedback)
            try:
                pos_feedback = list(map(int, pos_feedback))
                neg_feedback = list(map(int, neg_feedback))
            
            except:
                print('YOUR FEEDBACK IS GARBAGE MATE\n I need documentIDs to proceed')
                return

            # first i just compute the vocab
            # this may not bbe the fastest way
            # i will refine after a working program
            vocab, q_vector, pos_feedback_vectors, neg_feedback_vectors = self.create_rocchio_vectors(query, 
                                                                            pos_feedback, neg_feedback)
            
            final_vector = q_vector 
            pos_vector_sum = self.sum_vector_list(list(pos_feedback_vectors.values()))
            if len(neg_feedback_vectors) >0:
                neg_vector_sum = self.sum_vector_list(list(neg_feedback_vectors.values()))
            
            # weight vectors
            # and multiply by the number of documents 
            pos_vector_sum = list(np.array(pos_vector_sum) * beta / len(pos_feedback))
            if len(neg_feedback_vectors) >0:
                neg_vector_sum = list(np.array(neg_vector_sum) * gamma * -1 / len(neg_feedback))
                final_vector = self.sum_vector_list([final_vector, pos_vector_sum, neg_vector_sum]) 
            else:
                final_vector = self.sum_vector_list([final_vector, pos_vector_sum])
            weighted_d = {}
            vector_weighted = {}
            for index,val in enumerate(vocab):
                if final_vector[index] > 0:
                    vector_weighted[val] = final_vector[index]
                    weighted_d[self.termID_to_term[val]] = final_vector[index]
            
            
            return vocab,vector_weighted, weighted_d 
       



        # returns all the vectors to perform rocchio  
        def create_rocchio_vectors(self, query, pos_feedback, neg_feedback):
            
            vocab = set()
            for doc in pos_feedback:
                vocab |= set(self.doc_to_term[doc].keys())
            for doc in neg_feedback:
                vocab |= set(self.doc_to_term[doc].keys())
            #incase the top-k docs dont have all terms of the query
            vocab |= set(query.keys())
            #ok now i go through and create all of the vectors
            q_vector = []
            
            #{docID: doc_vector,...}
            pos_doc_vectors = {}
            neg_doc_vectors = {}
            #intiallize doc_vectors
            for i in pos_feedback:
                pos_doc_vectors[i] = []
            for i in neg_feedback:
                neg_doc_vectors[i] = []

            for term in vocab:
                if term in query:
                    q_vector.append(query[term] * self.posting_list[term][0])
                else:
                    q_vector.append(0)
             
                for doc in pos_doc_vectors:
                    if term in self.doc_to_term[doc]:
                        pos_doc_vectors[doc].append(self.doc_to_term[doc][term] * self.posting_list[term][0])
                    else:
                        pos_doc_vectors[doc].append(0)

                for doc in neg_doc_vectors:
                    if term in self.doc_to_term[doc]:
                        neg_doc_vectors[doc].append(self.doc_to_term[doc][term] * self.posting_list[term][0])
                    else:
                        neg_doc_vectors[doc].append(0)


            return vocab, q_vector, pos_doc_vectors, neg_doc_vectors

        '''
        def roc_2(self, query, pos_f, neg_f, alpha = 1, beta = 0.75, gamma = 0.15):

            # create vocab
            vocab = set()
            for i in pos_f+neg_f:
                vocab |= set(self.doc_to_term[i].keys())
            vocab |= set(query.keys())
            
            pos_vectors = [] #list of lists, each sub list is a doc vector
            neg_vectors = []
            
            for doc in pos_f:
                v = []
                for term in vocab:
                    if term in self.doc_to_term[doc]:
                        v.append(
        '''



        #function to print the terms and posting list in the index 
        def print_dict(self):
            for i in self.term_to_termID:
                print(i,self.term_to_termID[i] , self.posting_list[self.term_to_termID[i]])


	# function to print the documents and their document id
        
        def print_doc_list(self):
            for i in self.doc_to_docID:
                print(self.doc_to_docID[i], i)


        #################################
        # Helpers
        #################################
    
        #take raw query(string), parse and return a counter object of termID:frequency
        def filter_query(self, query):
            pattern = re.compile('\w+(?:-\w+)+|\w+[^ ]\w+|\w+',re.IGNORECASE)
           # pattern = re.compile('\w+',re.IGNORECASE)
            tokens = re.findall(pattern, query.lower())
            #tokens = query.lower().split()
            query_ret=[]
            for i in tokens:
                if i in self.term_to_termID:
                    query_ret.append(self.term_to_termID[i])
            query_ret = Counter(query_ret)
            return query_ret 

        #find all the documents with a term matching the query, return a set
        def find_all_docs(self, query):
            docs_to_score = []
            for termID in query:
                for doc in range(1,len(self.posting_list[termID])):
                    docs_to_score.append(self.posting_list[termID][doc][0])
            return set(docs_to_score)


        #score documents with the query
        # return a sorted list of documents, and scores (descending order)
        # query - iterable.. set, list
        # docs - iterable.. set, list
        def score_docs(self, query, docs):
            
            scores = [] # list of (score, docIDs)
            for i in docs:
                score = self.compute_cosine2(query, i)
                scores.append((score,i))

            return sorted(scores, key=lambda k:k[0], reverse=True)

        def score2(self, query, docs):
            scores = []
            for i in docs:
                q_v, d_v = self.compute_q_d_vectors(query,i)
                score = self.compute_cosine2(q_v,d_v)
                scores.append(tuple((score,i)))
            
            return sorted(scores, key=lambda k:k[0], reverse=True)


        #comepute vectors given a query and a doc 
        def compute_q_d_vectors(self, query, docID):
            q_v = []
            d_v = []
            for i in self.doc_to_term[docID]:
                tfidf = (1+math.log10(self.doc_to_term[docID][i])) * self.posting_list[i][0] 
                d_v.append(tfidf)
                if i in query:
                    idf = self.posting_list[i][0]
                    q_v.append(query[i] * idf)
                else:
                    q_v.append(0)
            
            for i in query:
                if i not in self.doc_to_term[docID]:
                    idf = self.posting_list[i][0]
                    q_v.append(query[i] * idf)
                    d_v.append(0)
            
            return q_v, d_v 

        def compute_cosine2(self, q_v, d_v):
            cos_sim = dot(q_v, d_v) /((norm(q_v)*norm(d_v)))
            return cos_sim


        # computing the cosine similarity score, 
        # in this version i will normalize and dot product.
        def compute_cosine(self, query, docID):
            query_vector = []
            doc_vector = []
            for index, i in enumerate(self.doc_to_term[docID]):
                TFIDF = (1 + math.log10(self.doc_to_term[docID][i])) * self.posting_list[i][0]
                doc_vector.append(TFIDF)

                if i in query:
                    TFIDF = (self.posting_list[i][0])
                    query_vector.append(query[i] * TFIDF)
                    #query_vector.append(query[i])
                else:
                    query_vector.append(0)

            for i in query:
                if i not in self.doc_to_term[docID]:
                    TFIDF = (self.posting_list[i][0])
                    query_vector.append(query[i] * (self.posting_list[i][0]))
                    #query_vector.append(self.posting_list[i][0])
                    #query_vector.append(query[i])
                    doc_vector.append(0)

            # normalize and dot product
            #query_vector = normalize([query_vector])[0]
            #doc_vector = normalize([doc_vector])[0]

            #score = 1 - spatial.distance.cosine(query_vector, doc_vector)
            #score = np.array(query_vector).dot(np.array(doc_vector))
             
            score = dot(query_vector, doc_vector)/(norm(query_vector)*norm(doc_vector))
            
            return score 
            
            
        # takes 2 vectors and returns one vector with each element 
        # being the sum of the two inputs
        # vectors must be of equal length
        def sum_2vectors(self, v1, v2):
            assert len(v1) == len(v2), 'vectors must be of equal length'
            vector_sum = [x + y for x, y in zip(v1, v2)]
                
            return vector_sum
            
        # takes a list of vectors, returns one vector with each element
        # being the sum of all vector inputs 
        # vectors must be of equal length
        def sum_vector_list(self, vectors):
            for v in vectors:
                assert len(v) == len(vectors[0]), 'vectors must be of equal length'
            final_vector = [0 for i in range(len(vectors[0]))]

            for element in range(len(final_vector)):
                for v in vectors:
                    final_vector[element] += v[element]

            return final_vector


def percision(results, r_docs):
    tp = list(set(results) & set(r_docs))
    fp = list(set(results) - set(tp))
    return len(tp) / (len(tp) + len(fp))

def recall(results, r_docs):
    tp = list(set(results) & set(r_docs))
    fn = list(set(r_docs) - set(tp))
    return len(tp) / (len(tp) + len(fn))

def mean_avg_per(results, r_docs):
    p = []
    r_tot = 0
    for i, v in enumerate(results):
        if v in r_docs:
            r_tot += 1
            p.append(r_tot / (i+1))
    return sum(p)/(i+1)

def run_rocchio(index, query, round_num, r_docs):
    query = index.filter_query(query)
    p = []
    r = []
    mean_ap = []
    
    
    for i in range(round_num):
        docs = index.exact_search(query, len(r_docs))
        print(docs)
        p.append(percision(docs, r_docs))
        r.append(recall(docs, r_docs))
        mean_ap.append(mean_avg_per(r_docs, docs))

        n_docs = (set(docs) - set(r_docs))
        print(set(docs), set(r_docs), set(n_docs))
        vocab, query, weighted_d = index.rocchio(query, r_docs, n_docs)
    
    print('Percision: \n {0}'.format(p))
    print('Recall: \n {0}'.format(r))
    print('MAP: \n {0}'.format(mean_ap))






def main():
    path = './TIME/'
    i = index(path)
    i.build_index2()
    query = ' KENNEDY ADMINISTRATION PRESSURE ON NGO DINH DIEM TO STOP SUPPRESSING THE BUDDHISTS .'
    r_docs = [268,288, 304, 308, 323, 326, 334]
    run_rocchio(i,query, 5, r_docs)



#    i.print_dict()
if __name__ == '__main__':
    main()


















