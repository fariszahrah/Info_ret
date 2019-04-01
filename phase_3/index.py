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

        # exact top-k search
        # return top-k docIDs and their scores
        def exact_search(self, query, k):

            docs_to_score = self.find_all_docs(query) 
            doc_scores = self.score_docs(query, docs_to_score)
            if len(doc_scores) > k:
                return doc_scores[:k]
            else:
                return doc_scores 

   
        

	#function to implement rocchio algorithm
	#pos_feedback - documents deemed to be relevant by the user - list,iterable
	#neg_feedback - documents deemed to be non-relevant by the user - list,iterable
	#Return the new query  terms and their weights
        def rocchio(self, query, pos_feedback, neg_feedback, alpha = 1, beta = 0.75 , gamma = 0.15):
            print(pos_feedback, neg_feedback)
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
            
            print('length of pos feedback: {0}, Length of negative feedback: {1}'.format(len(pos_feedback),len(neg_feedback)))
            final_vector = q_vector 
            pos_vector_sum = self.sum_vector_list(list(pos_feedback_vectors.values()))
            neg_vector_sum = self.sum_vector_list(list(neg_feedback_vectors.values()))
            
            # weight vectors
            # and multiply by the number of documents 
            pos_vector_sum = list(np.array(pos_vector_sum) * beta / len(pos_feedback))
            neg_vector_sum = list(np.array(neg_vector_sum) * gamma * -1 / len(neg_feedback))
            final_vector = self.sum_vector_list([final_vector, pos_vector_sum, neg_vector_sum]) 
                
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
                score = self.compute_cosine(query, i)
                scores.append((score,i))

            return sorted(scores, key=lambda k:k[0], reverse=True)

            


        # computing the cosine similarity score, 
        # in this version i will normalize and dot product.
        def compute_cosine(self, query, docID):
            query_vector = []
            doc_vector = []
            for index, i in enumerate(self.doc_to_term[docID]):
                TFIDF = (1+math.log10(self.doc_to_term[docID][i])) * self.posting_list[i][0]
                doc_vector.append(TFIDF)

                if i in query:
                    TFIDF = (self.posting_list[i][0])
                    query_vector.append(query[i] * TFIDF)
                else:
                    query_vector.append(0)

            for i in query:
                if i not in self.doc_to_term[docID]:
                    TFIDF = (self.posting_list[i][0])
                    query_vector.append(query[i] * (self.posting_list[i][0]))
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


        




def main():
    path = './TIME/'
    i = index(path)
    i.buildIndex()
    
    for x in i.posting_list:
        if i.posting_list[x][0] < 0:
            print(i.posting_list[x])
    
    query = ' BACKGROUND OF THE NEW CHANCELLOR OF WEST GERMANY, LUDWIG ERHARD . '
    query = i.filter_query(query)
    inp = 'y'
   
    while inp == ('y' or 'Y'):
        docs = i.exact_search(query ,10)
       
        # print results
        for z in docs:
            print(z, i.doc_to_docID[z[1]])
   

        
        inp = input('\nWould you like to proceed with Rocchio: (y/n): ')
        if inp == ('n' or 'N'):
            print('okay, I guess we are done here :)')
        elif inp ==('y' or 'Y'):
            pos_feedback = input('\nEnter relevant document ids separated by space: ').split()
            neg_feedback = input('\nEnter non relevant document ids seperated bs space: ').split() 
            vocab, query, weighted_d = i.rocchio(query, pos_feedback, neg_feedback)
            
            print('len of weighted dict: {0}'.format(len(weighted_d)))
        else:
            print("\nEnter 'y/n' idiot, I didnt give you any other options!")
            inp = 'y'


#    i.print_dict()
if __name__ == '__main__':
    main()


















