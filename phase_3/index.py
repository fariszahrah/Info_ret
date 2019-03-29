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
class index:
        def __init__(self,path):

            self.path = path 
            
            # {docid: {termID:freq, termID:freq, ...} ,docid: {}, ...} 
            self.doc_to_term = {}

            # {termID : [idf, (docID, tfTD), (docID, tfID) , ...] , termID: [..] , ...}
            self.posting_list = {}

            # {term : termID, term: termID, ...}
            self.term_to_termID = {}

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

                pattern = re.compile('\w+(?:-\w+)+|\w+[^ ]\w+|\w+',re.IGNORECASE)
                tokens = re.findall(pattern, docs[i][27:])
                
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
                        self.posting_list[termID] = [math.log10(len(docs)),(i,self.doc_to_term[i][termID])]
                    else:
                        self.posting_list[termID].append((i,self.doc_to_term[i][termID]))
                        self.posting_list[termID][0]= math.log10(len(docs)/len(self.posting_list[termID])-1)


            print('Index built in {0}'.format(time.time()-start_time))


        # exact top-k search
        # return top-k docIDs and their scores
        def exact_search(self, query, k):
            query = self.filter_query(query)

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

            query = self.filter_query(query)
            # first i just compute the vocab
            # this may not bbe the fastest way
            # i will refine after a working program
            q_vector, pos_feedback_vectors, neg_feedback_vectors = self.create_rocchio_vectors(query, pos_feedback, neg_feedback)

            print(q_vector)


        
        
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
                    print(query[term],self.posting_list[term][0])
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

            return q_vector, pos_doc_vectors, neg_doc_vectors

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
            query_copy = query.copy()
            for index, i in enumerate(self.doc_to_term[docID]):
                TFIDF = (1+math.log10(self.doc_to_term[docID][i])) * self.posting_list[i][0]
                doc_vector.append(TFIDF)

                if i in query:
                    TFIDF = (self.posting_list[i][0])
                    query_vector.append(query_copy[i] * TFIDF)
                    del query_copy[i]
                else:
                    query_vector.append(0)

            for i in query_copy:
                TFIDF = (self.posting_list[i][0])
                query_vector.append(query_copy[i] * (self.posting_list[i][0]))
                doc_vector.append(0)

            #normalize and dot product
            query_vector = normalize([query_vector])[0]
            doc_vector = normalize([doc_vector])[0]

            #score = 1 - spatial.distance.cosine(query_vector, doc_vector)
            score = np.array(query_vector).dot(np.array(doc_vector))
            return score 





        




def main():
    path = './TIME/'
    i = index(path)
    i.buildIndex()
    
    for x in i.posting_list:
        if i.posting_list[x][0] < 0:
            print(i.posting_list[x])
    
    query ='BACKGROUND OF THE NEW CHANCELLOR OF WEST GERMANY, LUDWIG ERHARD'
    docs = i.exact_search( query ,10)
    
    # print results
    for z in docs:
        print(z, i.doc_to_docID[z[1]])
   
    inp = 'y'
    while inp == ('y' or 'Y'):
        inp = input('\nWould you like to proceed with Rocchio: (y/n): ')
        if inp == ('n' or 'N'):
            print('okay, I guess we are done here :)')
        elif inp ==('y' or 'Y'):
            pos_feedback = input('\nEnter relevant document ids separated by space: ').split()
            neg_feedback = input('\nEnter non relevant document ids seperated bs space: ').split() 
            i.rocchio(query, pos_feedback, neg_feedback)

        else:
            print("\nEnter 'y/n' idiot, I didnt give you any other options!")
            inp = 'y'


#    i.print_dict()
if __name__ == '__main__':
    main()

















