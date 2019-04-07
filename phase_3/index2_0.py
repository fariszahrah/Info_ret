#Python 3.X

# This is bbecause I intended to rewrite the index because 
# the other code has a supposed 'bug' according to other code.

# this code will be returned to shortly, first i am just writing 
# ther report 

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
        # this is the main posting list
        # {termID : [idf, [docID, tf], [docID, tf] , ..], ...}
        self.posting_list = {}
        # {docID : title, ...}
        self.docId_to_title = {}
        # {term : termID, ...}
        self.term_to_termId = {}
        # {termID : term , ...}
        self.termId_to_term = {}
        # stop words
        self.stop_words = set()
        # {docId : doc_vector}
        self.document_vectors = {}
        # {docId:{ termID: freq, ...}, docID:{termID : freq},...}
        self.doc_to_term = {}
        # {docId : norm(doc)}
        self.docId_to_norm = {}
        # entire vocab, currently it is going to be an ordered list in ascending order of termId
        self.vocab = []
        # query norm, this changes lots
        query_norm = 0.0
        

    def buildIndex(self):
        path = self.path
        # create stop word list
        with open(path + 'TIME.STP') as f:
            p = re.compile('[a-z]+', re.IGNORECASE)
            self.stop_words.update(p.findall(f.read().lower()))
        # split document into docs based on * delimeter 

        with open(path + 'TIME.ALL') as f:
            docs = f.read().lower()
        with open(path + 'TIME.ALL') as f:
            docs = f.read().lower().replace('\n', ' ').split('*')[0:-1]
            
        for i,doc in enumerate(docs): 
            if i==0:
                continue
            docId = i
            docs[docId]= ' '.join(docs[docId].split())
            # set docID, title
            self.docId_to_title[docId] = re.search(r'\d+', doc).group()
            # initiate doc_to_term for the doc
            self.doc_to_term[docId] = {}

            p = re.compile('\W+',re.IGNORECASE)
            tokens = re.split(p, doc)
            # assign termId and populate doc_to_term for all terms per doc
            for term in tokens:
                if term not in self.stop_words:

                    # if new term, create new termId
                    if term not in self.term_to_termId:
                        termId = len(self.term_to_termId)
                        self.term_to_termId[term] = termId
                        self.termId_to_term[termId] = term
                    # if we have seen, then set termId to correct value
                    else:
                        termId = self.term_to_termId[term]

                    # populate doc_to_term
                    if termId not in self.doc_to_term[docId]:
                        self.doc_to_term[docId][termId] = 1
                    else:
                        self.doc_to_term[docId][termId] += 1
            
            # populate posting list, 1 doc at a time
            for termId in self.doc_to_term[docId]:
                # set idf to just log10(number of docs)
                if termId not in self.posting_list:
                    idf = math.log10(len(docs))
                    self.posting_list[termId] = [idf, [docId,self.doc_to_term[docId][termId]]]
                # set idf to log10(number of docs/ docs with term present) 
                else:
                    idf = math.log10(len(docs)/(len(self.posting_list[termId])-1))
                    self.posting_list[termId][0]=idf
                    self.posting_list[termId].append([docId,self.doc_to_term[docId][termId]])

        #create vocab
        self.vocab = sorted([i for i in self.posting_list])

        # converting posting_list to ordered dict
        # this seems like bad practice.. hmmm
        self.posting_list = collections.OrderedDict(self.posting_list)


        # create document vectors and populate doc_to_norm dict
        for docId in self.doc_to_term:
            self.document_vectors[docId] = [0 for i in range(len(self.vocab))]
            n = 0
            for termId in self.doc_to_term[docId]:
                tfidf = (1 + math.log10(self.doc_to_term[docId][termId])) * self.posting_list[termId][0]
                self.document_vectors[docId][termId] = tfidf
                n += tfidf ** 2
            self.docId_to_norm[docId] = math.sqrt(n)


    def exactSearch(self, query, k=10):
        docs = self.findDocs(query)
        scores = self.scoreDocs(query, docs, k)
        docs_to_ret = [i[1] for i in scores]
        return docs_to_ret

    
    
    # takes raw query, parses and create query vector of len(vocab)
    def createQueryVector(self, query):
        p = re.compile('\w+',re.IGNORECASE)
        tokens = Counter(re.findall(p, query.lower()))
        # instantiate a list the length of the vocab
        query_vector = [0 for i in range(len(self.vocab))]
        # n is for computing the norm of the query vector, only on initial query
        n = 0.0
        # set the element of the list to the tfidf of the query term, for all terms
        for token in tokens:
            if token not in self.stop_words and token in self.term_to_termId:
                termId = self.term_to_termId[token]
                wtf = 1 + math.log10(tokens[token])
                tfidf =  wtf * self.posting_list[termId][0]
                query_vector[termId] = tfidf 
                n += tfidf **2
        self.query_norm = math.sqrt(n)
        return query_vector 


 
    # takes a query vector, finds all docs with any terms in common, return set of docs
    def findDocs(self, query):    
        docs = []
        for i,termId in enumerate(query):
            if termId > 0.0:
                for doc in range(1,len(self.posting_list[i])):
                    docs.append(self.posting_list[i][doc][0])
        return set(docs)

    # takes docIds(iterable), query_vector, k(int) 
    # and returns top k similar docs by cosine sim
    def scoreDocs(self, query, docs, k):
        scores = []
        for docId in docs:
            doc_vector = self.document_vectors[docId]
            score = dot(doc_vector,query) / (self.query_norm*self.docId_to_norm[docId])
            scores.append(tuple((score, docId)))

        return sorted(scores, key=lambda k:k[0], reverse=True)[:k]


    def computeQueryNorm(self, query):
        weights = [i**2 for i in query if i>0.0]
        n = sum(weights)
        self.query_norm = math.sqrt(n)




    def rocchio(self, query, rel_docs, n_docs, alpha=1,beta=0.75, gamma = 0.25):
        p_vector_sum = [0 for i in range(len(self.vocab))]
        n_vector_sum = [0 for i in range(len(self.vocab))]
        for docId in rel_docs:
            doc_vector = self.document_vectors[docId]
            p_vector_sum = [x + y for x,y in zip(p_vector_sum, doc_vector)]

        for docId in n_docs:
            doc_vector = self.document_vectors[docId]
            n_vector_sum = [x+y for x,y in zip(n_vector_sum, doc_vector)]


        if  len(rel_docs) > 0:
            p_vector_sum = np.array(p_vector_sum) * beta / len(rel_docs)
        if len(n_docs) > 0:
            n_vector_sum = np.array(n_vector_sum) * gamma / len(n_docs)
        q_vector = np.array(query) * alpha 

        final_vector = list(q_vector + p_vector_sum - n_vector_sum)
        #remove all negative values
        for i in final_vector:

            if i < 0.0:
                i = 0

        
        return final_vector 



def percision(predicted, actual):
    tp = list(set(predicted) & set(actual))
    fp = list(set(predicted) - set(tp))
    return len(tp) / (len(tp) + len(fp))

def recall(predicted, actual):
    tp = list(set(predicted) & set(actual))
    fn = list(set(actual) - set(tp))
    return len(tp) / (len(tp) + len(fn))

def mapk(predicted, actual):
    p = []
    r_tot = 0
    for i, v in enumerate(predicted):
        if v in actual:
            r_tot += 1
            p.append(r_tot / (i+1))
    return sum(p)/(i+1)

def runRocchio(index, query, rounds, r_docs, verbose = False, print_V=False, pseudo = True):
    p = []
    r = []
    mean_ap  = []

    for i in range(rounds):
        docs = index.exactSearch(query, len(r_docs))
        p.append(round(percision(docs, r_docs),4))
        r.append(round(recall(docs, r_docs),4))
        mean_ap.append(round(mapk(docs, r_docs),4))
        
        
        if not pseudo:
            r_docs_in_q = list(set(docs) & set(r_docs))
            n_docs = (set(docs) - set(r_docs)).copy()
            if verbose:
                print('Round {2}:\nRelevant docs: {0}\nNonrelevant docs: {1}'.format(r_docs_in_q,n_docs,i+1))
            query = index.rocchio(query, r_docs_in_q, n_docs)    
        else:
            n = int(len(r_docs) / 2)
            r_docs_in_q = docs[:n]
            n_docs = docs[n:]
            if verbose:
                print('Round {2}:\nRelevant docs: {0}\nNonrelevant docs: {1}'.format(r_docs_in_q,n_docs,i+1))
            query = index.rocchio(query, r_docs_in_q, n_docs)
        if print_V:
            query_dict = {}
            for i,weight in enumerate(query):
                if weight > 0.0:
                    query_dict[index.termId_to_term[i]]=weight
            print(query_dict)

    if verbose:
        print('Percision: {0}\nrecall: {1}\nmap: {2}'.format(p,r,mean_ap))
    return p,r,mean_ap 


def main():
    path = './TIME/'
    start_t = time.time()
    i = index(path)
    i.buildIndex()
    print('Index built in: {0} seconds'.format(time.time() - start_t))
        
    verbose = True 
    print_V = False
    rounds = 7
    print('\nQuery 46:')
    query = " PRESIDENT DE GAULLE'S POLICY ON BRITISH ENTRY INTO THE COMMON MARKET ."
    query = i.createQueryVector(query)
    r_docs = [1, 20, 23, 32, 39, 47, 53, 54, 80, 93, 151, 157, 174, 202, 272, 291, 294, 348]
    p,r,mean_ap = runRocchio(i,query,rounds, r_docs, verbose, print_V)


    print('\nQuery 6:')
    query1 = 'CEREMONIAL SUICIDES COMMITTED BY SOME BUDDHIST MONKS IN SOUTH VIET NAM AND WHAT THEY ARE SEEKING TO GAIN BY SUCH ACTS .'
    query1 = i.createQueryVector(query1)
    r1_docs = [257, 268, 288, 304, 308, 323, 324, 326, 334]
    p,r,mean_ap = runRocchio(i,query1, rounds, r1_docs, verbose, print_V)

    print('\nQuery 12:')
    query2 = 'OPPOSITION OF INDONESIA TO THE NEWLY-CREATED MALAYSIA .'
    query2 = i.createQueryVector(query2)
    r2_docs = [61, 155, 156, 242, 269, 315, 339, 358]
    p,r,mean_ap = runRocchio(i ,query2, rounds, r2_docs, verbose, print_V)

    print('\nQuery 39:')
    query3 = 'COALITION GOVERNMENT TO BE FORMED IN ITALY BY THE LEFT-WING SOCIALISTS, THE REPUBLICANS, SOCIAL DEMOCRATS, AND CHRISTIAN DEMOCRATS .'
    query3 = i.createQueryVector(query3)
    r3_docs = [22, 73, 173, 189, 219, 265, 277, 360, 396]
    p,r,mean_ap = runRocchio(i ,query3, rounds, r3_docs,verbose, print_V)

    print('\nQuery 69:')
    query4 = ' THE BAATH (RENAISSANCE) PARTY FOUNDED BY MICHEL AFLAK, WHICH HAS GAINED CONTROL OF SYRIA AND IRAQ AND AIMS TO UNITE ALL ARAB COUNTRIES .'
    query4 = i.createQueryVector(query4)
    r4_docs = [70, 100, 115, 121, 139, 159, 194, 210, 224, 234, 309, 379, 388]
    p,r,mean_ap = runRocchio(i ,query4, rounds, r4_docs, verbose, print_V)

   


    
if __name__ == '__main__':
    main()























