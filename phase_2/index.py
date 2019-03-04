#Python 3.7
import re
import os
import collections
import time
import math

from numpy import dot
from numpy.linalg import norm

class index:
        #maps term ids to the doc ids and the positions of the term in each document
        posting_list = {}
        #maps document ids to there file name for returning file names
        docID = {}
        #maps terms to term ids
        termID = {}
        
        #this will contain docIDs as keys
        #as values will be a list of the terms in the document
        # and the number of times they appear in the document
        # it is used for vectorizing docs and queries
        doc_to_term = {}

        
        #words to ignore
        stop_words = set()
        with open('stop-list.txt','r') as f:
            for line in f:
                 for word in line.split():
                     stop_words.add(word)


        def __init__(self,path):
            self.path = path

        def buildIndex(self):
	    #function to read documents from collection, tokenize and build the index with tokens
	    #index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
	    #use unique document IDs
            #ignore stop words 

            path = self.path
            #just for timing
            start_time = time.time()

            #regular expression for parsing, this may need to get more complicated for normalizing, stemming, etc
            p = re.compile('[a-z]+',re.IGNORECASE)
            
            listing = os.listdir(path)
            print('total number of documents: %d'%len(listing))
            for infile in range(len(listing)):
            #for infile in range(1):
                f = open(path + listing[infile])
                self.docID[len(self.docID)] = listing[infile] 
                index = {}
                m = p.findall(f.read().lower())
                
                #adding the docID to a doc_to_term dictionary
                self.doc_to_term[len(self.docID)] = {}


                #this creates the termID dictionary
                #m = total terms per document
                #so this forloop iterates through each documents set of words and constructs a termlist
                for i in range(len(m)):
                    if m[i] not in self.stop_words:
                        if m[i] in self.termID:
                            ID = self.termID[m[i]]
                        else:
                            ID = len(self.termID)
                            self.termID[m[i]]=len(self.termID)
                        


                    #makes a posting list per document because modifying tuples is a bitch
                    #so i make them as lists, then I will cast them as I add them to the final list
                    #if they could just be lists, I wouldnt need to do this... AAAAAAAAA
                        if ID in index:
                            index[ID].append(i)
                        else:
                            index[ID] = [i]

                #Here we take our document posting list, and cast into tuples nad add to the final posting list        
                for i in index:
                    self.doc_to_term[len(self.docID)][i]=len(index[i])
                    if i in self.posting_list:  
                        self.posting_list[i].append((len(self.docID),len(index[i]) ,index[i]))
                        TFID = math.log10(len(listing)/(len(self.posting_list[i])-1))
                        self.posting_list[i][0]=TFID
                    else:
                        TFID = math.log10(len(listing))
                        self.posting_list[i]=[TFID, (len(self.docID),len(index[i]),index[i])]

            
            print('Index built in: ', time.time()-start_time)
            
                
############################################################################

    # takes a query, computes cosine for all documents with at least one term
    # which is similar to the query. and returns the documents in a ranked order
    
    # currently the scores and documents associated with each score is only stored 
    # in a list and then sorted based on score, this is inefficient and should be
    # a tree structure to reduce sort/retrieval complexity
        def exact_search(self, query, k):
            docs_to_score = set()
            scores = [] 
            if len(query) ==0:
                print('No documents with any words related, sorry:(')
                return
            query = self.remove_stop(query)
            query = self.query_to_termID(query)
            print(query)
            # obtain docs to score
            for termID in query:
                for doc in self.posting_list[termID]:
                    if type(doc) == tuple:
                        docs_to_score.add(doc[0])
        
            # compute score for each doc
            for docID in docs_to_score:
                query_vector, doc_vector = self.compute_doc_query_vectors(query,docID)
                if docID == 1:
                    print(query_vector,'\n\n', doc_vector)
                score = self.compute_cosine_sim(query_vector,doc_vector)
                scores.append((score,docID))
        
            #sort and return top k docs
            scores = sorted(scores, key=lambda k:(-k[0],k[1]))
            return scores[:k]
        



       
#############################################################################
        
        
        def query_to_termID(self,query):
            ID_query=[]
            for i in query:
                try:
                    ID_query.append(self.termID[i])
                except:
                    pass
            
            return ID_query




        #takes a query,document, and computes the document and query vectors
        def compute_doc_query_vectors(self, query, docID):
            query_vector =[]
            doc_vector=[]
            for i in self.doc_to_term[docID]:
                TFIDF = (1+math.log10(self.doc_to_term[docID][i])) * self.posting_list[i][0]
                doc_vector.append(TFIDF)
                if i in query:
                    TFIDF = (self.posting_list[i][0])
                    query_vector.append(TFIDF)
                else:
                    query_vector.append(0)
            for i in query:
                if i not in self.doc_to_term[docID]:
                    query_vector.append(self.posting_list[i][0])
                    doc_vector.append(0)
            return query_vector,doc_vector 




        # take two vectors and compute similarities
        def compute_cosine_sim(self, query_vector, doc_vector):
            cos_sim = dot(query_vector, doc_vector)/(norm(query_vector)*norm(doc_vector))
            return cos_sim 



        def remove_stop(self,query):
            to_ret = []
            for i in query:
                if i not in self.stop_words:
                    to_ret.append(i)
            return to_ret





        #function to print the terms and posting list in the index
        def print_dict(self):
            start_time = time.time()
            inv_map = {v: k for k, v in self.termID.items()}
            for i in range(len(self.posting_list)):
                time.sleep(.1)
                print(inv_map[i],self.posting_list[i])
            print('time to process and print: ', time.time()-start_time)





	#print all documents and their document id
        def print_doc_list(self):
            for row in self.docID:
                print(row, self.docID[row])







        #print the document names for the specified document ids
        def print_doc_title(self, docs):
            print("Requsted Documents")
            for doc in docs:
                print(self.docID[doc])


        


#for checking if all TermId's are in the same document
#for exact searching 
def is_equal(items):
    count = 0
    for a in terms:
        if items[0] == a:
            count += 1

    if count == len(item) - 1:
        return True
    else:
        return False






def main():
    ind = './../collection/'
    i = index(ind) 
    print('Index being used: %s' %ind)
    i.buildIndex()
   
   #debugging
    '''
    for x in range(0,100):
        if(len(i.posting_list[x]) ==2):
            print(x, i.posting_list[x],len(i.posting_list[x])) 
    '''
    
    while True:
        query = (input('Please enter the query terms:').strip().lower()).split(' ')
        docs = i.exact_search(query,5)
        if len(docs) > 0:
            print(docs)
    

if __name__ == '__main__':
    main()












