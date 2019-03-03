#Python 3.7
import re
import os
import collections
import time
import math

class index:
        #maps term ids to the doc ids and the positions of the term in each document
        posting_list = {}
        #maps document ids to there file name for returning file names
        docID = {}
        #maps terms to term ids
        termID = {}
        
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
                    if i in self.posting_list:  
                        self.posting_list[i].append((len(self.docID),len(index[i]) ,index[i]))
                        TFID = math.log10(len(listing)/(len(self.posting_list[i])-1))
                        self.posting_list[i][0]=TFID
                    else:
                        TFID = math.log10(len(listing))
                        self.posting_list[i]=[TFID, (len(self.docID),len(index[i]),index[i])]

            
            print('Index built in: ', time.time()-start_time)
            
                
            


       
##########
# This is an exact AND search
# all terms must be in document 
# in order to return document
##########
        def and_query(self, query_terms):
            start_time=time.time()
            final_docs = []
            
            term_list = []
            
            #taking the string and parsing into words
            query_terms = query_terms.split(' ')
            #print('length of query terms: ' , len(query_terms))
            
            #remove stop words
            query_terms = self.remove_stop(query_terms)
            
            #first just retreving the docIDs for each word being searched.
            #catching if a word is not in the index and telling the user to try again.
            for i in query_terms:
                try:
                    termID = self.termID[i]
                except KeyError:
                    termID = False 
                    print('No matching documents.')
                    return
                term_list.append(self.posting_list[termID])

            #now for the list merging. first i am making a dictionary to store pointers
            #I am going to use as many points as terms i have 
            #this ditionary 'term_pointers' tracks the index of where in the term poosting list we are.
            #it also tracks the length of each posting list, to know if we are at the end, and when to break
            term_pointers = {}
            flag = False 
            for i in range(len(term_list)):
                term_pointers[i] = [0,len(term_list[i])-2]

            #keep going until we reach the end of one of the posting lists        
            while flag is False:
                counter = 0
                #modified from phase 1 because the posting list now contains IDF values at the beginning of each
                #terms posting list.
                max_doc = term_list[0][term_pointers[0][0]+1][0]
                
                
                #compare the values of each pointer
                for a in range(1,len(term_list)):
                    if term_list[0][term_pointers[0][0]+1][0] == term_list[a][term_pointers[a][0]+1][0]:
                        counter += 1
                
                #keeping track of the maximum of the docIDs of the terms
                    if term_list[a][term_pointers[a][0]+1][0] > max_doc:
                        max_doc = term_list[a][term_pointers[a][0]+1][0]

               
                #if the counter is up to n-1, then we have found all of the words
                #in the same document, so we add the document to the final list to return
                #and add 1 to each index.
                #we also check if the index is greater than its length, in which case we set the
                #flag to false to exit the entire while loop
                if counter == len(term_list)-1:
                    final_docs.append(term_list[0][term_pointers[0][0]+1][0]-1)
                    
                    for a in term_pointers:
                        term_pointers[a][0] += 1
                        
                        if term_pointers[a][0] > term_pointers[a][1]:
                            flag = True

                #if we didnt, then we move all pointers which are not the max, up 1
                #we also check if the index is greater than its length, in which case we set the
                #flag to false to exit the entire while loop
                else:
                    for a in term_pointers:
                        
                        if term_list[a][term_pointers[a][0]+1][0] != max_doc:
                            term_pointers[a][0] += 1
                            
                            if term_pointers[a][0] >= term_pointers[a][1]:
                                flag = True
            self.print_doc_title(final_docs)

            
            print('total number of ducments returned: ', len(final_docs))
            print('total time to retrieve documents: ',time.time()-start_time)




#############################################################################
# helpers
#############################################################################
        
        #def compute_cosine_sim(self, 

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
#    for x in range(0,50):
#        print(x, i.posting_list[x][:1],len(i.posting_list[x])) 
    while True:
        i.and_query(input('Please enter the query terms:').strip().lower())
    


if __name__ == '__main__':
    main()












