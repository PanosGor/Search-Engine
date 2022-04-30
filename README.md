# Python Search-Engine
Search Engine made in Python based on Stack Overflow scraped comments

## Project Overview 

The main goal of this project was to create an information retrieval system in order to retrieve the most related documents from the corpus according to the user’s input.
For the creation of the corpus, I Scraped user Questions from Stack Overflow. In total 5475 User Questions in addition to their answers were scraped from Stack Overflow.
During the scrapping I encountered issues as the website blocked the IP address of the computer that was doing the data scrap because of the high number of requests to the website. 
In order to solve this issue, I incorporated time.sleep() in the code of the scraper in order to pause the code for a number of random seconds (1-4) in order to imitate
the actions of a real person. 
Even though the program became significantly slower the website stopped banning our IP address and we collected the data for 5475 User Questions.
The data were saved in a json file that was used as the main source to create the Corpus. 

The data were processed in order to remove duplicates, more specifically during the scrape many questions were scraped twice. 
In order to remove the duplicates a dictionary was used where the key for every document was the URL of the Stack Overflow Question, and the value was question itself. 
Any duplicates were overridden. In total 2351 Duplicates were removed.

## Vector Space Model

# Processing the data

The Vector Space Model was used as the main information retrieval model for this project. Each document in the Corpus was preprocessed before calculating TF-IDF. 
More specifically any stop words were removed (The nltk library was used in order to create a list of all stopwords) as well as any non-words (numbers dates e.t.c) 
that did not provide any actual value to the semantics of the document (Regex library was used in order to capture only words that contained letters). 
The remaining text was stemmed by using PorterStemmer. The same methodology was also used later in the program in order to process the User’s Query.

# TF-IDF

All the documents and the Query were vectorized by using the TF-IDF methodology. 
Then the inner product between the Vector of the Query and each document vector was calculated and all the results were stored sorted out in descending order. 
In order to choose the most similar to the query results the average inner product was calculated and only the documents that had inner product higher than the 
average are printed for the user.

# Auto Correct 

The Damerau-Levenshtein distance method was used in order to create an autocorrect algorithm for the user Queries. 
A Set of all the unique words in the corpus in addition to all the words in the English vocabulary was created, a total of 236.736 words. 
More specifically first we downloaded all the English Vocabulary from the nltk library. In total 236.736 words were downloaded from nltk.

The idea was that after the user inputs his Query and after processing the query (removing stopwords, non-words, stemming) each remaining word would be passed 
through a function that checks if it is a word or not if the word exists in the English vocabulary, then no further actions required, and the next word of the querywould be checked. 
To accomplish that by avoiding iterating through the whole vocabulary I created a dictionary were the key and the value for each word was the word itself (d={“table”:”table”}). 
Every time a word from the query was to be checked if it exists in the vocabulary, the word was put in the dictionary d, d[“table”]. 
By using try and except method if no error was occurring then the word existed in the vocabulary. Otherwise, (except) the Levenhstein distance between the 
Query word and all the words that have same length (-1,+1) and start with the same letter was calculated. 

A dictionary was created were all the keys were all the possible lengths in the English Voc. (1-24) and the values were lists with all the words having this length 
(2=[of,on,in,to e.t.c]). 
Then a new list was created by extending the lists that had the same length as the Query_word plus the lists with -1 and +1 lengths.
From that new list the Levenhstein distance was calculated for only the words starting with the same letter as the query word. 
That helped to narrow down the words from 245.638 to a couple of thousands minimizing that way the amount of Levenhstein distances that needed to be calculated. 
After the last step only the words with the minimum distance are presented to the user. 
If there is only one word, then it is substituted automatically in the query otherwise the user can choose from a list of recommendations and can even choose to leave the word as is. 
Because of the unique topics of that are discussed in Stack Overflow many of the words do not exist in the English voc (Java, SQL e.t.c). 
To solve this issue, a Python set was created of all the words in the corpus and passed each word from the autocorrect algorithm above to correct any typos for each word. 
Then I added all the non-existing words in the initial English vocabulary extra 8902 words related to Computer Science topics.

## Query Expansion

The idea here is that I wanted to expand the query so that it gives back results that are semantically related to the query. 
In order to achieve that I needed to use word embeddings with the help of Word2Vec. 
The vocabulary of the dataset revolves a lot around computer science terminology so I had to train Word2Vec so that it can “understand” the semantic meaning of the words. 


The way the model was trained, is that after the pre-processed phase the text from the data was created sentences and fed them to the model with k=5 positions away from each center word. 
Naturally Word2Vec considers all words of a sentence as center words so it creates relations between them by representing each word and context as n dimensional vectors. 
One problem I faced in the process of training the model is that our dataset was too small for the model to be effective but I managed to fix that by making the model read our data more than 30 times. 
In the pictures bellow you can see that for words like “python” or “panda” the relevant terms are not snakes or other kinds of bears.

The way that the query was expanded is that for each word that there is in the query I find relevant words from the model and build a second query in the background and match the second query with the tf-idf vectors.
I also tried to make a query sentence autocomplete using again Word2VVec but for this task we trained the model with phrases(bigrams) from our total vocabulary. 
In the next pictures you can see some results for the same words (“python”, ”pandas”) as before.

## Clustering

In order to further improve the effectiveness of the information retrieval system created so far, clustering was applied on the unlabeled data. 
The decision for applying clustering in the initial data was based on the need to cut down computational expense. 
Before clustering the data, a cosine similarity was applied between the query and all the documents. 
This means that there were 3124 computations taking place before the user got his answers. 
After the clustering, the query should take part in cosine similarity computations only with the documents belonging to the cluster that fits best. \
Another goal is to improve the performance of the model in the sense that retrieved documents with lower relevance level would still have a value to the user asking the query.
The method used to create the clusters is K-Means. 
A major challenge in the current dataset was the humongous vectors that represented documents. 
Basically, every word of the vocabulary was a dimension in the space the vectors exist in. 
So, there were 3124 documents represented on a vector space of 12960 dimensions (length of the corpus). 
In order to battle the so-called “Curse of dimensionality”, Singular Value Decomposition method was applied. 
This method was preferred because of how sparse the initial matrix of documents was. 
The dimensions of vectors were reduced to 100. This decision was made on the basis that the vectors needed to become small enough to be manageable in the expense of minimum information loss.
Before clustering the data, it was crucial to identify the optimal number of clusters needed to be created. 
This decision was based on a combination of metrics. First, the residual sum of squares was computed for and then it was plotted with the possible number of clusters to be created. The diagram that occurred is the following:

*Diagram 1: Residual sum of squares – Number of Clusters*

![image](https://user-images.githubusercontent.com/82097084/166110566-7e3f2208-54a2-4780-8dbc-cdbaf642182f.png)

Then, another metric named silhouette coefficients was computed and plotted with the possible number of clusters. 
The silhouette coefficient indicates the goodness of the clustering technique and it takes values between -1 and 1. 
The closer this value is to one the more clearly distinguished and well apart from each other the clusters are. 
From plotting the result was the following:

*Diagram 2: Silhouette coefficients – Number of Clusters*

![image](https://user-images.githubusercontent.com/82097084/166110585-9ce940c5-0249-42c5-8013-45f0e19e84ab.png)

Based on those diagrams the decision that three (3) is the optimum number of clusters was made.
To support this choice, a programmatical method was also used. 
The python kneed package was used and especially its function called KneeLocator. The result was the same. 

Finally, the clusters were created. The clusters were visualized with three different methods: SVD, PCA, TSNE. The two-dimensional visualizations are the following:

*Diagram 3: Cluster and Centroids (black points) 2-D visualization with SVD*

![image](https://user-images.githubusercontent.com/82097084/166110611-22566e2d-e282-4c3f-899a-cb84a45932f1.png)

*Diagram 4: Cluster 2-D visualization with PCA*

![image](https://user-images.githubusercontent.com/82097084/166110723-c1222cf8-0130-4a75-a8eb-5060c4132e18.png)

*Diagram 5: Cluster 2-D visualization with TSNE*

![image](https://user-images.githubusercontent.com/82097084/166110740-75f61992-7c74-4954-8918-9acb09e04eb6.png)


The initial documents used as the search base of the engine are unlabeled, meaning that the clusters are created upon mathematical measures such as distance of points
from the centers of the clusters, so there is no practical visualization about what is the ‘subject’ for each one. 
To ease this problem, some ‘key’ words were produced for each one. 
The results are presented in the following table:

*Table 1: Keywords produced for each cluster*

|**Clusters**|Keywords|
|----------|----------|
|**Cluster 0**|	string, values, object, like, column, value, list, array, table, data|
|**Cluster 1**| image, trying, want, app, like, use, using, file, code, error|
|**Cluster 2**| overflow, update, editing, post, want, hours, improve, ago, question, closed|

The words seem to have some cohesion between them, so some assumptions about the thremmatology of each of them can be made. 
As a second step, the query given by the user should be matched with the correct cluster. Here a major problem arose. 
The length of the query is random and probably small, but the vectors of the space created are of 100 dimensions. 
To get over this, from the initial frame of docs-words about 300 were randomly chosen and added to a data frame that also included the query. 
The goal of this process was to create a query vector that did have the same length with the other vectors (documents) but also with the centroids of the clusters. 
Then, Singular Value Decomposition was applied in this frame and the length of the vectors was reduced to 100 once again.
To identify in which cluster the query belongs to, the cosine similarity formula between the query vector and the centroid vectors is applied. 
The largest value of this metric, reveals in which centroid the query is closer to, thus concluding that the correct cluster the one with the specific center.

## BERT

As an alternative way to vectorize the documents in the Corpus as well as the Query, BERT model (“bert-base-nli-mean-tokens”) was also used from Sentence Transformers library. 
In order to save computational time after using BERT to vectorize all the documents in the Corpus once and then we saved the equivalent vectors for each document in a dictionary. 
Then the dictionary was saved locally as a json file and it is loaded in the program at the initial steps once every time we instigate our program. 
BERT was also used in order to vectorize the Query.  
A problem that was encountered was that after calculating cosine similarities for the most similar documents to the Query the most similar documents fetched were not accurate.
For example after copy and pasting a title for one of the Questions from our data:

*Query: Python labelling new data points in a histogram.*

Top 3 BERT Result Titles: 
-	“How do I traverse implicit code in RecursiveASTVisitor”
-	“How to resolve invalid package name error in npm”
-	“Angular Webpack can be used to load scripts dynamically”

Top 3 Results by using TF-IDF method:
-	“Python labelling new data points in a histogram”
-	“Having labels instead of number on axes”
-	“How to generate a 'label' using a json file in app configuration service?”

From the results above it is clear that TF-IDF vectorizing method was performing better than BERT.
The reason behind this is most likely the fact that BERT was not trained to recognize Computer Science topics.

A better approach would be to fine tune BERT  model to the Stack Overflow Dataset.



