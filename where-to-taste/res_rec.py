
import numpy as np 
import re
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize


# Top Restaurants
def topres(loc,top=10):
    data = pd.read_csv(loc+'.csv', encoding ='latin1')

    C = data['Res_ratings'].mean()
    m = data['Res_votes'].quantile(0.85)
    # Filter out all qualified restaurants into a new DataFrame
    dnew = data.copy().loc[data['Res_votes'] >= m]

    # Function that computes the weighted rating of each restaurant
    def weighted_rating(x, m=m, C=C):
        v = x['Res_votes']
        R = x['Res_ratings']
        # Calculating the score
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    dnew['score'] = dnew.apply(weighted_rating, axis=1)

    #Sort restaurant based on score calculated above
    dnew = dnew.sort_values('score', ascending=False)

    C = dnew['score'].mean()
    m = dnew['Res_reviews'].quantile(0.75)

    # Filter out all qualified restaurants into a new DataFrame
    dnewF = dnew.copy().loc[dnew['Res_reviews'] >= m]

    # Function that computes the weighted rating of each restaurant
    def weighted_ratingF(x, m=m, C=C):
        v = x['Res_reviews']
        R = x['score']
        # Calculating the score
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    dnewF['Fscore'] = dnewF.apply(weighted_ratingF, axis=1)

    #Sort restaurant based on score calculated above
    dnewF = dnewF.sort_values('Fscore', ascending=False)
    
    del dnewF['score']
    del dnewF['Fscore']

    #Print the top 10 restaurants in CP
    return dnewF.head(top)





# Restaurant with similar food as your searched restaurant


data_sample=[]
def resfoodmatch(loc,title,top=10):   
    try:
        data = pd.read_csv(loc+'.csv', encoding ='latin1')
        title = title.title()

        global data_sample       
        global cosine_sim
        global sim_scores
        global tfidf_matrix
        global corpus_index
        global feature
        global rest_indices
        global idx

        # When location comes from function ,our new data consist only location dataset
        data_sample = data
     
        
        #Feature Extraction
        data_sample['Split']="X"

        for i in range(0, data_sample.index[-1] + 1):
            # Split the 'Res_food' column into individual items
            split_data = re.split(r'[,]', data_sample['Res_food'].iloc[i])
            
            # Clean up and remove spaces
            split_data = [' '.join(re.sub(r'\s+', ' ', item.strip())) for item in split_data]
            
            # Join the list of foods into a single string
            split_data = ' '.join(split_data)
            
            # Assign the cleaned string back to the 'Split' column
            data_sample.loc[i, 'Split'] = split_data


        # for i in range(0,data_sample.index[-1]):
        #     split_data=re.split(r'[,]', data_sample['Res_food'][i])


            
        #     for k,l in enumerate(split_data):
        #         split_data[k]=(split_data[k].replace(" ", ""))
        #     split_data=' '.join(split_data[:])
        #     data_sample['Split'].iloc[i]=split_data
            
        #TF-IDF vectorizer
        #Extracting Stopword
        tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN for empty string
        data_sample['Split'] = data_sample['Split'].fillna('')
    #Applying TF-IDF Vectorizer
        tfidf_matrix = tfidf.fit_transform(data_sample['Split'])
        tfidf_matrix.shape
        
        # Using for see Cosine Similarty scores
        feature= tfidf.get_feature_names()
    #Cosine Similarity
        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) 
        
        # Column names are using for index
        corpus_index=[n for n in data_sample['Split']]
           
        #Construct a reverse map of indices    
        indices = pd.Series(data_sample.index, index=data_sample['Res_name']).drop_duplicates() 
        #index of the restaurant matchs the cuisines
        idx = indices[title]
    #Aggregate rating added with cosine score in sim_score list.
        sim_scores=[]
        for i,j in enumerate(cosine_sim[idx]):
            k=data_sample['Res_ratings'].iloc[i]
            if j != 0 :
                sim_scores.append((i,j,k))
                
        #Sort the restaurant names based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: (x[1],x[2]) , reverse=True)
        # 10 similar cuisines
        sim_scores = sim_scores[0:50]
        rest_indices = [i[0] for i in sim_scores] 
      
        data_x =data_sample[['Res_id','Res_name','Res_food','Res_address','Res_ratings','Res_votes','Res_reviews','Res_url']].iloc[rest_indices]
        
        data_x['Cosine Similarity']=0
        for i,j in enumerate(sim_scores):
            data_x['Cosine Similarity'].iloc[i]=round(sim_scores[i][1],2)
            
        del data_x['Cosine Similarity']
        return data_x.head(top)

    except:
        Error= 'Restaurant Name not found!'
        return Error

def resfoodmatchi(loc, title, top=10):
    try:
        # Load the dataset
        data = pd.read_csv(loc + '.csv', encoding='latin1')
        title = title.title()

        global data_sample
        global cosine_sim
        global sim_scores
        global tfidf_matrix
        global corpus_index
        global feature
        global rest_indices
        global idx

        # Set the global data sample for further processing
        data_sample = data
        
        # Feature extraction: splitting the 'Res_food' column
        data_sample['Split'] = "X"
        
        for i in range(0, data_sample.index[-1] + 1):
            # Split the 'Res_food' column into individual items
            split_data = re.split(r'[,]', data_sample['Res_food'].iloc[i])
            
            # Clean up and remove spaces
            split_data = [' '.join(re.sub(r'\s+', ' ', item.strip())) for item in split_data]
            
            # Join the list of foods into a single string
            split_data = ' '.join(split_data)
            
            # Assign the cleaned string back to the 'Split' column
            data_sample.loc[i, 'Split'] = split_data
        
        # TF-IDF Vectorizer to transform the 'Split' column into a matrix
        tfidf = TfidfVectorizer(stop_words='english')
        data_sample['Split'] = data_sample['Split'].fillna('')  # Replace NaN with empty string
        tfidf_matrix = tfidf.fit_transform(data_sample['Split'])
        
        # Compute cosine similarity between the restaurants
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Create a list of features from the vectorizer
        feature = tfidf.get_feature_names_out()
        
        # Create an index of restaurant names
        corpus_index = [n for n in data_sample['Split']]
        
        # Map restaurant names to their indices in the DataFrame
        indices = pd.Series(data_sample.index, index=data_sample['Res_name']).drop_duplicates()
        
        # Get the index of the restaurant you're looking for
        idx = indices[title]
        
        # Initialize a list for storing similarity scores
        sim_scores = []
        
        # Loop through each similarity score for the given restaurant
        for i, j in enumerate(cosine_sim[idx]):
            k = data_sample['Res_ratings'].iloc[i]
            if j != 0:
                sim_scores.append((i, j, k))
        
        # Sort the restaurants based on similarity scores and ratings
        sim_scores = sorted(sim_scores, key=lambda x: (x[1], x[2]), reverse=True)
        
        # Select top 50 restaurants based on similarity
        sim_scores = sim_scores[:50]
        rest_indices = [i[0] for i in sim_scores]
        
        # Get relevant columns for the similar restaurants
        data_x = data_sample[['Res_id', 'Res_name', 'Res_food', 'Res_address', 'Res_ratings', 'Res_votes', 'Res_reviews', 'Res_url']].iloc[rest_indices]
        
        # Add cosine similarity scores to the result
        data_x['Cosine Similarity'] = 0
        for i, j in enumerate(sim_scores):
            data_x.loc[i, 'Cosine Similarity'] = round(sim_scores[i][1], 2)
        
        # Remove the 'Cosine Similarity' column before returning (if you don't need it)
        del data_x['Cosine Similarity']
        
        # Return the top 'n' similar restaurants
        return data_x.head(top)
    
    except KeyError:
        return "Restaurant Name not found!"



def search(loc,name):
    data = pd.read_csv(loc+'.csv', encoding ='latin1')
    name = name.lower()
    Search =data[data["Res_name"].str.lower()==name]
    
    Similar =data[data["Res_name"].str.lower().str.contains(name)]
    
    if Search.empty!=True:
        return Search
    elif Similar.empty!=True:
        return Similar   
    else:
        print('No result found')
#Top n restaurants

#topres('Gurgaon')



# Top 10 similar restaurant with cuisine of 'Selected' restaurant
#resfoodmatch("Gurgaon",'bIryani blues',20)

