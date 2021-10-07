from os import link
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(layout="wide")
from sklearn.linear_model import LinearRegression, SGDRegressor,Ridge, Lasso, ElasticNet
# 'Statsmodels' is used to build and analyze various statistical models
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from statsmodels.graphics.gofplots import qqplot
#from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import plotly as pl
import plotly.express as pex
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN
# import functions from scipy to perform clustering
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy.linalg as lin
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial.distance import pdist
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


# suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# Sidebar Radio Buttons
sbar = st.sidebar.radio(label = "USL Take Homes", 
                           options = ["Day 01", "Day 02", "Day 03"])

# List of Containers
header =st.container()
q1 = st.container()
q2 = st.container()
q3 = st.container()
q4 = st.container()
q5 = st.container()
q6 = st.container()
q7 = st.container()
q8 = st.container()
q9 = st.container()
q10 = st.container()
q11 = st.container()
q12 = st.container()
q13 = st.container()
q14 = st.container()
q15 = st.container()
q16 = st.container()
q17 = st.container()
q18 = st.container()
q19 = st.container()
q20 = st.container()

with header:
    st.title("USL Take Homes")
    st.subheader("Let's begin with some hands-on practice exercises")
    
    if sbar =="Day 01":
        notes = f'''
        **The dataset includes the yearly spendings (in dollars) of the retailers on the products purchased from the wholesale market. Our objective is to group the retailers based on their purchase.**
        
        The data definition is as follows:
        * **Region:** City of the retailer
        * **Vegetables:** Annual spending on vegetables 
        * **Personal_care:** Annual spending on personal care products
        * **Milk:** Annual spending on milk and milk products
        * **Grocery:** Annual spending on grocery
        * **Plasticware:** Annual spending on plasticware (container, bottles, dishes and so on)'''
        st.write(notes)
    
# Reading the Data...wholesale_cust.csv
df_retail= pd.read_csv("data/wholesale_cust.csv")

############################################ D A Y 0 1 ##############################################################################
#***********************************************************************************************************************************#
#Load the Dataset
with q1:
    if sbar=="Day 01":
        st.markdown("**Load the Dataset**")
        pressed = st.button("View the Data", True)
        if pressed:
            st.write(df_retail.head())
            st.write("The Shape of the Data is ", df_retail.shape)
        st.markdown("---")  


# 1. Is there any retailer whose entry is recorded more than once? If yes, do the needful.
with q2:
    if sbar=="Day 01":
        st.markdown("** Q.1. Is there any retailer whose entry is recorded more than once? If yes, do the needful.**")
        pressed = st.button("Q.1. Solution", True)
        if pressed:
            st.write(df_retail[df_retail.duplicated() == True].index)
            st.write("The indices in the above output represents the duplicate records. Let us remove these observations from the original data.")
            df_retail = df_retail.drop_duplicates(ignore_index = True)
            st.write(df_retail.duplicated().value_counts())
            st.write("Interpretation: The above output shows that there are no duplicates in the data. And now there are 439 unique entries in the data.")
        st.markdown("---")

# 2. Identify the different cities to which the retailers belong. Also, visualize their count in different cities.
with q3:
    if sbar=="Day 01":
        st.markdown("** Q.2. Identify the different cities to which the retailers belong. Also, visualize their count in different cities.**")
        pressed = st.button("Q.2. Solution", True)
        if pressed:
            st.write(df_retail.Region.unique())
            fig = pex.bar(x = df_retail.Region).update_layout(xaxis_title = "Region", yaxis_title = "Number of Retailers",
                                                              title = "Distribution of Region")
            st.write(fig)
            st.write("Interpretation: The above plot shows that 315 retailers are from Rochester. The number of retailers from Albany is least.")
        st.markdown("---")

# 3. Identify the extreme observations in the data using a visualization technique.
with q4:
    if sbar=="Day 01":
        st.markdown("** Q.3. Identify the extreme observations in the data using a visualization technique.**")
        pressed = st.button("Q.3. Solution", True)
        if pressed:
            df_num = df_retail.drop(['Region'], axis = 1)
            
        st.markdown("---")

# 4. Use the appropriate technique to remove the observations greater than 3*IQR above the third quartile.
with q5:
    if sbar=="Day 01":
        st.markdown("** Q.4. Use the appropriate technique to remove the observations greater than 3*IQR above the third quartile.**")
        pressed = st.button("Q.4. Solution", True)
        if pressed:
            Q1 = df_retail.quantile(0.25)
            Q3 = df_retail.quantile(0.75)
            IQR = Q3 - Q1
            with st.echo():
                df_retail = df_retail[~(df_retail > (Q3 + 3 * IQR)).any(axis=1)]
            st.write(df_retail.shape)
            st.write("**Interpretation:** The above output shows that 33 extreme outliers are removed from the data.")
        st.markdown("---")
        
# 5. Transform the numerical variables such that the values will be between 0 and 1.
with q6:
    if sbar=="Day 01":
        st.markdown("** Q.5. Transform the numerical variables such that the values will be between 0 and 1.**")
        pressed = st.button("Q.5. Solution", True)
        if pressed:
            num_var = df_retail.drop(['Region'], axis=1)
            X_norm = MinMaxScaler()
            num_norm = X_norm.fit_transform(num_var)
            X_scaled = pd.DataFrame(num_norm, columns = num_var.columns)
            st.markdown("**DataFrame after MinMax Scaler**")
            st.write(X_scaled.head())
            st.write('Minimum: \n', X_scaled.min())
            st.write('Maximum: \n', X_scaled.max())
        st.markdown("---")
        
# 6. Perform K-Means clustering with varying K from 2 to 4, and identify the optimal number of clusters using the Silhouette plot.
with q7:
    if sbar=="Day 01":
        st.markdown("** Q.6. Perform K-Means clustering with varying K from 2 to 4, and identify the optimal number of clusters using the Silhouette plot.**")
        pressed = st.button("Q.6. Solution", True)
        if pressed:
            n_clusters = [2, 3, 4]
            num_var = df_retail.drop(['Region'], axis=1)
            X_norm = MinMaxScaler()
            num_norm = X_norm.fit_transform(num_var)
            X_scaled = pd.DataFrame(num_norm, columns = num_var.columns)
            X_scaled = np.array(X_scaled)
            for K in n_clusters:
                # create a subplot with 1 row and 2 columns
                fig, ax = plt.subplots(1,1)
            
                # set the figure size
                fig.set_size_inches(15, 8)

                # initialize the cluster with 'K' value and a random generator
                model = KMeans(n_clusters = K, random_state = 10)
                
                # fit and predict on the scaled data
                cluster_labels = model.fit_predict(X_scaled)

                # the 'silhouette_score()' gives the average value for all the samples
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            
                # compute the silhouette coefficient for each sample
                sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
                
                y_lower = 10
            for i in range(K):
                
                # aggregate the silhouette scores for samples belonging to cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                
                # sort the silhouette coefficient
                ith_cluster_silhouette_values.sort()
                
                # calculate the size of the cluster
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                # color each cluster 
                color = cm.nipy_spectral(float(i+1) / K)
                st.write(ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color))

                # label the silhouette plots with their cluster numbers at the middle
                st.write(ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i)))

                # compute the new y_lower for next plot
                y_lower = y_upper + 10 

                # plot the vertical line for average silhouette score of all the values
                # pass the required color and linestyle
                st.write(ax.axvline(x = silhouette_avg, color = "red", linestyle = "--"))

                # clear the y-axis ticks
                ax.set_yticks([])  
                
                # set the ticks for x-axis 
                ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8])

                # set the axes and plot label
                # set the font size using 'fontsize'
                st.write(ax.set_title('Silhouette Plot (K = ' + str(K) + ')', fontsize = 15))
                ax.set_xlabel('Silhouette Coefficient', fontsize = 15)
                ax.set_ylabel('Cluster Label', fontsize = 15)
            plt.show()
        st.markdown("---")
        
# 7. Consider the numerical variables to create two clusters and visualize them using the variables 'Vegetables' and 'Personal_care'.
with q8:
    if sbar=="Day 01":
        st.markdown("** Q.7. Consider the numerical variables to create two clusters and visualize them using the variables 'Vegetables' and 'Personal_care'.**")
        pressed = st.button("Q.7. Solution", True)
        st.caption("Let us perform K-Means clustering for K = 2 using the numerical variables.")
        if pressed:
            num_var = df_retail.drop(['Region'], axis=1)
            X_norm = MinMaxScaler()
            num_norm = X_norm.fit_transform(num_var)
            X_scaled = pd.DataFrame(num_norm, columns = num_var.columns)
            st.write(X_scaled.head())
            clusters = KMeans(n_clusters = 2, random_state = 10)
            clusters.fit(X_scaled)
            df_cluster = df_retail.copy()
            df_cluster['Cluster_id'] = clusters.labels_
            st.caption("DataFrame with Cluster IDs")
            st.write(df_cluster.head())
            fig = pex.scatter(df_cluster, x = 'Vegetables', y = 'Personal_care', color = "Cluster_id")
            st.write(fig)
            st.markdown("Interpretation: The above scatter plot shows that the majority of the retailers in the 1st cluster tends to deal in vegetables over personal care products. Some of the retailers are prone to buy the vegetables as well as personal care products.")
        st.markdown("---")
        
# 8. Draw insights from the clusters formed in the previous question with respect to each variable in the data.
with q9:
    if sbar=="Day 01":
        st.markdown("** Q.8. Draw insights from the clusters formed in the previous question with respect to each variable in the data.**")
        st.caption("In the previous question, we clustered the data into two groups. Now let us retrieve the clusters and interpret them.")
        pressed = st.button("Q.8. Solution", True)
        if pressed:
            num_var = df_retail.drop(['Region'], axis=1)
            X_norm = MinMaxScaler()
            num_norm = X_norm.fit_transform(num_var)
            X_scaled = pd.DataFrame(num_norm, columns = num_var.columns)
            st.info("Evaluating 1st Cluster")
            st.write(X_scaled.head())
            clusters = KMeans(n_clusters = 2, random_state = 10)
            clusters.fit(X_scaled)
            df_cluster = df_retail.copy()
            df_cluster['Cluster_id'] = clusters.labels_
            st.caption("DataFrame with Cluster IDs")
            st.write(df_cluster.head())
            st.write(df_cluster[df_cluster['Cluster_id'] == 0].shape)
            st.caption("Describing df_cluster where the Cluster Id=0")
            st.write(df_cluster[df_cluster['Cluster_id'] == 0].describe())
            var = pd.DataFrame(df_cluster[df_cluster['Cluster_id'] == 0].describe(include = 'object'))
            res =f" The Statistics are: **{var}**"
            st.write(res)
            st.markdown("**Interpretation:** This cluster contains the records of 301 retailers out of which 71% are from Rochester. The above output shows that, majority of the retailers in this cluster are purchasing vegetables from the wholesaler. They are spending the least money on buying the personal care products.")
            st.info("Evaluating 02nd Cluster")
            st.write(df_cluster[df_cluster['Cluster_id'] == 1].shape)
            st.caption("Describing df_cluster where the Cluster Id=1")
            st.write(df_cluster[df_cluster['Cluster_id'] == 1].describe())
            var1 = df_cluster[df_cluster['Cluster_id'] == 1].describe(include = 'object')
            res =f" The Statistics are: **{var1}**"
            st.write(res)
            st.markdown("**Interpretation:** This cluster contains the records of 105 retailers out of which 73% are from Rochester. The above output shows that, majority of the retailers in this cluster are purchasing grocery along with the milk and milk products from the wholesaler. They are spending the least money on buying the plasticware products.")
        st.markdown("---")
        
# 9. Group the retailers from Oneonta into 1 to 5 clusters and find the optimal number of clusters using within cluster sum of squares.
with q10:
    if sbar=="Day 01":
        st.markdown("** Q.9. Group the retailers from Oneonta into 1 to 5 clusters and find the optimal number of clusters using within cluster sum of squares.**")
        pressed = st.button("Q.9. Solution", True)
        if pressed:
            df_Oneonta = df_retail[df_retail['Region'] == 'Oneonta']
            st.write(df_Oneonta.head())
            X_norm = MinMaxScaler()
            num_norm = X_norm.fit_transform(df_Oneonta.drop('Region', axis = 1))
            X_scaled = pd.DataFrame(num_norm, columns = ['Vegetables', 'Personal_care', 'Milk', 'Grocery', 'Plasticware'])
            st.caption("Min Max Scaled Dataset")
            st.write(X_scaled.head())
            wcss  = []
            for i in range(1, 6):
                kmeans = KMeans(n_clusters = i, random_state = 10)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)

            # print the WCSS for each K 
            st.write("WCSS", wcss)
            st.write("Interpretation: We can see that the WCSS is decreasing rapidly for K = 2 and 3. \
                The rate of decrease in WCSS is reduced after K = 3. Thus we can say that K = 3 is the optimal number of clusters.")
        st.markdown("---")

#10. Group the retailers from Oneonta into the optimal number of clusters obtained in Q9. Also, find the number of retailers in each cluster.
with q11:
    if sbar=="Day 01":
        st.markdown("** Q.10. Group the retailers from Oneonta into the optimal number of clusters obtained in Q9. Also, find the number of retailers in each cluster.**")
        pressed = st.button("Q.10. Solution", True)
        if pressed:
            df_Oneonta = df_retail[df_retail['Region'] == 'Oneonta']
            X_norm = MinMaxScaler()
            num_norm = X_norm.fit_transform(df_Oneonta.drop('Region', axis = 1))
            X_scaled = pd.DataFrame(num_norm, columns = ['Vegetables', 'Personal_care', 'Milk', 'Grocery', 'Plasticware'])
            with st.echo():
                clusters = KMeans(n_clusters = 3, random_state = 10)
            clusters.fit(X_scaled)
            df_Oneonta['Cluster'] = clusters.labels_
            st.write(df_Oneonta.Cluster.value_counts())
            st.write("**Interpretation:** The retailers from Oneonta are clustered into 3 groups. \
                The 1st cluster is the largest one with 58 retailers. Both the 2nd and the 3rd cluster contain 13 & 8 retailers each.")
        st.markdown("---")


############################################ D A Y 0 1 ##############################################################################
#***********************************************************************************************************************************#

with header:
    if sbar =="Day 02":
        notes = f'''
        **The first column in the dataset corresponds to the different food items and the remaining columns record the amount of nutrients present in that food.**
        
        The data definition is as follows:
        * **Food:** Name of the food item
        * **Calories:** Calories present in the food (in kcal) 
        * **Fat:** Fat present in the food (in g)
        * **Sodium:** Sodium present in the food (in mg)
        * **Potassium:** Potassium present in the food (in mg)
        * **Carbohydrate:** Carbohydrate present in the food (in g)
        * **Protein:** Protein present in the food (in g)
        * **Vitamin A:** Vitamin A present in the food (in mg)
        * **Vitamin C:** Vitamin C present in the food (in mg)
        * **Calcium:** Calcium present in the food (in mg)
        * **Iron:** Iron present in the food (in mg)'''
        st.write(notes)
    
# Reading the Data...wholesale_cust.csv
df_food= pd.read_csv("data/Nutrients.csv")

#Load the Dataset
with q1:
    if sbar=="Day 02":
        st.markdown("**Load the Dataset**")
        pressed = st.button("View the Data", True)
        if pressed:
            st.write(df_food.head())
            st.write("The Shape of the Data is ", df_food.shape)
        st.markdown("---")  
    
# 1. Set the name of the food item as the identifier for each observation.
with q2:
    if sbar=="Day 02":
        st.markdown("** Q.1. Set the name of the food item as the identifier for each observation.**")
        pressed = st.button("Q.1. Solution", True)
        if pressed:
            with st.echo():
                df_food = df_food.set_index('Food')
            st.write(df_food.head())
        st.markdown("---")

# 2. Plot the distribution of all the numerical variables and identify the type of skewness.
with q3:
    if sbar=="Day 02":
        st.markdown("** Q.2. Plot the distribution of all the numerical variables and identify the type of skewness.**")
        pressed = st.button("Q.2. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')
            nums = df_food.select_dtypes(include = np.number)
            cols = ['Calories', 'Fat', 'Sodium', 'Potassium', 'Carbohydrate', 'Protein','Vitamin A', 'Vitamin C']
            st.write(nums.skew().reset_index(name = "Skewness"))
            a = int(len(cols)/2) # no of rows
            b = 2 # no of columns
            c = 1 # initiate counter

            plt.figure(figsize=(12,8))
            for i in cols: # Loop Runs on columns
                plt.subplot(a,b,c) # Defining the 
                 # printing the Title
                fig = plt.figure()
                plt.hist(x = nums.loc[:,i], color = "coral") # Plotting the Distplot
                c = c+1
                st.write(i)
                pass
                #plt.tight_layout()
                st.plotly_chart(fig)    
            st.write("**Interpretation:** The above histograms show that all the variables in the data are **positively skewed.**")
        st.markdown("---")
# 3. Plot the correlation between the various nutrients. And find the variables with the strongest correlation.      

with q4:
    if sbar=="Day 02":
        st.markdown("** Q.3. Plot the correlation between the various nutrients. And find the variables with the strongest correlation.**")
        st.caption("Note - The Plot will open in other window as the Size is Large")
        pressed = st.button("Q.3. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')
            nums = df_food.select_dtypes(include = np.number)
            plt.figure(figsize = [15,10])
            fig = ff.create_annotated_heatmap(np.array(nums.corr()), x =list(nums.columns), y =list(nums.columns), colorscale='Viridis')
            fig.show()
            
            st.write('''
                    * **Interpretation:** The diagonal entries show the correlation of a variable with itself; thus, it is always equal to 1. The strongest **positive correlation** is between **Protein and Calories** (i.e. 0.8).
                    * No two variables have the strongest negative correlation. But, there is a **moderate negative correlation** between **Protein and Carbohydrates** (i.e. -0.66).
                    ''')
        st.markdown("---")
        
# 4. Perform the appropriate normalization technique to transform the variables to have minimum 0 and maximum 1 value.
with q5:
    if sbar=="Day 02":
        st.markdown("** Q.4. Perform the appropriate normalization technique to transform the variables to have minimum 0 and maximum 1 value.**")
        #st.caption("Note - The Plot will open in other window as the Size is Large")
        pressed = st.button("Q.4. Solution", True)
        if pressed:
            scaler = MinMaxScaler()
            df_food = df_food.set_index('Food')
            scaled_data = scaler.fit_transform(df_food)
            df_scaled = pd.DataFrame(scaled_data, columns = df_food.columns)
            st.write(df_scaled.describe())
            st.write('''
                     **Interpretation:** The above statistical summary shows that, the minimum and maximum value of each variable is 0 and 1 respectively. \
                         Thus, we have transformed all the variabels to the same scale.''')
        st.markdown("---")

# 5. Create a dictionary to store the cophenetic correlation coefficient for the following linkage methods: 'Single', 'Complete', and 'Average'. Identify which linkage method works best in quantifying the dissimilarities between the observations.

with q6:
    if sbar=="Day 02":
        st.markdown("** Q.5. Create a dictionary to store the cophenetic correlation coefficient for the following linkage methods: 'Single', 'Complete', and 'Average'. Identify which linkage method works best in quantifying the dissimilarities between the observations.**")
        #st.caption("Note - The Plot will open in other window as the Size is Large")
        pressed = st.button("Q.5. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_food)
            df_scaled = pd.DataFrame(scaled_data, columns = df_food.columns)
            single_link = linkage(df_scaled, method = 'single')  
            complete_link = linkage(df_scaled, method = 'complete')
            avg_link = linkage(df_scaled, method = 'average')
            out_dict = {}
            eucli_dist = euclidean_distances(df_scaled)
            dist_array = eucli_dist[np.triu_indices(61, k = 1)]
            coeff_single, cophenet_dist = cophenet(single_link, dist_array)
            coeff_comp, cophenet_dist = cophenet(complete_link, dist_array)
            coeff_avg, cophenet_dist = cophenet(avg_link, dist_array)
            out_dict.update({'Single Linkage': coeff_single, 
                 'Complete Linkage': coeff_comp,
                 'Average Linkage': coeff_avg})
            st.write(out_dict)
            st.write("**Interpretation:** The above dictionary represents the cophenetic correlation coefficient for different linkage \
                methods. The **coefficient for the average linkage method is the highest.** Thus, we can conclude that this linkage method works best in quantifying the dissimilarities between the observations among the three linkages.")
            
        st.markdown("---")

# 6. Plot the dendrogram using the 'ward' linkage method and decide the optimal number of clusters.

with q7:
    if sbar=="Day 02":
        st.markdown("** Q.6. Plot the dendrogram using the 'ward' linkage method and decide the optimal number of clusters.**")
        #st.caption("Note - The Plot will open in other window as the Size is Large")
        pressed = st.button("Q.6. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_food)
            df_scaled = pd.DataFrame(scaled_data, columns = df_food.columns)
            linkage_mat = linkage(df_scaled, method = 'ward')
            fig = ff.create_dendrogram(linkage_mat)
            fig.update_layout(width=800, height=500, title = "Dendogram-Ward Linkage", xaxis_title = "Observation",yaxis_title = "Distance")
            st.plotly_chart(fig)
            st.write("**Interpretation:** The different clusters are represented by green and red color. Finally, these two clusters are merging into a single cluster. The dendrogram shows that the number of clusters remains 2 for the maximum distance. Thus, we can conclude that the optimal number of clusters is 2.")
        st.markdown("---")
# 7. Build the optimal number of clusters as per the previous question and interpret them.

with q8:
    if sbar=="Day 02":
        st.markdown("** Q.7. Build the optimal number of clusters as per the previous question and interpret them.**")
        st.caption("Note - Try Agglomerative Clustering. Linkage = Ward, Clusters = 2")
        pressed = st.button("Q.7. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_food)
            df_scaled = pd.DataFrame(scaled_data, columns = df_food.columns)
            clustering = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
            clustering.fit(df_scaled)
            df_food['Cluster'] = clustering.labels_
            st.subheader("Cluster 01 ")
            st.write("Cluster 01 Statistics: ", df_food[df_food.Cluster==0].describe())
            list_items = f"Food Items in the Cluster: **{list(df_food[df_food.Cluster==0].index)}**"
            st.write(list_items)
            st.write("**Interpretation:** This cluster contains 21 food items with high amount of nutrients like proteins, vitamins and minerals (sodium, calcium, iron, potassium). Also these are high calorie products.This cluster represents the seafood variety (except for spinach). Thus, we can segment this group under Seafood. We can see that spinach is included in this cluster, as it is a prominent source of vitamins and minerals.")
            
            st.subheader("Cluster 02 ")
            st.write("Cluster 02 Statistics: ", df_food[df_food.Cluster==1].describe())
            list_items = f"Food Items in the Cluster: **{list(df_food[df_food.Cluster==1].index)}**"
            st.markdown(list_items)
            st.write("**Interpretation:** This cluster contains 40 food items with high amount of nutrients like carbohydrates, vitamins and minerals (calcium, potassium).This cluster represents variety of fruits and vegetables. Thus, we can segment this group under Fruits & Vegetables.")
            
        st.markdown("---")

#8. Perform a DBSCAN algorithm, where a point is in the neighborhood of another point if the euclidean distance between them is less than 0.6, and a core point should have at least 2 points in its neighborhood (excluding itself). And find the number of data points in each cluster.

with q9:
    if sbar=="Day 02":
        st.markdown("** Q.8. Perform a DBSCAN algorithm, where a point is in the neighborhood of another point if the euclidean distance between them is less than 0.6, and a core point should have at least 2 points in its neighborhood (excluding itself). And find the number of data points in each cluster.**")
        st.caption("Note - eps = 0.6, min_samples = 3")
        pressed = st.button("Q.8. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_food)
            df_scaled = pd.DataFrame(scaled_data, columns = df_food.columns)
            model = DBSCAN(eps = 0.6, min_samples = 3)
            st.write(model.fit(df_scaled))
            st.write(set(model.labels_))
            st.write("**Interpretation:** The algorithm has created 2 clusters (with labels 0 and 1), and some points are identied as outliers (with label -1).")
            df_food['Cluster_DBSCAN'] = model.labels_
            st.write(df_food['Cluster_DBSCAN'].value_counts())
            st.write("**Interpretation:** The 1st cluster formed by DBSCAN includes 40 observations, 2nd cluster contains 16 observations and 5 observations are marked as outliers.")
        st.markdown("---")

#9. Interpret the outliers identified by DBSCAN algorithm.

with q10:
    if sbar=="Day 02":
        st.markdown("** Q.9. Perform a DBSCAN algorithm, where a point is in the neighborhood of another point if the euclidean distance between them is less than 0.6, and a core point should have at least 2 points in its neighborhood (excluding itself). And find the number of data points in each cluster.**")
        st.caption("Note - eps = 0.6, min_samples = 3")
        pressed = st.button("Q.9. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_food)
            df_scaled = pd.DataFrame(scaled_data, columns = df_food.columns)
            model = DBSCAN(eps = 0.6, min_samples = 3)
            st.write(model.fit(df_scaled))
            st.write(set(model.labels_))
            #st.write("**Interpretation:** The algorithm has created 2 clusters (with labels 0 and 1), and some points are identied as outliers (with label -1).")
            df_food['Cluster_DBSCAN'] = model.labels_
            #st.write(df_food['Cluster_DBSCAN'].value_counts())
            
            with st.echo():
                df_food[df_food.Cluster_DBSCAN==-1]
            st.write("""
                     * **Interpretation:** The above dataframe shows the outliers identified by DBSCAN. Out of 5 observarions, 4 belong to seafood. These observations have extreme values for at least one of the variables.
                     * The **Iron content in Octopus is the highest**, **Sodium** content is the **highest in Blue Crab**, \
                         **Lobster** has **highest Calories and Protein content**, \
                             **Spinach** is richest in **Potassium, Carbohydrates, Vitamin C & Calcium**; and \
                                 **Rainbow Trout** has the highest amount of **Vitamin A.**
                     """)
        st.markdown("---")

# 10. Visualize the clusters formed using hierarchical clustering and DBSCAN.
from plotly.subplots import make_subplots
import plotly.graph_objects as go

with q11:
    if sbar=="Day 02":
        st.markdown("** Q.10. Perform a DBSCAN algorithm, where a point is in the neighborhood of another point if the euclidean distance between them is less than 0.6, and a core point should have at least 2 points in its neighborhood (excluding itself). And find the number of data points in each cluster.**")
        st.caption("Approach: Generate a Dataframe with Agglomerative & DBSCAN Clustering. The Column Names are 'Cluster' & 'Cluster_DBSCAN'")
        pressed = st.button("Q.10. Solution", True)
        if pressed:
            df_food = df_food.set_index('Food')

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df_food)
            df_scaled = pd.DataFrame(scaled_data, columns = df_food.columns)
            clustering = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
            clustering.fit(df_scaled)
            df_food['Cluster'] = clustering.labels_
            model = DBSCAN(eps = 0.6, min_samples = 3)
            #st.write("**Interpretation:** The algorithm has created 2 clusters (with labels 0 and 1), and some points are identied as outliers (with label -1).")
            model.fit(df_scaled)
            df_food['Cluster_DBSCAN'] = model.labels_
            st.write(df_food)
            st.markdown("**Cluster Wise Scatterplots**")
            fig1=pex.scatter(df_food, x = 'Protein', y = 'Calcium', color = "Cluster").update_layout(title = "Hierarchical Clustering",
                                                                                                     xaxis_title = "Protein",
                                                                                                     yaxis_title = "Calcium")
            fig2 = pex.scatter(df_food, x = 'Protein', y = 'Calcium', color = "Cluster_DBSCAN").update_layout(title = "DBSCAN Clustering",
                                                                                                     xaxis_title = "Protein",
                                                                                                     yaxis_title = "Calcium")
            st.plotly_chart(fig1) 
            st.plotly_chart(fig2)
            st.write("""
                     **Interpretation:** The yellow & red dots in both the subplots represent the largest cluster formed by algorithms. \
                         The blue ones represent the second cluster formed in Chart 01. The blue circles in the DBSCAN clustering represent the outliers. These outliers are grouped in the second cluster by hierarchical clustering.""")
        st.markdown("---")
############################################ D A Y 0 3 ##############################################################################
#***********************************************************************************************************************************#

#Load the Dataset
with header:
    if sbar =="Day 03":
        notes = f'''
        **The data includes the information about the various factors affecting the popularity of the movies.**
        
        The data definition is as follows:
        * **Language:** The language in which the movie is produced
        * **Country:** The name of the country in which the movie is produced
        * **Duration:** Time duration of the movie in minutes
        * **Budget:** Overall cost to produce the movie in dollars
        * **Gross_Earnings:** Box office collection of the movie
        * **Facebook_Likes_Director:** Likes on Facebook for director of the movie
        * **Facebook_Likes_Actor:** Likes on Facebook for an actor of the movie
        * **Facebook_Likes_Music:** Likes on Facebook for the music of the movie
        * **Facebook_Likes_Actress:** Likes on Facebook for an actress of the movie
        * **Facebook_Likes_Total_Cast:** Likes on Facebook for all the characters in the movie
        * **Facenumber_in_posters:** Number of faces on the movie poster
        * **User_upvotes:** Upvotes given by the users
        * **Reviews_by_Users:** Number of reviews given by the users
        * **IMDB_Score:** Internet Movie Database (IMDB) score of the movie which is based on individual ratings'''
        st.write(notes)
    
# Reading the Data...wholesale_cust.csv
df_movies= pd.read_csv("data/Movies_Data.csv")

with q1:
    if sbar=="Day 03":
        st.markdown("**Load the Dataset**")
        pressed = st.button("View the Data", True)
        if pressed:
            st.write(df_movies.head())
            st.write("The Shape of the Data is ", df_movies.shape)
        st.markdown("---")  
        
# 1. 1. Find the dimension of the data and check if there are any missing values present in the data.
with q2:
    if sbar=="Day 03":
        st.markdown("** Q.1. Find the dimension of the data and check if there are any missing values present in the data.**")
        pressed = st.button("Q.1. Solution", True)
        if pressed:
            st.write("Dimensions of the Data", df_movies.shape)
            st.write(df_movies.isnull().sum())
            st.write("**Interpretation:** The above output shows that there are no missing values present in the data.")
        st.markdown("---")
 
#2. Check the data type of each variable and consider only the numerical variables for the further analysis.       
with q3:
    if sbar=="Day 03":
        st.markdown("** Q.2. Check the data type of each variable and consider only the numerical variables for the further analysis.**")
        pressed = st.button("Q.2. Solution", True)
        if pressed:
            num_cols = df_movies.select_dtypes(include = np.number).columns
            cat_cols = df_movies.select_dtypes(exclude = np.number).columns
            
            num_ = f"Numerical Columns are **{list(num_cols)}**"
            cat_ = f"Cat Columns are **{list(cat_cols)}**"
            st.markdown(num_)
            st.markdown(cat_)
            st.write("**Interpretation:** The above output shows that the 'Language' and 'Country' are categorical variables whereas all other variables are numerical.")
            st.caption("Dropping the Categorical Columns")
            with st.echo():
                df_movie_num = df_movies.drop(['Language', 'Country'], axis = 1)
            st.write(df_movie_num.head())
            st.write("Shape after removal of 02 Columns", df_movie_num.shape)
        st.markdown("---")

df_movie_num = df_movies.drop(['Language', 'Country'], axis = 1)

#3. Find the summary statistics of each variable.
with q4:
    if sbar=="Day 03":
        st.markdown("** Q.3. Find the summary statistics of each variable.**")
        pressed = st.button("Q.3. Solution", True)
        if pressed:
            st.write(df_movie_num.describe())
            st.write("""
                     **Interpretation:**
                     * The average duration of a movie is 112 minutes.
                     * The average production cost of the movie is 28064254 dollars. There is so much variation between the production cost as the standard deviation is 84671282 dollars.
                     * The average facebook likes for all the characters in the movie is 7834.
                     * The 50% of the box office collection of the movies is between 8882564 dollars and 66815566 dollars. The average production cost is less than the gross earnings of the movie.
                     """)
        st.markdown("---")
            
# 4. Check if there are any outliers present in the data and visualize using boxplot.

with q5:
    if sbar=="Day 03":
        st.markdown("** Q.4. Check if there are any outliers present in the data and visualize using boxplot.**")
        pressed = st.button("Q.4. Solution", True)
        if pressed:
            cols = df_movie_num.columns
            st.write(df_movie_num.skew().reset_index(name = "Skewness"))
            a = int(len(cols)/2) # no of rows
            b = 2 # no of columns
            c = 1 # initiate counter

            plt.figure(figsize=(12,8))
            for i in cols: # Loop Runs on columns
                plt.subplot(a,b,c) # Defining the 
                 # printing the Title
                fig = plt.figure()
                sns.boxplot(df_movie_num.loc[:,i]) # Plotting the Distplot
                c = c+1
                st.write(i)
                pass
                #plt.tight_layout()
                st.pyplot(fig)    
            st.write("**Interpretation:** From the boxplots above we can see that there are outliers in the data. We need to remove these outliers before further analysis.**")
        st.markdown("---")


# 5. Remove the outliers using 1.5IQR.

with q6:
    if sbar=="Day 03":
        st.markdown("** Q.5. Remove the outliers using 1.5 * IQR.**")
        pressed = st.button("Q.5. Solution", True)
        if pressed:
            Q1 = df_movie_num.quantile(0.25)
            Q3 = df_movie_num.quantile(0.75)
            IQR = Q3 - Q1
            df_movie_num = df_movie_num[~((df_movie_num < (Q1 - 1.5 * IQR)) | (df_movie_num > (Q3 + 1.5 * IQR))).any(axis=1)]
            st.write(df_movie_num.shape)
            st.write("**Interpretation:** From the dimension of the data, we can see that there are 816 observations and 12 variables in the data. Thus, we have removed the outliers.")
        st.markdown("---")
Q1 = df_movie_num.quantile(0.25)
Q3 = df_movie_num.quantile(0.75)
IQR = Q3 - Q1
df_movie_num = df_movie_num[~((df_movie_num < (Q1 - 1.5 * IQR)) | (df_movie_num > (Q3 + 1.5 * IQR))).any(axis=1)]

# 6. Transform the variables in the standard form such that they will have mean 0 and standard deviation 1 and get the summary.
with q7:
    if sbar=="Day 03":
        st.markdown("** Q.6. Transform the variables in the standard form such that they will have mean 0 and standard deviation 1 and get the summary.**")
        pressed = st.button("Q.6. Solution", True)
        if pressed:
            scaler = StandardScaler()
            with st.echo():
                features_scaled = scaler.fit_transform(df_movie_num) # Movies Data Post Outlier Removal
                summary = pd.DataFrame(features_scaled, columns = df_movie_num.columns) # Scaled Dataset
            st.write(summary.describe())
            st.write("**Interpretation:** Here all the variables have a mean 0 and standard deviation of 1. Now they are in the standard form.")
            st.caption("Now the Data is Ready for PCA")
        st.markdown("---")

# 7. Obtain a covariance matrix and find the eigenvalues. Hence arrange the eigenvalues in decreasing order.
with q8:
    if sbar=="Day 03":
        st.markdown("** Q.7. Obtain a covariance matrix and find the eigenvalues. Hence arrange the eigenvalues in decreasing order.**")
        pressed = st.button("Q.7. Solution", True)
        if pressed:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df_movie_num)
            trans_features_scaled = np.transpose(features_scaled)
            cov_matrix = np.cov(trans_features_scaled)
            st.caption("Covariance Matrix")
            st.write(cov_matrix)
            eig_values, eig_vectors = lin.eig(cov_matrix)
            st.write("Eigen Values", eig_values)
            eig_values = eig_values.tolist()
            eig_values_sorted = sorted(eig_values ,reverse = True)
            st.write("Soted Eigen Values", eig_values_sorted)
            st.write("**Interpretation:** The above list represents the eigenvalues in decreasing order. The highest eigenvalue is 3.2822 and the lowest value is 0.0029.")
        st.markdown("---")  
            
# 8. Draw the scree plot and obtain the optimal number of components. Hence find the percentage of variation explained by the optimal number of components.
with q9:
    if sbar=="Day 03":
        st.markdown("** Q.8. Draw the scree plot and obtain the optimal number of components. Hence find the percentage of variation explained by the optimal number of components.**")
        pressed = st.button("Q.8. Solution", True)
        if pressed:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df_movie_num)
            trans_features_scaled = np.transpose(features_scaled)
            cov_matrix = np.cov(trans_features_scaled)
            #st.caption("Covariance Matrix")
            #st.write(cov_matrix)
            eig_values, eig_vectors = lin.eig(cov_matrix)
            #st.write("Eigen Values", eig_values)
            eig_values = eig_values.tolist()
            eig_values_sorted = sorted(eig_values ,reverse = True)
            fig = plt.figure()
            plt.plot(eig_values_sorted)
            plt.title("Scree Plot for Optimal Number of Components")
            plt.ylabel("Eigenvalues")
            plt.xlabel("# of Components")
            st.pyplot(fig)
            st.write("**Interpretation:** From the above scree plot, we can see an elbow point is for number of components = 3. Note that, after the elbow point, the principal components do not contribute much to the variance in the data.")
            st.caption("Find the percentage of variation explained by the first three components.")
            with st.echo():
                p1 = eig_values_sorted[0] / sum(eig_values_sorted)
            st.write('Percentage of variation explained by 1st component', p1)
            p2 = eig_values_sorted[1] / sum(eig_values_sorted)
            st.write('Percentage of variation explained by 2nd component', p2)
            p3 = eig_values_sorted[2] / sum(eig_values_sorted)
            st.write('Percentage of variation explained by 3rd component', p3)
            st.write('percentage of variation explained by the first three components', p1+p2+p3)
            st.write("**Interpretation: Here the 1st three principal components explain approximately 55% of the variation in the data.**")
        st.markdown("---")
            
# 9. Perform the principal component analysis using python libraries with the optimal number of components obtained in the previous question.
with q10:
    if sbar=="Day 03":
        st.markdown("** Q.9. Perform the principal component analysis using python libraries with the optimal number of components obtained in the previous question.**")
        st.caption("Use: pca = PCA(n_components = 3, random_state = 10)")
        pressed = st.button("Q.9. Solution", True)
        if pressed:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df_movie_num)
            pca = PCA(n_components = 3, random_state = 10)
            components = pca.fit_transform(features_scaled)
            df_pca = pd.DataFrame(data = components, columns = ['principal_component 1','principal_component 2','principal_component 3'])
            st.write(df_pca.head())
            st.write("**Interpretation: The above output returns the required principal components. Here the number of variables in the data is reduced from 12 to 3.**")
        st.markdown("---")

# 10. Check the equality of the equation Ax =  λ x, where A is the covariance matrix,  λ  is the largest eigenvalue of A and x is the eigenvector corresponding  λ .

with q11:
    if sbar=="Day 03":
        st.markdown("** Q.10. Check the equality of the equation Ax =  λ x, where A is the covariance matrix,  λ  is the largest eigenvalue of A and x is the eigenvector corresponding  λ .**")
        st.caption("In Q7, we obtained the eigenvalues and eigenvectors of the covariance matrix. To check the equality of the given equation, let us first calculate the left hand side and right hand side of the equation separately; and then equate both sides.")
        pressed = st.button("Q.10. Solution", True)
        if pressed:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(df_movie_num)
            trans_features_scaled = np.transpose(features_scaled)
            cov_matrix = np.cov(trans_features_scaled)
            eig_values, eig_vectors = lin.eig(cov_matrix)
            #st.write("Eigen Values", eig_values)
            eig_values = eig_values.tolist()
            eig_values_sorted = sorted(eig_values ,reverse = True)
            
            LHS = np.dot(cov_matrix, eig_vectors[:,0])
            RHS = eig_values[0]*eig_vectors[:,0]
            lhs = f" Checking Equality of LHS vs RHS: **{LHS}**"
            rhs = f" Checking Equality of LHS vs RHS: **{RHS}**"
            st.write(lhs)
            st.write(rhs)
            st.write("**Interpretation: The above output shows that the equation on the left hand side (i.e. Ax) and the equation on the right hand side (i.e. 𝜆x) has same elements in the array. Thus, we can say that Ax = 𝜆x where A is the covariance matrix, 𝜆 is the largest eigenvalue of A and x is the eigenvector corresponding to 𝜆.**")
        st.markdown("---")
