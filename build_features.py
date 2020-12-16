import numpy as np
from numpy import cov
from numpy.linalg import eig
import pandas as pd
import sys
from scipy.spatial import KDTree





def get_the_three_closet_points(point):

    NDIM = 3 # number of dimensions

    # read points into array
    a = np.fromfile('three_points_extracted.txt', sep=' ', dtype=float)
    a.shape = int(a.size / NDIM), NDIM                                                                      
    
    # find 3 nearest points controlled by k
    tree = KDTree(a, leafsize=a.shape[0]+1)
    distances, ndx = tree.query([point], k=3)

    # print 10 nearest points to the chosen one
    top_three_closest_points =  a[ndx][0]

    return(top_three_closest_points)


orig_df = pd.read_csv("Vaihingen3D_Traininig.csv")

nrows = len(orig_df) #change the numebr if you would like to limit your dataset to certain number - sample size

df = orig_df.iloc[0:nrows,0:3]



global_min_height = orig_df["Z"].min()    #total 1 million dataset min
global_max_height = orig_df["Z"].max()    #total 1 million datatset max

extracted_min_height = df["Z"].min() #min height in the extracted - sample. local means only three points closest
extracted_max_height = df["Z"].max() #max height in the extracted - sample. local means only three points closest


df["lambda1"]=""
df["lambda2"]=""
df["lambda3"]=""
df["sum_of_evs"]=""
df["omnivariance"]=""
df["eigenotropy"]=""
df["anisotropy"]=""
df["planarity"]=""
df["linearity"]=""
df["surface_variation"]=""
df["sphericity"]=""
df["local_vertical_range"]=""
df["local_height_below"]=""
df["local_height_above"]=""
df["global_height_above"]=""
df["global_height_below"]=""
df["local_avg_height"]=""
df["local_average_above_or_below"]=""

df_len = len(df)

for row in range(df_len):
    print(row)
    point = list(df.iloc[row,:3])
    #print(point)

    top_three_closest_points = get_the_three_closet_points(point)   
    covariance_matrix = cov(top_three_closest_points)
    eigen_values, eigen_vectors = eig(covariance_matrix)
    #print("Eigen Values")
    #print(eigen_values)

    #print("Eigen Vectors")
    #print(eigen_vectors)

    l1,l2,l3 = eigen_values[0],eigen_values[1],eigen_values[2]

    sum_of_evs = l1+l2+l3

    omnivariance = (l1*l2*l3) ** (1. / 3)
    #print("Omnivariance",omnivariance)

    eigenotropy = -((l1*np.log(l1))+(l2*np.log(l2))+(l3*np.log(l3)))
    #print("eigenotropy",eigenotropy)

    anisotropy = (l1-l3)/l1
    #print("anisotropy",anisotropy)

    planarity = (l2-l3)/l1
    #print("planarity",planarity)

    linearity = (l1-l2)/l1
    #print("linearity",linearity)

    surface_variation = l3*(l1+l2+l3)
    #print("surface_variation",surface_variation)

    sphericity = l3/l1
    #print("sphericity",sphericity)

    top_three_z = list(top_three_closest_points[:,2])

    local_vertical_range = max(top_three_z) - min(top_three_z)
    #print("local_vertical_range",local_vertical_range)

    local_height_below = point[2] - min(top_three_z)
    #print("local_height_below",local_height_below)

    local_height_above = max(top_three_z) - point[2]
    #print("local_height_above",local_height_above)

    global_height_above = point[2] - global_min_height
    #print("global_height_above",global_height_above)

    global_height_below = global_max_height - point[2]
    #print("global_height_below",global_height_below)

    local_avg_height = sum(top_three_z)/3
    #print("local_avg_height",local_avg_height)

    local_average_above_or_below = local_avg_height - point[2]
    #print("local_average_above_or_below",local_average_above_or_below)

    extracted_height_above = point[2] - extracted_min_height 

    extracted_height_below = extracted_max_height - point[2]

    
    df.loc[row,"lambda1"]=l1
    df.loc[row,"lambda2"]=l2
    df.loc[row,"lambda3"]=l3
    df.loc[row,"sum_of_evs"]=sum_of_evs
    df.loc[row,"omnivariance"]=omnivariance
    df.loc[row,"eigenotropy"]=eigenotropy
    df.loc[row,"anisotropy"]=anisotropy
    df.loc[row,"planarity"]=planarity
    df.loc[row,"linearity"]=linearity
    df.loc[row,"surface_variation"]=surface_variation
    df.loc[row,"sphericity"]=sphericity
    df.loc[row,"local_vertical_range"]=local_vertical_range
    df.loc[row,"local_height_below"]=local_height_below
    df.loc[row,"local_height_above"]=local_height_above
    df.loc[row,"global_height_above"]=global_height_above
    df.loc[row,"global_height_below"]=global_height_below
    df.loc[row,"extracted_height_above"]= extracted_height_above
    df.loc[row,"extracted_height_below"]= extracted_height_below

    df.loc[row,"local_avg_height"]=local_avg_height
    df.loc[row,"local_average_above_or_below"]=local_average_above_or_below


    if (row==100 or row==10000 or row==20000 or row==30000 or row==40000 or row==50000 or row==60000 or row==70000 or row==75000 or row==100000 or row==150000 or row==200000 or row==250000 or row==300000 or row==350000 or row==400000 or row==450000 or row==500000 or row==550000 or row==600000 or row==650000 or row==700000 or row==750000):
        df.to_csv("dataset_"+str(row)+".csv",index=False)
    


df["Intensity"] = orig_df.loc[:nrows,"Intensity"]
df["return_number"] = orig_df.loc[:nrows,"return_number"]
df["number_of_returns"] = orig_df.loc[:nrows,"number_of_returns"]
df["label"] = orig_df.loc[:nrows,"label"]



df.to_csv("feature_dataset.csv",index=False)
