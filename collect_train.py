from collect import *

pos_path = '2017\\train\\positive_examples_anonymous_chunks'
neg_path = '2017\\train\\negative_examples_anonymous_chunks'

# Load the data as dictionary in the form {ID:TEXTS} 
pos_data = getData(getXmlFromPath(pos_path))
neg_data = getData(getXmlFromPath(neg_path))
print("Number of negative examples: ",len(neg_data))
print("Number of positive examples: ",len(pos_data))

# Create the dataframe for positve examples
df_pos = pd.DataFrame(list(pos_data.items()), columns =['Id','Text'])
df_pos['Target'] = 1
print(df_pos.head())
print(df_pos.shape)

# Create the dataframe for negative examples
df_neg = pd.DataFrame(list(neg_data.items()), columns =['Id','Text'])
df_neg['Target'] = 0
print(df_neg.head())
print(df_neg.shape)

# Concat both data frames into one
df_train = pd.concat([df_neg,df_pos], ignore_index=True)
print(df_train.head())
print(df_train.shape)

# Save as train.csv 
df_train.to_csv('train.csv')