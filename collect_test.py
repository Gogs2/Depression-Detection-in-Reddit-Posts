from collect import *

data_path = "2017\\test"
targets_path = "2017\\test\\test_golden_truth.txt"

# Load the data as dictionary in the form {ID:TEXTS} 
data = getData(getXmlFromPath(data_path))

# Load the targets as dictionary in the form {ID:TARGET}
targets_file = open(targets_path,'r')
contents = targets_file.readlines()
targets = dict()
for line in contents[:-1]:
    line = line.replace('\n','')
    tokens = line.split('\t')
    tokens[0]=tokens[0].strip()
    targets[tokens[0]] = tokens[1]

# Create DataFrame for the text from subjects
df_text = pd.DataFrame(list(data.items()),columns=['Id','Text'])
print(df_text.head())

# Create DataFrame for target class from subjects, 1 = depressed, 0 = not depressed
df_target = pd.DataFrame(list(targets.items()),columns=['Id','Target'])
print(df_target.head())

# Merge the two DataFrames on Id 
df_merged = pd.merge(df_text,df_target,on='Id')
print(df_merged.head())

# Save to .csv file
# drop one subject because he had no text in his posts
df_merged = df_merged[df_merged.Id != 'test_subject2476']
print(df_merged.shape)
df_merged.to_csv('test.csv')
