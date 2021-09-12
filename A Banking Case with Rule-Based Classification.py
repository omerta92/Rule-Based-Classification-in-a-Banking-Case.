
# /*\*/*\*/*\*/*\*/*\*/*\*/*\*/*\ Importing Libraries /*\*/*\*/*\*/*\*/*\*/*\*/*\*/*\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# /*\*/*\*/*\*/*\*/*\*/*\*/*\*/*\ Reading Datasets /*\*/*\*/*\*/*\*/*\*/*\*/*\*/*\
df = pd.read_csv("C:/Users/omery/Desktop/Medium/Rule-Based Classification/BankCustomerData.csv")

pd.set_option('display.max_columns',None)

# /*\*/*\*/*\* Descriptive Analysis *\*/*\*/*\*/

def check_df(dataframe, head=5, tail=5, quan=True):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, head=5, tail=5)

# /*\*/*\*/*\* Selection of Categorical and Numerical Variables *\*/*\*/*\*/

def grab_col_names(dataframe, cat_th=10, car_th=30):
    """
        This function to perform the selection of numeric and categorical variables in the data set in a parametric way.
        Note: Variables with numeric data type but with categorical properties are included in categorical variables.
        Parameters
        ----------
        dataframe: dataframe
            The data set in which Variable types need to be parsed
        cat_th: int, optional
            The threshold value for number of distinct observations in numerical variables with categorical properties.
            cat_th is used to specify that if number of distinct observations in numerical variable is less than
            cat_th, this variables can be categorized as a categorical variable.
        car_th: int, optional
            The threshold value for categorical variables with  a wide range of cardinality.
            If the number of distinct observations in a categorical variables is greater than car_th, this
            variable can be categorized as a categorical variable.
        Returns
        -------
            cat_cols: list
                List of categorical variables.
            num_cols: list
                List of numerical variables.
            cat_but_car: list
                List of categorical variables with  a wide range of cardinality.
        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))
        Notes
        ------
            Sum of elements in lists the cat_cols,num_cols  and  cat_but_car give the total number of variables in dataframe.
        """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# /*\*/*\*/*\* General View to Categorical Datas *\*/*\*/*\*/

def cat_summary(dataframe):
    # cat_cols = grab_col_names(dataframe)["Categorical_Data"]
    for col_name in cat_cols:
        print("############## Frequency & Percentage Values of Categorical Datas ########################")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))

cat_summary(df)

# /*\*/*\*/*\* General View to Numerical Datas *\*/*\*/*\*/

num_cols = [col for col in df.columns if df[col].dtypes !="O" and col not in ["day","campaign","pdays","previous"]]
numerical_col = num_cols

def num_summary(dataframe, numerical_col):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print("########## Summary Statistics of " + numerical_col + " ############")
    print(dataframe[numerical_col].describe(quantiles).T)

for col in num_cols:
    num_summary(df, col)

# /*\*/*\*/*\* Data Analysis *\*/*\*/*\*/

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

def data_analysis(dataframe):
    # mean, min, max amount of balance, last transaction and age by job, marital and education.
    print(dataframe.groupby(["job","marital","education"]).agg({"balance" : ["mean","min","max"],
             "last_transaction" : ["mean","min","max"],
             "age" : ["mean","min","max"]}))


data_analysis(df)

# /*\*/*\*/*\* Defining Personas *\*/*\*/*\*/

agg_df = df.groupby(["job","marital","education","age"]).agg({"balance" : "mean",
                                                              "last_transaction" : "mean"})

agg_df = agg_df.reset_index()

age_list = ["0_17","18_24","25_34","35_44","45_54","55_69","70_90"]
agg_df["age_cat"] = pd.cut(agg_df["age"], [0,18,25,35,45,55,70,90], labels = age_list)



# /*\*/*\*/*\* Creating Rule-Based Personas *\*/*\*/*\*/

agg_df["customer_level_based"] = [row[0].upper()+"_"+ row[1].upper()+"_"+ row[2].upper()+"_"+ str(row[6]).upper() for
                                  row in agg_df.values]


del_list = ["job", "marital", "education", "age", "age_cat"]

agg_df.drop(del_list, axis=1, inplace=True)

agg_df = agg_df.groupby(agg_df["customer_level_based"]).agg({"balance": "mean",
                                                             "last_transaction" : "mean" }).sort_values(("balance"),
                                                                                                        ascending=False)

agg_df.reset_index(inplace=True)

# /*\*/*\*/*\* Rule-Based Segmentation *\*/*\*/*\*/
agg_df["segment"]= pd.qcut(agg_df["balance"], 4, labels=["D","C","B","A"])

seg= agg_df.groupby(agg_df["segment"]).agg({"balance" : ["mean", "max", "sum"],
                                            "last_transaction" : ["mean", "max", "min"]})


seg.head().sort_values(("segment"),ascending=False)
seg.reset_index(inplace=True)


# /*\*/*\*/*\* Prediction *\*/*\*/*\*/

lost_customer = "SERVICES_SINGLE_SECONDARY_45_54"

agg_df[agg_df["customer_level_based"] == lost_customer]
seg.head()

