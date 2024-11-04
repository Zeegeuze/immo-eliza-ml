import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler

def import_df(valid=True):
    '''
    This will take the csv from the data folder called raw_data.csv, turn it into a df and return
    '''
    if valid:
        df =  pd.read_csv("../data/properties.csv")
    else:
        df =  pd.read_csv("../data/raw_data.csv")

    return df

def preprocess(df):
    '''
    Takes a df, will go through all needed preprocessing steps and return the cleaned df.
    '''
    df = compress(df)
    df = convert_types(df)
    df = df.drop_duplicates()
    df = drop_cols_and_fill_nas(df)

    # Split the data to avoid data leakage
    X, y = split_df(df)

    # Different encoders
    X = encoding_df(X)

    # Normalising
    X_final = pd.DataFrame(MinMaxScaler().fit_transform(X))
    X_final.columns = X.columns


    return X_final, y

def compress(df, **kwargs):
    """
    Reduces the size of the DataFrame by downcasting numerical columns
    """
    input_size = df.memory_usage(index=True).sum()/ 1024**2
    print("old dataframe size: ", round(input_size,2), 'MB')

    in_size = df.memory_usage(index=True).sum()

    for t in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=t))

        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=t)

    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print("optimized size by {} %".format(round(ratio,2)))
    print("new DataFrame size: ", round(out_size / 1024**2,2), " MB")

    return df

def convert_types(df):
    '''
    Takes a df, convert all int8 columns into int16 type and returns that df
    '''
    df.fl_furnished = df.fl_furnished.astype("int16")
    df.fl_open_fire = df.fl_open_fire.astype("int16")
    df.fl_terrace  = df.fl_terrace .astype("int16")
    df.fl_garden = df.fl_garden.astype("int16")
    df.fl_swimming_pool = df.fl_swimming_pool.astype("int16")
    df.fl_floodzone = df.fl_floodzone.astype("int16")
    df.fl_double_glazing  = df.fl_double_glazing .astype("int16")

    return df

def drop_cols_and_fill_nas(df):
    '''
    Takes a df, will fill N/As or delete the column and return a df
    '''

    df = df.drop('id', axis='columns')
    df = df.drop('epc', axis='columns')
    df = df.drop('cadastral_income', axis='columns')
    df['surface_land_sqm'].fillna(df['surface_land_sqm'].mean(), inplace=True)
    df['construction_year'].fillna(2002, inplace=True)
    df = df.drop('primary_energy_consumption_sqm', axis='columns')
    df = df.drop('nbr_frontages', axis='columns')
    df = df.drop('latitude', axis='columns')
    df = df.drop('longitude', axis='columns')
    df['terrace_sqm'].fillna(df['terrace_sqm'].mean(), inplace=True)
    df['total_area_sqm'].fillna(df['total_area_sqm'].mean(), inplace=True)
    df['garden_sqm'].fillna(df['garden_sqm'].mean(), inplace=True)

    return df


def encoding_df(X):
    '''
    Takes and returns a df (X), will do all necessary encoding
    '''
    # One hot encoding
    cols = ['property_type', 'subproperty_type', 'region', 'province', 'heating_type']
    X = one_hot(X, cols)

    # Ordinal encoding: State building
    categories = ["AS_NEW", "JUST_RENOVATED", "GOOD", "TO_BE_DONE_UP", "TO_RENOVATE", "TO_RESTORE", "MISSING"]
    X = ordinal(X, categories, "state_building")

    # Ordinal encoding: kitchen
    categories = ["USA_HYPER_EQUIPPED", "HYPER_EQUIPPED", "USA_SEMI_EQUIPPED", "SEMI_EQUIPPED", "USA_INSTALLED", "INSTALLED", "USA_UNINSTALLED", "NOT_INSTALLED", "MISSING"]
    X = ordinal(X, categories, "equipped_kitchen")

    # Label encoding zip_code and locality
    label_encoder = LabelEncoder()

    # Encode labels in column 'species'.
    X['zip_code']= label_encoder.fit_transform(X['zip_code'])
    X['locality']= label_encoder.fit_transform(X['locality'])

    return X

def one_hot(X, cols):
    '''
    Takes and returns X after one_hot encoding. Also takes a list with features to encode.
    '''
    OH_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[cols]))
    OH_cols.index = X.index
    OH_cols.columns = OH_encoder.get_feature_names_out()
    X = X.drop(cols, axis=1)
    X = pd.concat([X, OH_cols], axis=1)

    return X

def ordinal(X, categories, feature):
    '''
    Takes X, a feature and a list of categories of that feature and ordinal encodes them. Returns X
    '''
    # Instantiate the Ordinal Encoder
    # print(X[categories])
    ordinal_encoder = OrdinalEncoder(categories = [categories])

    # ordinal_encoder = OrdinalEncoder(categories = [["AS_NEW", "JUST_RENOVATED", "GOOD", "TO_BE_DONE_UP", "TO_RENOVATE", "TO_RESTORE", "MISSING"]])


    # Fit it
    ordinal_encoder.fit(X[[feature]])

    # Transforming categories into ordered numbers
    X[feature] = ordinal_encoder.transform(X[[feature]])

    return X

def split_df(df):
    X = df.drop('price', axis='columns')
    y = df['price']

    return X, y
