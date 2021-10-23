
def replace_str_to_num(col, X):
    for i, name in enumerate(X[col].unique()):
        X[col] = X[col].str.replace(name, f'{i}', regex=True)

    return X


def clean_table(df):
    df = df.replace({'<0\.\d*': '0'}, regex=True)
    df = df.replace({'-': '-1'}, regex=True)
    df["V"] = df["V"].str.split(",", expand=True, )
    df["SteelGradeId"] = replace_str_to_num("SteelGradeId", df)
    df["Name"] = replace_str_to_num("Name", df)
    return df
