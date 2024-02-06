import one_hot_encode
import ordinal_encoder
import target_encoder


if __name__ == '__main__':
    # # How many rows does the DataFrame have?
    # print(df.shape[0])
    # # How many columns does the DataFrame have?
    # print(df.shape[1])
    # # Are there any missing values in DataFrame?
    # print(df.isnull().values.any())
    # # What is the max number of rooms across the house dataset?
    # print(df['Room'].max())
    # # What is the mean are of the houses in the dataset?
    # print(df['Area'].mean())
    # # How many unique values does colum Zip_loc contain?
    # print(df['Zip_loc'].nunique())

    one_hot = one_hot_encode.report
    ordinal = ordinal_encoder.report
    target = target_encoder.report

    f1_score_onehot = round(one_hot['macro avg']['f1-score'], 2)
    f1_score_ordinal = round(ordinal['macro avg']['f1-score'], 2)
    f1_score_target = round(target['macro avg']['f1-score'], 2)

    print(f"OneHotEncoder: {f1_score_onehot}")
    print(f"OrdinalEncoder: {f1_score_ordinal}")
    print(f"TargetEncoder: {f1_score_target}")

