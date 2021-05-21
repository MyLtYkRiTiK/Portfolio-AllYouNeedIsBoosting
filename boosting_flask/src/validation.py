from datatable import dt, f, unique


def validate_train(user_csv_train, user_target, user_type, user_delete_numeric):
    try:
        dataframe = dt.fread(user_csv_train, fill=True,
                             na_strings=[''])
    except IOError:
        return "Please, check your train csv file - it can't be read", None, None

    try:
        dataframe[:, user_target]
    except KeyError:
        return "Please, check your target column name - it isn't in dataframe", None, None

    train_columns = list(dataframe.names)
    dataframe[:, user_target] = dataframe[:, dt.float32(f[user_target])]
    s, _ = dataframe[f[user_target] == None, user_target].shape
    if s != 0:
        return "Please, check dtype of your target column - it should be numeric", None, None

    if user_type != 'regression':
        unique_target_values = set(unique(dataframe[:, user_target]).to_dict( )[user_target])
        if user_type == 'binary':
            if len(unique_target_values) != 2:
                return "Please, check your target column - number of unique target values doesn't equal 2", None, None
            elif {0, 1} != unique_target_values:
                return "Please, check your target column - target values should be 0 and 1", None, None
        elif user_type == 'multiclassova':
            if len(unique_target_values) <= 2:
                return "Please, check your target column - number of unique target values " \
                       "should be more then 2", None, None
            elif set(range(len(unique_target_values))) != unique_target_values:
                return "Please, check your target column - target values should be from 0 to n-1, " \
                       "where n is number of classes", None, None

    train_columns.remove(user_target)

    delete_columns = set( )
    dataframe.replace(None, [-1, -1.0])
    for col in train_columns:
        dataframe[:, col] = dataframe[:, dt.float32(f[col])]
        s, _ = dataframe[f[col] == None, col].shape
        if s != 0:
            del dataframe[:, col]
            delete_columns.add(col)
    [train_columns.remove(col) for col in delete_columns]

    if len(train_columns) == 0:
        return f"Please, check check your dataset - you haven't any numeric columns", None, None

    if user_delete_numeric == 'no' and len(delete_columns) > 0:
        delete_columns = ', '.join(delete_columns)
        return f'Please, check next columns with non-numeric values or empty rows: {delete_columns}. ' \
               f'Or select process with deletion', None, None

    return None, train_columns, dataframe


def validate_test(user_csv_test, train_columns, user_target, user_delete_numeric='no'):
    try:
        dataframe = dt.fread(user_csv_test)
    except IOError:
        return "Please, check your test csv file - it can't be read", None

    try:
        dataframe[:, user_target]
        return "Please, check your test csv file - there is a target column in it", None
    except KeyError:
        pass

    try:
        dataframe[:, train_columns]
    except KeyError:
        return "Please, check your test columns - there is some difference in the train and test set of columns", None

    dataframe = dataframe[:, train_columns]
    delete_columns = set( )
    dataframe.replace(None, [-1, -1.0])
    for col in train_columns:
        dataframe[:, col] = dataframe[:, dt.float32(f[col])]
        s, _ = dataframe[f[col] == None, col].shape
        if s != 0:
            del dataframe[:, col]
            delete_columns.add(col)
    if user_delete_numeric == 'no' and len(delete_columns) > 0:
        delete_columns = ', '.join(delete_columns)
        return f'Please, check next columns with non-numeric values or empty rows: {delete_columns}.' \
               f' Or select process with deletion', None

    return None, dataframe


def validate_regression_metrics(user_type, user_regression_metrics, user_classification_metrics):
    if user_type == 'regression' and user_classification_metrics:
        return 'Please, with regression task use regression metrics'
    if user_type == 'regression' and not user_regression_metrics:
        return 'Please, select at least one regression metric'


def validate_classification_metrics(user_type, user_classification_metrics, user_regression_metrics):
    if (user_type == 'binary' or user_type == 'multiclassova') \
            and user_regression_metrics:
        return 'Please, with classification task use classification metrics'
    if (user_type == 'binary' or user_type == 'multiclassova') \
            and not user_classification_metrics:
        return 'Please, select at least one classification metric'


def validate(user_csv_train: object,
             user_target: str,
             user_type: str,
             user_delete_numeric: str,
             user_regression_metrics: [list, None],
             user_classification_metrics: [list, None],
             user_csv_test: object = None):
    error = validate_regression_metrics(user_type, user_regression_metrics, user_classification_metrics)
    if error:
        return error, None

    error = validate_classification_metrics(user_type, user_classification_metrics, user_regression_metrics)
    if error:
        return error, None

    error, train_columns, dataframe_train = validate_train(user_csv_train, user_target,
                                                           user_type, user_delete_numeric)
    if error:
        return error, None

    dataframe_test = None
    if user_csv_test:
        error, dataframe_test = validate_test(user_csv_test, train_columns, user_target, user_delete_numeric)
    if error:
        return error, None

    if user_type == 'regression':
        metrics = user_regression_metrics
    else:
        metrics = user_classification_metrics

    return None, (dataframe_train, dataframe_test, train_columns, user_target, metrics)
