import os
import pickle
from datetime import datetime

import daal4py as d4p
from flask import Flask, request, render_template, send_file

from src.boosting_model import Trainer
from src.validation import validate

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
unique_name = datetime.strftime(datetime.now( ), "%Y%m%d%H%M%S-%f")
model_name = f'daal_model_{unique_name}'
prediction_name = f'prediction_{unique_name}'


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('mainPage.html')

    if request.method == 'POST':
        request_csv_train = request.files['csvTrain']
        request_csv_test = request.files['csvTest']
        request_target = request.form["targetColumn"]
        request_type = request.form["gridType"]
        request_balanced = request.form["gridBalance"]
        request_delete_numeric = request.form["gridNumeric"]
        request_regression_metrics = request.form.getlist("regressionMetrics")
        request_classification_metrics = request.form.getlist("classificationMetrics")
        validationError, train_parameters = validate(user_csv_train=request_csv_train,
                                                     user_csv_test=request_csv_test,
                                                     user_target=request_target,
                                                     user_type=request_type,
                                                     user_delete_numeric=request_delete_numeric,
                                                     user_regression_metrics=request_regression_metrics,
                                                     user_classification_metrics=request_classification_metrics,
                                                     )
        if validationError:
            return render_template('validationErrors.html',
                                   validationError=validationError)
        else:
            u_dataframe_train, u_dataframe_test, u_train_columns, u_user_target, u_metrics = train_parameters
            trainer = Trainer(dataframe_train=u_dataframe_train,
                              dataframe_test=u_dataframe_test,
                              train_columns=u_train_columns,
                              user_target=request_target,
                              metrics=u_metrics,
                              user_type=request_type,
                              user_balance=request_balanced)
            trainer.train( )
            daal_model = trainer.train_final( )
            cv_scores = trainer.calculate_cv_scores( )

            with open(f'{model_name}.pkl', 'wb') as out:
                pickle.dump(daal_model, out)
            model_address = f'http://95.217.131.11:60008/download_{model_name}'

            if request_csv_test:
                if request_type == 'regression':
                    lgbm_prediction = d4p.gbt_regression_prediction(
                    ).compute(u_dataframe_test.to_numpy( ), daal_model).prediction
                elif request_type == 'binary':
                    lgbm_prediction = d4p.gbt_classification_prediction(2
                                                                        ).compute(u_dataframe_test.to_numpy( ),
                                                                                  daal_model).prediction
                else:
                    lgbm_prediction = d4p.gbt_classification_prediction(trainer.params['num_class']
                                                                        ).compute(u_dataframe_test.to_numpy( ),
                                                                                  daal_model).prediction
                with open(f'{prediction_name}.csv', 'w') as file:
                    for l in lgbm_prediction:
                        file.write(','.join([str(x) for x in l]))
                        file.write('\n')

                prediction_address = f'http://95.217.131.11:60008/download_prediction_{prediction_name}'

                return render_template('resultPage_prediction.html',
                                       model_address=model_address,
                                       prediction_address=prediction_address,
                                       cv_scores=cv_scores)
            else:
                return render_template('resultPage.html',
                                       model_address=model_address,
                                       cv_scores=cv_scores)


@app.route(f'/download_{model_name}')
def downloadFile():
    path = f"{APP_ROOT}/{model_name}.pkl"
    return send_file(path, as_attachment=True)


@app.route(f'/download_prediction_{prediction_name}')
def downloadPrediction():
    path = f"{APP_ROOT}/{prediction_name}.csv"
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(threaded=True)
