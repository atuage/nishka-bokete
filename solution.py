import pandas as pd
import numpy as np

import MeCab

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#import lightgbm as lgb
import optuna.integration.lightgbm as lgb
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    mecab = MeCab.Tagger("-Owakati")

    texts_wakati_train = [mecab.parse(text).strip() for text in train["text"].values]
    texts_wakati_test = [mecab.parse(text).strip() for text in test["text"].values]
    corpus = texts_wakati_train+ texts_wakati_test

    tfidf = TfidfVectorizer(max_features=250)

    tfidf.fit(corpus)
    result = tfidf.transform(texts_wakati_train).toarray()
    names = tfidf.get_feature_names_out()

    result_test = tfidf.transform(texts_wakati_test).toarray()

    for i in range(result.shape[1]):
        train["tf"+names[i]] = result.T[i]

    for i in range(result_test.shape[1]):
        test["tf"+names[i]] = result_test.T[i]

    countVec = CountVectorizer(max_features=100)
    countVec.fit(corpus)

    result = countVec.transform(texts_wakati_train).toarray()
    names = countVec.get_feature_names_out()

    result_test = countVec.transform(texts_wakati_test).toarray()

    for i in range(result.shape[1]):
        train["cVec"+names[i]] = result.T[i]

    for i in range(result_test.shape[1]):
        test["cVec"+names[i]] = result_test.T[i]

    cv = KFold(n_splits=5)

    scores=0
    models=[]

    y = train["is_laugh"]
    train.drop(["is_laugh","id","odai_photo_file_name","text"],inplace=True,axis=1)

    for train_index, test_index in cv.split(train):
        X_train, X_test = train.iloc[train_index], train.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        lgbm_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'auc',
            'num_leaves':80,
            'learning_rate': 0.05,
            "num_boost_round":3000,
            "early_stopping_rounds":200,
        }

        model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval,)
        models.append(model)
        
        y_pred =np.exp(model.predict(X_test, num_iteration=model.best_iteration))
        y_pred = np.where(y_pred < 0, 0, y_pred)
        print(roc_auc_score(y_test, y_pred))
        scores += roc_auc_score(y_test, y_pred)/5

    print(scores)


    for i in range(len(models)):
        sub_preds = model.predict(test[train.columns],num_iteration=model.best_iteration)

        submission = pd.DataFrame(sub_preds, columns=["is_laugh"])
    submission["id"] = test["id"]
    submission.reindex(columns=["id","is_laugh"]).to_csv("sub_text_lgb.csv", index = False)

if __name__ == "__main__":
    main()
