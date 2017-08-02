# _*_ coding: utf-8 _*_

import os.path
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

base_dir = os.path.dirname(os.path.abspath(__file__))
path_sep = os.path.sep


def get_dir(abs_path):
    return os.path.dirname(os.path.abspath(abs_path))


def cwd(abs_path):
    return get_dir(abs_path) + path_sep


class BayesClassify(object):
    """

    """

    def __init__(self):
        self._clf = None
        self._tfidf = None
        self._naive_bayes_model_path = '%smodel%sbayes%snaive_bayes.model' % (cwd(__file__), path_sep, path_sep)
        self._tfidf_model_path = '%smodel%sbayes%stfidf.model' % (cwd(__file__), path_sep, path_sep)
        if os.path.exists(self._naive_bayes_model_path):
            self._clf = joblib.load(self._naive_bayes_model_path)
            self._tfidf = joblib.load(self._tfidf_model_path)
        else:
            print '用户Query分类模型不存在'

    def _load_data(self, train_data):
        _X_data = []
        _y_train = []
        for line in file(train_data):
            _X_y = line.split(' ')
            _X_data.append(' '.join(jieba.lcut(_X_y[0], cut_all=False)))
            _y_train.append(_X_y[1])
        return _X_data, _y_train

    def _select_feature(self, X_data):
        self._tfidf = TfidfVectorizer()
        _X_train = self._tfidf.fit_transform(X_data)
        return _X_train

    def _save_model(self):
        joblib.dump(self._tfidf, self._tfidf_model_path)
        joblib.dump(self._clf, self._naive_bayes_model_path)

    def _train(self, train_data):
        """
        模型训练方法，训练样本：{X y}
        y in [0, 1]，0为现象标签，1为原因标签
        例如：
        空调关不上 0
        半抬离合倒挡吱吱异响 0
        正常行驶2档、4档变速箱不好摘挡 0
        前轮轴承磨损 1
        玻璃密封条脏污 1
        换挡模块故障 1
        玻璃升降器松动 1
        :param train_data: 训练样本文件
        :return: None
        """
        _X_data, _y_train = self._load_data(train_data)
        _X_train = self._select_feature(_X_data)
        self._clf = MultinomialNB().fit(_X_train, _y_train)
        self._save_model()

    def _test(self):
        pass

    def predict(self, text):
        _cls = -1
        if self._clf:
            _feature = self._tfidf.transform(jieba.lcut(text, cut_all=False))
            _cls = int(self._clf.predict(_feature)[0])
        return _cls, 1  # todo 置信度1


bayes = BayesClassify()

if __name__ == "__main__":
    print bayes.predict('冷车行驶发动机金属摩擦声')
    print bayes.predict('发动进排气门油封损耗')
