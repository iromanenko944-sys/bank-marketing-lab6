"""Восстановленный по отчёту проект. Все результаты вычисляются при запуске."""
from pathlib import Path
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)
ROOT = Path(__file__).resolve().parents[1]
NUM = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
CAT = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'

def download():
    target = ROOT / 'data' / 'bank-full.csv'
    if target.exists():
        return target
    target.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        content = response.read()
    def find_csv(blob):
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for name in archive.namelist():
                if Path(name).name == 'bank-full.csv':
                    return archive.read(name)
            for name in archive.namelist():
                if name.lower().endswith('.zip'):
                    result = find_csv(archive.read(name))
                    if result is not None:
                        return result
        return None
    csv = find_csv(content)
    if csv is None:
        raise RuntimeError('bank-full.csv не найден. Скачайте bank.zip с UCI и положите CSV в data/.')
    target.write_bytes(csv)
    return target

def load_split():
    frame = pd.read_csv(download(), sep=';')
    if len(frame) != 45211 or frame.shape[1] != 17:
        raise ValueError('Нужен полный bank-full.csv: 45211 строк, 17 столбцов.')
    X = frame[NUM + CAT].copy()  # duration намеренно исключён
    y = frame['y'].map({'no': 0, 'yes': 1})
    if y.isna().any():
        raise ValueError('Неожиданные метки целевой переменной.')
    return frame, train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

def preprocessing(scale=False):
    numeric = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        numeric.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(numeric), NUM),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CAT)])

def pipeline(model, scale=False):
    return Pipeline([('prep', preprocessing(scale)), ('model', model)])

class BalancedGB(ClassifierMixin, BaseEstimator):
    """Рассчитывает веса заново при каждом fit, включая CV-фолды."""
    def __init__(self, n_estimators=120, learning_rate=.1, max_depth=2, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
    def fit(self, X, y):
        self.model_ = GradientBoostingClassifier(n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, max_depth=self.max_depth, random_state=self.random_state)
        self.model_.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self
    def predict(self, X):
        return self.model_.predict(X)
    def predict_proba(self, X):
        return self.model_.predict_proba(X)

def trees():
    return {
        'Tree gini d=3': pipeline(DecisionTreeClassifier(max_depth=3, criterion='gini', class_weight='balanced', random_state=42)),
        'Tree entropy d=3': pipeline(DecisionTreeClassifier(max_depth=3, criterion='entropy', class_weight='balanced', random_state=42)),
        'Tree deep': pipeline(DecisionTreeClassifier(min_samples_leaf=10, class_weight='balanced', random_state=42))}

def cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def prepare_models(X, y, tune=True):
    models = trees()
    rf = pipeline(RandomForestClassifier(n_estimators=160, min_samples_leaf=5,
        class_weight='balanced', oob_score=True, random_state=42, n_jobs=1))
    gb = pipeline(BalancedGB())
    if tune:
        for name, estimator, grid in [
            ('Random Forest', rf, {'model__n_estimators': [80, 160], 'model__max_depth': [8, None]}),
            ('Gradient Boosting', gb, {'model__n_estimators': [60, 120], 'model__learning_rate': [.05, .1]})]:
            search = GridSearchCV(estimator, grid, scoring='roc_auc', cv=cv(), n_jobs=-1)
            search.fit(X, y)
            print(name, search.best_params_, 'CV AUC:', search.best_score_, flush=True)
            models[name] = search.best_estimator_
    else:
        models.update({'Random Forest': rf, 'Gradient Boosting': gb})
    lr = pipeline(LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42), scale=True)
    # Состав Voting восстановлен по описанию: RF + GB + логистическая регрессия.
    models['Voting'] = VotingClassifier(estimators=[('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']), ('lr', lr)], voting='soft', n_jobs=1)
    return models

def compare(models, X_train, X_test, y_train, y_test):
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    figs = ROOT / 'figures'; figs.mkdir(exist_ok=True)
    # Сначала CV и окончательное обучение всех моделей; test не участвует в выборе.
    scores = {}
    for name, model in models.items():
        print('CV:', name, flush=True)
        result = cross_validate(model, X_train, y_train, cv=cv(),
            scoring=['roc_auc', 'f1', 'recall'], n_jobs=-1)
        scores[name] = result
        model.fit(X_train, y_train)
    best_name = max(('Random Forest', 'Gradient Boosting', 'Voting'),
                    key=lambda n: scores[n]['test_roc_auc'].mean())
    rows = []
    for name, model in models.items():
        p = model.predict_proba(X_test)[:, 1]; pred = (p >= .5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        oob = np.nan
        if name == 'Random Forest':
            probabilities = model.named_steps['model'].oob_decision_function_
            valid = np.isfinite(probabilities).all(axis=1) & (probabilities.sum(axis=1) > 0)
            oob = roc_auc_score(np.asarray(y_train)[valid], probabilities[valid, 1])
        score = scores[name]
        rows.append({'model': name, 'CV ROC-AUC': score['test_roc_auc'].mean(),
            'CV std': score['test_roc_auc'].std(), 'CV F1': score['test_f1'].mean(),
            'CV Recall': score['test_recall'].mean(), 'OOB ROC-AUC': oob,
            'Train ROC-AUC': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
            'Test ROC-AUC': roc_auc_score(y_test, p), 'Test F1': f1_score(y_test, pred),
            'Test Recall': recall_score(y_test, pred), 'Test Precision': precision_score(y_test, pred, zero_division=0),
            'Test Accuracy': accuracy_score(y_test, pred),
            'G-mean': np.sqrt(tp/(tp+fn) * tn/(tn+fp)), 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
        (out / (name.replace(' ', '_') + '.txt')).write_text(
            classification_report(y_test, pred, digits=4, zero_division=0), encoding='utf-8')
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['no', 'yes'])
        plt.title(name); plt.tight_layout()
        plt.savefig(figs / (name.replace(' ', '_') + '_confusion.png'), dpi=150); plt.close()
    fig, ax = plt.subplots()
    for name in ['Random Forest', 'Gradient Boosting', 'Voting']:
        RocCurveDisplay.from_estimator(models[name], X_test, y_test, name=name, ax=ax)
    fig.tight_layout(); fig.savefig(figs / 'roc.png', dpi=150); plt.close(fig)
    table = pd.DataFrame(rows); table.to_csv(out / 'metrics.csv', index=False)
    print('Лучшая ансамблевая модель по CV:', best_name)
    return table, best_name

def explain(model, X_train):
    figs = ROOT / 'figures'; figs.mkdir(exist_ok=True)
    prep = model.named_steps['prep']; forest = model.named_steps['model']
    names = prep.get_feature_names_out()
    importance = pd.Series(forest.feature_importances_, index=names).sort_values()
    importance.tail(15).plot.barh(figsize=(9, 6)); plt.tight_layout()
    plt.savefig(figs / 'feature_importance.png', dpi=150); plt.show(); plt.close()
    import shap
    sample = prep.transform(X_train.sample(n=min(200, len(X_train)), random_state=42))
    explanation = shap.TreeExplainer(forest)(sample)
    values = explanation.values
    if values.ndim == 3:
        values = values[:, :, 1]
    shap.summary_plot(values, sample, feature_names=names, show=False)
    plt.tight_layout(); plt.savefig(figs / 'shap_summary.png', dpi=150); plt.show(); plt.close()
