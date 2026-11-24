import argparse
from project import load_split, prepare_models, compare, explain
parser = argparse.ArgumentParser()
parser.add_argument('--skip-search', action='store_true', help='Использовать параметры из отчёта без поиска')
args = parser.parse_args()
_, (X_train, X_test, y_train, y_test) = load_split()
models = prepare_models(X_train, y_train, tune=not args.skip_search)
table, best = compare(models, X_train, X_test, y_train, y_test)
print(table.to_string(index=False))
explain(models['Random Forest'], X_train)
