import tempfile
import os
import subprocess

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def evaluate_model(output_loc, actual, predicted, target_names):
    clf_report = classification_report(actual, predicted, target_names=target_names)
    print("\n[INFO] Saving classification report to: {}".format(output_loc + '/clf_report'))
    with open(output_loc + '/clf_report', 'a+') as f:
        f.write(clf_report)

    cnf_matrix = confusion_matrix(actual, predicted)
    fig = print_confusion_matrix(cnf_matrix, class_names=target_names)
    print("\n[INFO] Saving confusion matrix to: {}".format(output_loc + '/cnf_matrix.png'))
    fig.savefig(output_loc + '/cnf_matrix.png')


def write_to_disk(docs, write_loc):
    print("\n[INFO] Writing training data to: {}".format(write_loc))
    with open(write_loc, 'a+') as f:
        for doc in docs:
            f.write(doc)
            f.write('\n')


def cleanup_text(text_col):
    return text_col.replace(r'\n', ' ', regex=True)


def train_fasttext(train_data, data_loc, model_loc, args):

    write_to_disk(train_data, data_loc)

    args_list = ['fasttext', 'supervised', '-input', data_loc, '-output', model_loc]
    for k, v in args.items():
        args_list.append('-' + k)
        args_list.append(str(v))

    print("\n[INFO] Training classifier with command: {0}".format(" ".join(args_list)))
    subprocess.run(args_list)
    print("\n[INFO] Classifier successfully trained!")


def predict_fasttext(test_data, data_loc, model_loc, output_loc):

    write_to_disk(test_data, data_loc)

    args_list = ['fasttext', 'predict', model_loc + '.bin', data_loc]
    print("\n[INFO] Predicting classes with command: {0}".format(" ".join(args_list)))
    result = subprocess.run(args_list, stdout=subprocess.PIPE)
    predictions = result.stdout.decode("utf-8").replace('__label__', '').split('\n')
    print("\n[INFO] Classes successfully predicted")

    # Last row is an empty string (stdout)
    return predictions[:-1]

tmp_dir = tempfile.TemporaryDirectory().name
config = {
    'tmp_dir': tmp_dir,
    'train_data_loc': tmp_dir + '/train_data',
    'test_data_loc': tmp_dir + '/test_data',
    'test_pred_loc': tmp_dir + '/test_pred',
    'model_loc': tmp_dir + '/ftclf_model'
}

ft_args = {
    'lr': 0.05,
    'lrUpdateRate': 100,
    'dim': 300,
    'ws': 5,
    'epoch': 50,
    'neg': 5,
    'loss': 'ns',
    'thread': 3,
    'pretrainedVectors': '/home/dorian/Downloads/wiki-news-300d-1M.vec'
}


if not os.path.exists(config['tmp_dir']):
    os.makedirs(config['tmp_dir'])

print("\nRunning with config:")
for k, v in config.items():
    print('>', k, ":", v)

if __name__ == "__main__":

    # Load data
    news = fetch_20newsgroups()
    all_docs = pd.DataFrame({
        'data': news.data,
        'target': news.target,
        'label': [news.target_names[i] for i in news.target]
    })

    all_docs['cleaned_text'] = cleanup_text(all_docs['data'])
    all_docs = all_docs.assign(ft_format=lambda x: '__label__' + x.label + ' ' + x.cleaned_text)
    train_docs, test_docs = train_test_split(all_docs, test_size=0.2)

    train_fasttext(
        train_data=train_docs['ft_format'].tolist(),
        data_loc=config['train_data_loc'],
        model_loc=config['model_loc'],
        args=ft_args)

    predictions = predict_fasttext(
        test_data=test_docs['ft_format'].tolist(),
        data_loc=config['test_data_loc'],
        model_loc=config['model_loc'],
        output_loc=config['test_pred_loc'])

    print(predictions[0:5])
    print(predictions[-5:])

    # Calculate statistics
    pred_targets = [news.target_names.index(cls) for cls in predictions]
    evaluate_model(config['tmp_dir'], test_docs['target'].tolist(), pred_targets, news.target_names)