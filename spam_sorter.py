import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

columns = ['word_freq_make', 'word_freq_address', 'word_freq_all',
           'word_freq_3d', 'word_freq_our', 'word_freq_over',
           'word_freq_remove', 'word_freq_internet', 'word_freq_order',
           'word_freq_mail', 'word_freq_receive', 'word_freq_will',
           'word_freq_people', 'word_freq_report', 'word_freq_addresses',
           'word_freq_free', 'word_freq_business', 'word_freq_email',
           'word_freq_you', 'word_freq_credit', 'word_freq_your',
           'word_freq_font', 'word_freq_000', 'word_freq_money',
           'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
           'word_freq_650', 'word_freq_lab', 'word_freq_labs',
           'word_freq_telnet', 'word_freq_857', 'word_freq_data',
           'word_freq_415', 'word_freq_85', 'word_freq_technology',
           'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
           'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
           'word_freq_original', 'word_freq_project', 'word_freq_re',
           'word_freq_edu', 'word_freq_table', 'word_freq_conference',
           'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
           'char_freq_$', 'char_freq_#', 'capital_run_length_average',
           'capital_run_length_longest', 'capital_run_length_total', 'spam']

df = pd.read_csv('spambase_data.csv', names=columns)
X_ham = df[df['spam'] == 0]
X_spam = df[df['spam'] == 1]

# spam_column = df.pop('spam')
# x_ham = x_ham.drop('spam', axis=1)
# x_spam = x_spam.drop('spam', axis=1)
# y_ham = ['ham'] * len(x_ham)
# y_spam = ['ham'] * len(x_spam)
# print(len(x_ham)+len(x_spam))
# print(len(y_ham)+len(y_spam))

X_ham_train, X_ham_test = (train_test_split(
                           X_ham, test_size=0.4, random_state=101))

X_spam_train, X_spam_test = (train_test_split(
                             X_spam, test_size=0.4, random_state=101))


y_train = ['ham'] * len(X_ham_train) + ['spam'] * len(X_spam_train)
y_test = ['ham'] * len(X_ham_test) + ['spam'] * len(X_spam_test)
X_train = X_ham_train + X_spam_train
X_test = X_ham_test + X_spam_test


print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

model = make_pipeline(CountVectorizer(), MultinomialNB())

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


model.fit(X_train, y_train)

# print('Train score is:', model.score(X_train, y_train))
# print('Test score is:', model.score(X_test, y_test))

# predictions = model.predict(ham_test)
# print(list(predictions))
# predictions = model.predict(spam_test)
# print(list(predictions))
