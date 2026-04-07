from nltk.book import *
# Introductory Examples
import matplotlib.pyplot as plt

# text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# print(len(set(text3)) / len(text3))
# print(text3.count("smote"))
# # print(len(text3))
# # print(sorted(set(text3)))
# # print(len(set(text3)))
# # plt.show()
# def lexical_diversity(text):
#     return len(set(text)) / len(text) [2]


# def percentage(count, total):
#     return 100 * count / total

# saying = ['After', 'all', 'is', 'said', 'and', 'done', 'more', 'is', 'said', 'than', 'done']

# tokens = set(saying)

# print(tokens)
# tokens = sorted(tokens)
# print(tokens)
# print(tokens[-3:])

V = set(text1)

long_words = [w for w in V if len(w)>15]

print(sorted(long_words))

fdist5 = FreqDist(text5)
print(sorted(w for w in set(text5) if len(w) > 7 and fdist5[w] > 7))
