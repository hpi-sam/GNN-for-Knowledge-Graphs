import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest


def run_analysis(distribution: dict, metric: str, title=""):
    fig, ax = plt.subplots(1, 1)
    ax.hist(distribution[metric])
    dist = np.array(distribution[metric])
    print(f'{title} - {metric}', 'mean', dist.mean(), 'var', dist.var())
    ax.set_title(f'{title} - {metric}')
    plt.show()


def run_analysis_combined(distribution_1, distribution_2, distribution_3, distribution_4, metric, title=""):
    fig, ax = plt.subplots(1, 1)
    ax.hist(distribution_1[metric], alpha=0.5, label='super_node')
    ax.hist(distribution_2[metric], alpha=0.5, label='average')
    ax.hist(distribution_3[metric], alpha=0.5, label='timeless')
    ax.hist(distribution_4[metric], alpha=0.5, label='query')
    ax.legend(loc='upper left')
    ax.set_title(f'{title} - {metric}')
    plt.show()


def main():
    combined = True

    with open('data/metrics/super_node.pickle', 'rb') as handle:
        super_node = pickle.load(handle)
    with open('data/metrics/average.pickle', 'rb') as handle:
        average = pickle.load(handle)
    with open('data/metrics/timeless.pickle', 'rb') as handle:
        timeless = pickle.load(handle)
    with open('data/metrics/query.pickle', 'rb') as handle:
        query = pickle.load(handle)

    if not combined:
        run_analysis(super_node, 'jaccard', title='super_node')
        run_analysis(super_node, 'cosine', title='super_node')
        run_analysis(super_node, 'ndcg', title='super_node')
        run_analysis(super_node, 'kendalltau', title='super_node')

        run_analysis(average, 'jaccard', title='average')
        run_analysis(average, 'cosine', title='average')
        run_analysis(average, 'ndcg', title='average')
        run_analysis(average, 'kendalltau', title='average')

        run_analysis(timeless, 'jaccard', title='timeless')
        run_analysis(timeless, 'cosine', title='timeless')
        run_analysis(timeless, 'ndcg', title='timeless')
        run_analysis(timeless, 'kendalltau', title='timeless')

        run_analysis(query, 'jaccard', title='query')
        run_analysis(query, 'cosine', title='query')
        run_analysis(query, 'ndcg', title='query')
        run_analysis(query, 'kendalltau', title='query')
    else:
        run_analysis_combined(super_node, average, timeless, query, 'jaccard', title='comparison')
        run_analysis_combined(super_node, average, timeless, query, 'cosine', title='comparison')
        run_analysis_combined(super_node, average, timeless, query, 'ndcg', title='comparison')
        run_analysis_combined(super_node, average, timeless, query, 'kendalltau', title='comparison')

    # kstest_result_sa = kstest(super_node['ndcg'], average['ndcg'])
    # kstest_result_sq = kstest(timeless['ndcg'], average['ndcg'])
    # kstest_result_st = kstest(super_node['ndcg'], timeless['ndcg'])
    # print(kstest_result_sa)
    # print(kstest_result_sq)
    # print(kstest_result_st)

    for metric in ['jaccard', 'cosine', 'ndcg', 'kendalltau']:

        kstest_result_sa = kstest(super_node[metric], average[metric])
        kstest_result_st = kstest(super_node[metric], timeless[metric])
        kstest_result_sq = kstest(super_node[metric], query[metric])
        kstest_result_at = kstest(average[metric], timeless[metric])
        kstest_result_aq = kstest(average[metric], query[metric])
        kstest_result_tq = kstest(timeless[metric], query[metric])

        print(f'kstest {metric}: super_node, average', kstest_result_sa)
        print(f'kstest {metric}: super_node, timeless', kstest_result_st)
        print(f'kstest {metric}: super_node, query', kstest_result_sq)
        print(f'kstest {metric}: average, timeless', kstest_result_at)
        print(f'kstest {metric}: average, query', kstest_result_aq)
        print(f'kstest {metric}: timeless, query', kstest_result_tq)


if __name__ == '__main__':
    main()
