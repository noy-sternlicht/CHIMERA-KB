import argparse
import os.path

from util import setup_default_logger
import pandas as pd


def domain_comparison(row):
    source_part = row['source_domain'].split('.')[0]
    target_part = row['target_domain'].split('.')[0]

    return source_part != target_part


def get_kg_basic_stats(all_relations):
    unique_nodes = set(all_relations['source_id'].unique()).union(set(all_relations['target_id'].unique()))
    nr_edges = len(all_relations)
    nr_comb_edges = len(all_relations[all_relations['relation'] == 'combination'])
    nr_insp_edges = len(all_relations[all_relations['relation'] == 'inspiration'])
    cross_domain_edges = all_relations.apply(domain_comparison, axis=1)
    nr_cross_domain_edges = len(all_relations[cross_domain_edges])
    cross_domain_combinations = all_relations[cross_domain_edges & (all_relations['relation'] == 'combination')]
    nr_cross_domain_combinations = len(cross_domain_combinations)
    cross_domain_inspirations = all_relations[cross_domain_edges & (all_relations['relation'] == 'inspiration')]
    nr_cross_domain_inspirations = len(cross_domain_inspirations)

    logstr = ''
    logstr += f'Inspiration edges & {nr_cross_domain_inspirations} & {nr_insp_edges} \\\\ \n'
    logstr += f'Blend edges  & {nr_cross_domain_combinations} & {nr_comb_edges} \\\\ \n'
    logstr += f'Edges  & {nr_cross_domain_edges} & {nr_edges} \\\\ \n'
    logstr += f'Nodes  & N/A & {len(unique_nodes)} \\\\ \n'

    return logstr


def get_node_domain_analysis(all_relations, include_other_nodes=False):
    unique_entities = set(all_relations['source_id'].unique()).union(set(all_relations['target_id'].unique()))
    entities_text = pd.read_csv(os.path.join(args.data_path, 'entities_text.csv'))
    entities_text = entities_text[entities_text['entity_id'].isin(unique_entities)]
    entities_per_domains = entities_text['domain'].value_counts()
    if not include_other_nodes:
        entities_per_domains = entities_per_domains[entities_per_domains.index != 'other']
    percentile_threshold = entities_per_domains.quantile(args.entity_count_percentile)
    entities_per_domains = entities_per_domains[entities_per_domains > percentile_threshold]

    logstr = ''
    row_length = 3
    table_row = ''
    curr_row_length = 0
    for domain, count in entities_per_domains.items():
        if curr_row_length == row_length:
            logstr += table_row[:-2] + '\\\\ \n'
            table_row = ''
            curr_row_length = 0
        table_row += f'{domain} & {count} & '
        curr_row_length += 1
    return logstr, entities_per_domains


def get_common_rel_edges(all_relations, rel_type, include_other_nodes=False):
    all_combinations = all_relations[all_relations['relation'] == rel_type]

    comb_res = pd.DataFrame(all_combinations)
    comb_data_processed = comb_res.groupby(['source_domain', 'target_domain']).size().reset_index(name='value')
    domain_data = comb_data_processed[['target_domain', 'source_domain', 'value']]

    if not include_other_nodes:
        domain_data = domain_data[domain_data['source_domain'] != 'other']
        domain_data = domain_data[domain_data['target_domain'] != 'other']

    percentile_threshold = domain_data['value'].quantile(args.pair_count_percentile)
    domain_data = domain_data[domain_data['value'] > percentile_threshold]
    domain_data = domain_data.sort_values(by=['value'], ascending=False)

    row_length = 1
    logstr = ''
    table_row = ''
    curr_row_length = 0
    for index, row in domain_data.iterrows():
        if curr_row_length == row_length:
            logstr += table_row[:-2] + '\\\\ \n'
            table_row = ''
            curr_row_length = 0
        table_row += f'{row["target_domain"]} & {row["source_domain"]} & {row["value"]} & '
        curr_row_length += 1

    return logstr, domain_data


def get_non_arxiv_inspiration_sources(all_relations):
    inspirations = all_relations[all_relations['relation'] == 'inspiration']
    inspirations = inspirations[inspirations['source_domain'] != 'other']
    inspirations = inspirations[inspirations['target_domain'] != 'other']
    inspirations = inspirations[inspirations['target_domain'].apply(lambda x: '.' not in x)]

    domain_data = inspirations.groupby(['source_domain', 'target_domain']).size().reset_index(name='value')
    domain_data = domain_data[['source_domain', 'target_domain', 'value']]

    percentile_threshold = domain_data['value'].quantile(args.pair_count_percentile)
    domain_data = domain_data[domain_data['value'] > percentile_threshold]

    return domain_data


def get_common_inspiration_sources_of_domains(all_relations,
                                              source_domains=['cs.ai', 'cs.ro', 'cs.cl', 'cs.cv', 'cs.lg']):
    inspirations = all_relations[all_relations['relation'] == 'inspiration']
    inspirations = inspirations[inspirations['source_domain'] != 'other']
    inspirations = inspirations[inspirations['target_domain'] != 'other']

    results = {}

    for source_domain in source_domains:
        domain_inspirations = inspirations[inspirations['source_domain'] == source_domain]
        domain_inspirations = domain_inspirations[domain_inspirations['target_domain'] != source_domain]
        domain_data = domain_inspirations.groupby(['source_domain', 'target_domain']).size().reset_index(name='value')
        domain_data = domain_data[['target_domain', 'value']]

        percentile_threshold = domain_data['value'].quantile(args.pair_count_percentile)
        domain_data = domain_data[domain_data['value'] > percentile_threshold]
        domain_data = domain_data.sort_values(by=['value'], ascending=False)
        results[source_domain] = domain_data

    return results



def main():
    data_path = args.data_path
    relations_path = os.path.join(data_path, 'raw_edges.csv')
    all_relations = pd.read_csv(relations_path)
    all_relations.dropna(subset=['publication_year'], inplace=True)
    all_relations = all_relations[all_relations['publication_year'] != "other"]
    all_relations['publication_year'] = all_relations['publication_year'].astype(int)

    stats_str = '\n'

    basic_stats = get_kg_basic_stats(all_relations)
    stats_str += '---Basic Stats:\n'
    stats_str += basic_stats

    node_domain_analysis, node_domains_df = get_node_domain_analysis(all_relations, True)
    stats_str += '---Node Domain Analysis:\n'
    stats_str += node_domain_analysis
    node_domains_output_path = os.path.join(args.output_path, 'node_domains.csv')
    node_domains_df.to_csv(node_domains_output_path)
    logger.info(f'Node domains saved to {node_domains_output_path}')

    common_combination_edges, domain_data = get_common_rel_edges(all_relations, 'combination')
    stats_str += '---Common Combination Edges:\n'
    stats_str += common_combination_edges
    common_combinations_output_path = os.path.join(args.output_path, 'common_combinations.csv')
    domain_data.to_csv(common_combinations_output_path, index=False)
    logger.info(f'Common combinations saved to {common_combinations_output_path}')

    common_inspiration_edges, domain_data = get_common_rel_edges(all_relations, 'inspiration')
    stats_str += '---Common Inspiration Edges:\n'
    stats_str += common_inspiration_edges
    common_inspirations_output_path = os.path.join(args.output_path, 'common_inspirations.csv')
    domain_data.to_csv(common_inspirations_output_path, index=False)
    logger.info(f'Common inspirations saved to {common_inspirations_output_path}')

    non_arxiv_inspiration_sources = get_non_arxiv_inspiration_sources(all_relations)
    stats_str += '---Non-Arxiv Inspiration Sources:\n'
    stats_str += non_arxiv_inspiration_sources.to_string(index=False)
    non_arxiv_inspiration_sources_output_path = os.path.join(args.output_path, 'non_arxiv_inspiration_sources.csv')
    non_arxiv_inspiration_sources.to_csv(non_arxiv_inspiration_sources_output_path, index=False)
    logger.info(f'Non-Arxiv inspiration sources saved to {non_arxiv_inspiration_sources_output_path}')

    common_inspiration_sources_of_domains = get_common_inspiration_sources_of_domains(all_relations)
    stats_str += '---Common Inspiration Sources of Domains:\n'
    for source_domain, domain_data in common_inspiration_sources_of_domains.items():
        stats_str += f'\n---{source_domain}:\n'
        stats_str += domain_data.to_string(index=False)
        common_inspiration_sources_output_path = os.path.join(args.output_path,
                                                              f'common_inspiration_sources_{source_domain}.csv')
        domain_data.to_csv(common_inspiration_sources_output_path, index=False)
        logger.info(f'Common inspiration sources for {source_domain} saved to {common_inspiration_sources_output_path}')
    logger.info(stats_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get data for sankey diagram')
    parser.add_argument('--data-path', type=str, help='Path to knowledge base')
    parser.add_argument('--output-path', type=str, help='Path to output')
    parser.add_argument('--min-year', type=int, help='Minimum year for papers')
    parser.add_argument('--max-year', type=int, help='Maximum year for papers')
    parser.add_argument('--pair_count_percentile', type=float, help='Minimum pair count for sankey diagram')
    parser.add_argument('--entity_count_percentile', type=float, help='Minimum entity count for sankey diagram')
    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)
    main()
