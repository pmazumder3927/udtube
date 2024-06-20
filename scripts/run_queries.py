import argparse
import sqlite3


def set_up_parser():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('input_db')
    parser_.add_argument('table_name')
    return parser_


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    with sqlite3.connect(args.input_db) as connection:
        cursor = connection.cursor()
        query = f"""   
        WITH totals AS (
            SELECT SUM(count) AS total_count
            FROM {args.table_name}
        ),
        all_freqs AS (
            SELECT
            *,
            (CAST(count AS FLOAT) / totals.total_count) AS abs_freqs
            FROM {args.table_name} CROSS JOIN totals
        ),
        lemma_probs AS (
            SELECT
            *,
            (CAST(SUM(count) AS FLOAT) / totals.total_count) AS lem_prob
            FROM {args.table_name} CROSS JOIN totals
            GROUP BY lemma
        ),
        feats_probs AS (
            SELECT
            *,
            (CAST(SUM(count) AS FLOAT) / totals.total_count) AS feats_prob
            FROM {args.table_name} CROSS JOIN totals
            GROUP BY feats
        ),
        final_probs AS (
            SELECT
            af.*,
            lf.lem_prob AS lemma_prob, 
            ff.feats_prob AS feature_prob
            FROM all_freqs af
            LEFT JOIN lemma_probs lf ON af.lemma = lf.lemma
            LEFT JOIN feats_probs ff ON af.feats = ff.feats
        )
        SELECT         
        *,   
        abs_freqs - (lemma_prob * feature_prob) AS absolute_divergence,
        log2(lemma_prob * feature_prob) - log2(lemma_prob) - log2(feature_prob) As log_odds_ratio
        FROM
        final_probs
        """
        res = cursor.execute(query)
        for r in res.fetchall():
            form, lemma, pos, feats, count, total_count, abs_freq, lem_prob, feats_prob, abs_divergence, log_odds = r
