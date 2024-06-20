import conllu

# Overriding the parser for the conllu reader. This is needed so that feats can be read in as a str instead of a dict
OVERRIDDEN_FIELD_PARSERS = {
    "id": lambda line, i: conllu.parser.parse_id_value(line[i]),
    "xpos": lambda line, i: conllu.parser.parse_nullable_value(line[i]),
    "feats": lambda line, i: conllu.parser.parse_nullable_value(
        line[i]
    ),
    "head": lambda line, i: conllu.parser.parse_int_value(line[i]),
    "deps": lambda line, i: conllu.parser.parse_paired_list_value(line[i]),
    "misc": lambda line, i: conllu.parser.parse_dict_value(line[i]),
}