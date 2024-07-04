import conllu

# Overriding the feats parser for the conllu reader
OVERRIDDEN_FIELD_PARSERS = {
    "feats": lambda line, i: conllu.parser.parse_nullable_value(
        line[i]
    )
}