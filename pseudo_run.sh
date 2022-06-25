#!/bin/bash
python fetch_data.py
python preprocess.py
python transform.py --module="EventTri"
python event_triplet.py
python entity_extraction.py
# python event_filter.py  # to-be-update
# python sentiment.py  # to-be-update
python neo4j_store.py