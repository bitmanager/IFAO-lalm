"""Derive inspection tables from the same immutable pass snapshot."""
from pathlib import Path


DERIVED_VIEWS = {'examples': 'review_examples', 'progress': 'review_progress'}


def derive_tables(passes_path, directory):
    import duckdb

    directory = Path(directory)
    with duckdb.connect() as db:
        db.execute("SET threads=1")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET preserve_insertion_order=false")
        db.execute("SET enable_progress_bar=false")
        db.execute("SET max_temp_directory_size='4GiB'")
        spill = directory / 'aggregate-spill'
        spill.mkdir()
        db.execute("SET temp_directory='" + str(spill).replace("'", "''") + "'")
        db.read_parquet(str(passes_path)).create_view('raw_passes')
        # The flat annotation column cannot distinguish JSON-looking strings
        # from objects. Provenance retains the original JSON type and JSON null.
        invalid = db.execute("SELECT count(*) FROM raw_passes WHERE "
            "source_provenance IS NOT NULL AND "
            "json_type(source_provenance) IS DISTINCT FROM 'OBJECT'").fetchone()[0]
        if invalid:
            raise ValueError('inspection_source_provenance_must_be_object')
        db.execute("CREATE VIEW review_passes AS SELECT * REPLACE ("
            "COALESCE(json_extract(source_provenance,'$.original_label'),"
            "json_extract(source_provenance,'$.source_label')) AS original_annotation) "
            "FROM raw_passes")
        # Use the repository's existing aggregate definitions, not a second set
        # of counts, flags or consensus rules maintained for the notebook.
        definitions = Path(__file__).with_name('review_views.sql').read_text()
        tables = {}
        for name, view in DERIVED_VIEWS.items():
            start = definitions.index('CREATE OR REPLACE VIEW ' + view + ' AS')
            end = definitions.index(';', start) + 1
            db.execute(definitions[start:end])
            path = directory / (name + '.parquet')
            literal = str(path).replace("'", "''")
            db.execute(f"COPY (SELECT * FROM {view}) TO '{literal}' "
                       "(FORMAT PARQUET, COMPRESSION ZSTD)")
            tables[name] = path
        return tables
