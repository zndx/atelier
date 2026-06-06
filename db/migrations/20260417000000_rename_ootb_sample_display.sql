-- migrate:up

-- Rename the built-in sample source's display name from "OOTB Sample"
-- to "Sample" in the UI.  The 'ootb-sample' source_id stays as an
-- internal marker -- only the user-facing display_name changes.  The
-- WHERE clause is defensive so we don't overwrite a display_name that
-- an operator has already customized.
UPDATE data_sources
SET display_name = 'Sample'
WHERE id = 'ootb-sample' AND display_name = 'OOTB Sample';

-- Same treatment for any auto-seeded dataset row under that source.
UPDATE datasets
SET name = 'Sample v1'
WHERE source_id = 'ootb-sample' AND name = 'OOTB Sample v1';

-- migrate:down

UPDATE datasets
SET name = 'OOTB Sample v1'
WHERE source_id = 'ootb-sample' AND name = 'Sample v1';

UPDATE data_sources
SET display_name = 'OOTB Sample'
WHERE id = 'ootb-sample' AND display_name = 'Sample';
