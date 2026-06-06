-- migrate:up

INSERT INTO agents (id, name, description, role, tool_ids) VALUES
  ('sampler',
   'Sampler',
   'Discovers tables, samples column metadata, and loads controlled vocabulary annotations from hive connections.',
   'sampler',
   '["discover-tables","sample-metadata","load-annotations"]')
ON CONFLICT (id) DO NOTHING;

INSERT INTO agents (id, name, description, role, tool_ids) VALUES
  ('classifier',
   'Classifier',
   'Runs parallel classification methods against column feature vectors using cosine similarity, CatBoost, SVM, regex pattern detection, and feature extraction.',
   'classifier',
   '["classify-columns","run-classifiers","detect-patterns","extract-features"]')
ON CONFLICT (id) DO NOTHING;

INSERT INTO agents (id, name, description, role, tool_ids) VALUES
  ('evidence_fuser',
   'Evidence Fuser',
   'Converts classifier outputs into Dempster-Shafer mass functions, combines them via Dempster''s rule, and diagnoses high-conflict items.',
   'evidence_fuser',
   '["build-mass-functions","apply-dempster-rule","diagnose-conflicts"]')
ON CONFLICT (id) DO NOTHING;

INSERT INTO agents (id, name, description, role, tool_ids) VALUES
  ('synth_generator',
   'Synth Generator',
   'Creates deterministic synthetic training tables with representative column data for classifier bootstrapping.',
   'synth_generator',
   '["generate-synth-tables"]')
ON CONFLICT (id) DO NOTHING;

INSERT INTO agents (id, name, description, role, tool_ids) VALUES
  ('visualization_director',
   'Visualization Director',
   'Generates UMAP/t-SNE projections for Embedding Atlas, computes SAGE global feature importance, and produces per-item SHAP explanations.',
   'visualization_director',
   '["prepare-atlas-projection","compute-sage-importance","generate-shap-explanations"]')
ON CONFLICT (id) DO NOTHING;

-- migrate:down

DELETE FROM agents WHERE id IN (
  'sampler', 'classifier', 'evidence_fuser',
  'synth_generator', 'visualization_director'
);
