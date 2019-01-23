CREATE TABLE IF NOT EXISTS doctor (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS class (
  id INTEGER PRIMARY KEY,
  description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS net (
  id TEXT PRIMARY KEY,
  net TEXT NOT NULL,
  filename TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unit_annotation (
  unit_id INTEGER NOT NULL,
  net_id TEXT NOT NULL,
  doctor_id INTEGER NOT NULL,
  threshold REAL,
  descriptions TEXT,
  shows_concept INTEGER,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(doctor_id) REFERENCES doctor(id),
  PRIMARY KEY(unit_id, net_id, doctor_id)
);

CREATE TABLE IF NOT EXISTS image (
  id INTEGER PRIMARY KEY,
  image_path TEXT NOT NULL,
  ground_truth INTEGER NOT NULL,
  split TEXT NOT NULL,
  FOREIGN KEY(ground_truth) REFERENCES class(id)
);

CREATE TABLE IF NOT EXISTS image_unit_activation (
  net_id TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  unit_id INTEGER NOT NULL,
  class_id INTEGER NOT NULL,
  activation REAL NOT NULL,
  rank INTEGER NOT NULL,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(image_id) REFERENCES image(id),
  FOREIGN KEY(class_id) REFERENCES class(id),
  PRIMARY KEY(net_id, image_id, unit_id, class_id)
);

CREATE TABLE IF NOT EXISTS image_classification (
  net_id TEXT NOT NULL,
  image_id INTEGER NOT NULL,
  class_id INTEGER NOT NULL,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(image_id) REFERENCES image(id),
  FOREIGN KEY(class_id) REFERENCES class(id),
  PRIMARY KEY(net_id, image_id)
);

CREATE TABLE IF NOT EXISTS patch_unit_activation (
  net_id TEXT NOT NULL,
  patch_filename TEXT NOT NULL,
  unit_id INTEGER NOT NULL,
  class_id INTEGER NOT NULL,
  activation REAL NOT NULL,
  rank INTEGER NOT NULL,
  ground_truth INTEGER NOT NULL,
  FOREIGN KEY(net_id) REFERENCES net(id),
  FOREIGN KEY(class_id) REFERENCES class(id),
  FOREIGN KEY(ground_truth) REFERENCES class(id),
  PRIMARY KEY(net_id, patch_filename, unit_id, class_id)
);

INSERT INTO class (
  id,
  description
) VALUES (
  0, "normal"
), (
  1, "benign"
), (
  2, "cancer"
);
