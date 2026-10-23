# Reference Ledger

Store only redistributable metadata and reading notes here. Do not commit copyrighted PDFs unless their license explicitly permits redistribution.

Each literature claim should record title, authors, year, venue, DOI/arXiv identifier, access date, and the exact proposition it supports or challenges.

## Procedural references

- Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner. “Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.” arXiv:2609.00065, 2026. https://doi.org/10.48550/arXiv.2609.00065. Accessed 2026-10-24 from the current arXiv record. Used in R004A for the experimental-design requirement that replication and calibration be counted at the independent recording-group level rather than at the repeated frame or exported-fragment level; it is not evidence for AORTA's method claims.

## Implementation references

- OpenCV 4.13.0 calibration documentation, “Camera Calibration and 3D Reconstruction,” https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html, accessed 2026-10-24. Used only to specify the five-coefficient forward projection convention `[k1, k2, p1, p2, k3]` and to define `cv2.projectPoints` as the independent numerical oracle for R004B.
- Motional, `nuplan-devkit`, camera ORM schema at commit `e9241677997dd86bfc0bcd44817ab04fe631405b`, https://github.com/motional/nuplan-devkit/blob/e9241677997dd86bfc0bcd44817ab04fe631405b/nuplan/database/nuplan_db_orm/camera.py, accessed 2026-10-24. Used to confirm that intrinsic and distortion are stored as separate camera calibration fields; the schema alone does not establish coefficient semantics.
