# Changelog

All notable changes to this project will be documented in this file.

The format is based on 'Keep a Changelog' and follows Semantic Versioning.

## Unreleased
- Minor README polish and repository housekeeping.
- CI/CD workflow consolidation across GitHub Actions, GitLab CI, and Jenkins.

## [0.3.0] - 2025-12-02
- Added Jenkins pipeline for end-to-end CI/CD:
  - Stages: setup, build ModelKit, validate, deploy Vertex AI pipeline, monitor pipeline job.
  - ModelKit versioning uses the Git commit hash for traceable builds.
- Improvements to build/push workflow for ModelKit container artifacts.
- Automation and pipeline monitoring enhancements.

## [0.2.0] - 2025-11-22
- Added GitHub Actions workflow for ML pipeline automation.
- Added GitLab CI configuration for ModelKit build/test/deploy.
- Implemented continuous training pipeline with conditional retraining and deployment logic.
- Added multi-stage KFP (Kubeflow Pipelines) pipeline for KitOps model training, comparison, and versioning.
- Added KFP pipeline optimization examples for caching, parallel processing, resource configuration, and efficient data loading.

## [0.1.0] - 2025-10-30
- Initial commit: basic project structure and README.
- Implemented core ML pipeline using KitOps and Vertex AI.

---

Notes:
- For release tagging and more detailed release notes (linking PRs and issues), we can extend this file to include links to PR numbers and contributors.
- If you prefer a different versioning scheme or want CHANGELOG.md placed under a docs/ directory, tell me and I can update it.
