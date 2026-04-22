"""Script to submit parallel jobs for MLP HPO."""

# Take [configs].json and mlp_hpo_last_submitted_index.txt as input.
# For a specified num_jobs, submit the mlp_hpo_last_submitted_index to mlp_hpo_last_submitted_index+num_jobs
# configs in [configs].json using a job_runner.sh template script.