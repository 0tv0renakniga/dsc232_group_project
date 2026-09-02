# Slurm Quick Reference

## Update Paths

* **Note:** Update the paths in the slurm scripts to point to your directories(rrogers --> yourusername).

```bash
# Update paths in slurm scripts
export LOCAL_SCRATCH=/expanse/lustre/projects/uci157/rrogers/temp
export SPARK_LOCAL_DIRS=/expanse/lustre/projects/uci157/rrogers/temp

cd /expanse/lustre/projects/uci157/rrogers

```

## Submit Jobs

```bash
# Step 1: Feature engineering (MUST run first)
sbatch run_fe.sh

# Step 2: Model training (run AFTER FE completes)
sbatch run_xgb.sh
sbatch run_classic.sh
sbatch run_corrected_stack.sh
```

## Auto-chain (models wait for FE)

* **Note:** This will run all models in parallel after FE completes.

* Don't worry about it running all the models, it's fine. It's just there if you want to run them all at once.

```bash
FE_ID=$(sbatch --parsable run_fe.sh)
sbatch --dependency=afterok:$FE_ID run_xgb.sh
sbatch --dependency=afterok:$FE_ID run_classic.sh
sbatch --dependency=afterok:$FE_ID run_corrected_stack.sh
```

## Monitor

```bash
squeue -u $USER                    # list your jobs
tail -f xgb_pipeline_<JOBID>.out   # live stdout
tail -f xgb_pipeline_<JOBID>.err   # live stderr
```

## Cancel

```bash
scancel <JOBID>                    # cancel one job
scancel <ID1> <ID2> <ID3>         # cancel multiple
scancel -u $USER                   # cancel ALL your jobs
```

## Output Files

| Script | stdout log | stderr log |
|---|---|---|
| `run_fe.sh` | `fe_pipeline_<JOBID>.out` | `fe_pipeline_<JOBID>.err` |
| `run_xgb.sh` | `xgb_pipeline_<JOBID>.out` | `xgb_pipeline_<JOBID>.err` |
| `run_classic.sh` | `classic_pipeline_<JOBID>.out` | `classic_pipeline_<JOBID>.err` |
| `run_corrected_stack.sh` | `cstack_pipeline_<JOBID>.out` | `cstack_pipeline_<JOBID>.err` |

## XGBoost Model Selection

Edit `ml_pipeline_xgb.py` line 65:
```python
XGB_MODEL = "XGB_Baseline"   # or "XGB_Tuned"
```
Only the selected model will train (saves memory).
