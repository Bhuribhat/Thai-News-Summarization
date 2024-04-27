# Access tara

```bash
ssh -Y  bratanas@tara.nstda.or.th
```

## Main working directory

```bash
cd /tarafs/data/project/proj0183-ATS/finetune/lanta-finetune
```

## Upload & Download a file

Run this script in your local terminal

```bash
# Upload
scp -r <source_path> bratanas@tara.nstda.or.th:<destination_path>

# Download
scp -r bratanas@tara.nstda.or.th:<destination_path> <source_path>
```

## Submit a job

A job must be submitted in main working directory

```bash
sbatch <script_file>.sh
```

## View jobs status

```bash
myqueue
```

## Cancel a job

```bash
scancel <job_id>
```

## View output

```bash
cat slurm-<job_id>.out
```