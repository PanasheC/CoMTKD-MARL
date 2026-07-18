# Deploying CoMTKD-MARL to GitHub

The repository is prepared for the target remote:

```text
https://github.com/PanasheC/CoMTKD-MARL
```

## HTTPS deployment

Create the empty repository on GitHub without adding a README, license, or `.gitignore`, then run:

```bash
git init
git add .
git commit -m "Initial CoMTKD-MARL implementation"
git branch -M main
git remote add origin https://github.com/PanasheC/CoMTKD-MARL.git
git push -u origin main
```

GitHub may request a personal access token instead of an account password.

## SSH deployment

```bash
git init
git add .
git commit -m "Initial CoMTKD-MARL implementation"
git branch -M main
git remote add origin git@github.com:PanasheC/CoMTKD-MARL.git
git push -u origin main
```

## Deploy from the supplied Git bundle

The downloadable `.bundle` contains the initial commit and branch history. Clone it locally, then attach the GitHub remote:

```bash
git clone CoMTKD-MARL.bundle CoMTKD-MARL
cd CoMTKD-MARL
git remote rename origin bundle
git remote add origin https://github.com/PanasheC/CoMTKD-MARL.git
git push -u origin main
```

## Post-deployment verification

```bash
python -m pip install -r requirements.txt
make test
make smoke
```

The included GitHub Actions workflow repeats compilation, unit tests, and the end-to-end smoke test for every push and pull request.
