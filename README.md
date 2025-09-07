# activate environment (Git Bash)
source venv/Scripts/activate

# populate Mongo (choose ONE of these depending on your layout)
python -m src.populate_db
# OR
cd src && python populate_db.py

# run end-to-end benchmarks again (now with 7 features fixed)
python run_benchmarks.py

# run the pipeline demo
python main_pipeline.py

# start API (test /psi and /predict/<nic>)
pip install flask
flask --app app run


git add -A
git commit -m "Mongo integration: populate A/B hospital collections; keep 7-feature schema; PSI/MPC from DB"
git push origin main
