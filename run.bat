rem Step 1: Activate Anaconda
call "%userprofile%\anaconda3\Scripts\activate.bat" "%userprofile%\anaconda3"

rem Step 2: Activate the new environment
call conda activate searcher_new

rem Step 3: Run the app
call streamlit run Welcome.py

rem Step 4: Pause the script
pause
cmd /k